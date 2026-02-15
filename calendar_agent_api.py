import json
import os
import re
from datetime import datetime
from enum import Enum
from typing import Optional, List

import anthropic
import requests
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

app = FastAPI(title="Calendar Agent API")

# Get API key from environment variable
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")

# Ollama configuration
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")

anthropic_client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)


class Provider(str, Enum):
    anthropic = "anthropic"
    ollama = "ollama"


class AnthropicModel(str, Enum):
    claude_sonnet_old = "claude-3-5-sonnet-20241022"
    claude_sonnet_new = "claude-3-7-sonnet-20250219"


# Default settings
DEFAULT_PROVIDER = Provider.anthropic
DEFAULT_ANTHROPIC_MODEL = AnthropicModel.claude_sonnet_new
DEFAULT_OLLAMA_MODEL = "llama3.2:latest"

# Base URL for calendar API
CALENDAR_API_BASE = "https://calendar.andrewbrowne.org"


class CalendarRequest(BaseModel):
    message: str
    email: Optional[str] = None
    password: Optional[str] = None
    conversation_id: Optional[str] = None
    provider: Optional[str] = None  # "anthropic" or "ollama"
    model: Optional[str] = None  # specific model name


class CalendarResponse(BaseModel):
    response: str
    conversation_id: str
    completed: bool


# In-memory storage for conversation history
conversations = {}


def get_ollama_models() -> List[dict]:
    """Fetch available models from Ollama server"""
    try:
        response = requests.get(f"{OLLAMA_BASE_URL}/api/tags", timeout=5)
        response.raise_for_status()
        data = response.json()
        return data.get("models", [])
    except requests.exceptions.RequestException:
        return []


def call_ollama(model: str, messages: list, system_prompt: str = None) -> str:
    """Call Ollama API and return the response text"""
    # Convert messages to Ollama format
    ollama_messages = []

    # Add system prompt as first message if provided
    if system_prompt:
        ollama_messages.append({"role": "system", "content": system_prompt})

    for msg in messages:
        # Skip the first assistant message if it's the system prompt (old format)
        if msg["role"] == "assistant" and msg["content"].startswith("You are a meticulous calendar"):
            continue
        ollama_messages.append({"role": msg["role"], "content": msg["content"]})

    response = requests.post(
        f"{OLLAMA_BASE_URL}/api/chat",
        json={
            "model": model,
            "messages": ollama_messages,
            "stream": False,
        },
        timeout=120,
    )
    response.raise_for_status()
    return response.json()["message"]["content"]


def call_anthropic(model: str, messages: list) -> str:
    """Call Anthropic API and return the response text"""
    message = anthropic_client.messages.create(
        model=model, max_tokens=2048, messages=messages
    )
    return message.content[0].text


@app.get("/models")
async def list_models():
    """List all available models from both providers"""
    # Anthropic models (static list)
    anthropic_models = [
        {"name": m.value, "provider": "anthropic", "display_name": m.name.replace("_", " ").title()}
        for m in AnthropicModel
    ]

    # Ollama models (dynamic from server)
    ollama_raw = get_ollama_models()
    ollama_models = [
        {
            "name": m["name"],
            "provider": "ollama",
            "display_name": m["name"],
            "size": m.get("size"),
            "modified_at": m.get("modified_at"),
        }
        for m in ollama_raw
    ]

    return {
        "providers": ["anthropic", "ollama"],
        "default_provider": DEFAULT_PROVIDER.value,
        "default_model": {
            "anthropic": DEFAULT_ANTHROPIC_MODEL.value,
            "ollama": DEFAULT_OLLAMA_MODEL,
        },
        "models": {
            "anthropic": anthropic_models,
            "ollama": ollama_models,
        },
    }


def get_observation_message(
    iteration: int, response: str, session: dict
) -> tuple[int, str, bool]:
    """Process THINK/ACT response and execute tool calls"""
    observation_message = None
    is_final = False

    # Regex patterns for tool calls
    GET_DATETIME_REGEX = r'ACT:\s*get_current_datetime\(\)'
    LOGIN_REGEX = r'ACT:\s*login\((?:email=["\'](.+?)["\']\s*,\s*password=["\'](.+?)["\']\s*)?\)'
    GET_CALENDARS_REGEX = r'ACT:\s*get_calendars\(token=["\'](.+?)["\']\)'
    GET_EVENTS_REGEX = r'ACT:\s*get_events\(token=["\'](.+?)["\']\)'
    POST_EVENT_REGEX = r'ACT:\s*post_event\(token=["\'](.+?)["\']\s*,\s*calendar_id=["\'](.+?)["\']\s*,\s*title=["\'](.+?)["\']\s*,\s*start_time=["\'](.+?)["\']\s*,\s*end_time=["\'](.+?)["\']\s*,\s*description=["\'](.*)["\']\s*(?:,\s*reminder_minutes=\[([^\]]*)\])?\)'
    PATCH_EVENT_REGEX = r'ACT:\s*patch_event\(token=["\'](.+?)["\']\s*,\s*event_id=["\'](.+?)["\']\s*(?:,\s*title=["\']([^"\']*)["\'])?\s*(?:,\s*start_time=["\']([^"\']*)["\'])?\s*(?:,\s*end_time=["\']([^"\']*)["\'])?\s*(?:,\s*description=["\']([^"\']*)["\'])?\s*(?:,\s*reminder_minutes=\[([^\]]*)\])?\)'
    DELETE_EVENT_REGEX = r'ACT:\s*delete_event\(token=["\'](.+?)["\']\s*,\s*event_id=["\'](.+?)["\']\)'
    FINAL_ANSWER_REGEX = r'ACT:\s*final_answer\((?:answer=)?["\'](.+?)["\']\)'

    # Check for final_answer first
    final_match = re.search(FINAL_ANSWER_REGEX, response, re.DOTALL)
    if final_match:
        answer = final_match.group(1)
        observation_message = f"OBSERVE:\nTask completed successfully\n\nFinal Answer: {answer}"
        is_final = True
    else:
        # Check for get_current_datetime
        datetime_match = re.search(GET_DATETIME_REGEX, response, re.DOTALL)
        if datetime_match:
            now = datetime.now()
            # Format: ISO 8601 with day name and readable format
            iso_format = now.strftime("%Y-%m-%dT%H:%M:%S%z")
            readable_format = now.strftime("%A, %B %d, %Y at %I:%M %p")
            observation_message = f"OBSERVE:\nCurrent date and time: {iso_format}\n({readable_format})"
        else:
            # Check for login
            login_match = re.search(LOGIN_REGEX, response, re.DOTALL)
            if login_match:
                email = login_match.group(1) if login_match.group(1) else "andrewbrowne161@gmail.com"
                password = login_match.group(2) if login_match.group(2) else "Sierra-Ciara$"
                try:
                    result = requests.post(
                        f"{CALENDAR_API_BASE}/api/auth/login/",
                        json={"email": email, "password": password},
                        headers={"Content-Type": "application/json"},
                    )
                    result.raise_for_status()
                    response_data = result.json()
                    if "access" in response_data:
                        session["token"] = response_data["access"]
                        observation_message = f'OBSERVE:\n{{"token": "{response_data["access"]}", "message": "Login successful"}}'
                    else:
                        observation_message = f"OBSERVE:\n{response_data}"
                except requests.exceptions.RequestException as e:
                    observation_message = f"ERROR: Login failed - {str(e)}"
            else:
                # Check for get_calendars
                get_calendars_match = re.search(GET_CALENDARS_REGEX, response, re.DOTALL)
                if get_calendars_match:
                    token = get_calendars_match.group(1)
                    try:
                        result = requests.get(
                            f"{CALENDAR_API_BASE}/api/calendars/",
                            headers={
                                "Authorization": f"Bearer {token}",
                                "Content-Type": "application/json",
                            },
                        )
                        result.raise_for_status()
                        observation_message = f"OBSERVE:\n{json.dumps(result.json(), indent=2)}"
                    except requests.exceptions.RequestException as e:
                        observation_message = f"ERROR: Get calendars failed - {str(e)}"
                else:
                    # Check for get_events
                    get_events_match = re.search(GET_EVENTS_REGEX, response, re.DOTALL)
                    if get_events_match:
                        token = get_events_match.group(1)
                        try:
                            result = requests.get(
                                f"{CALENDAR_API_BASE}/api/events/",
                                headers={
                                    "Authorization": f"Bearer {token}",
                                    "Content-Type": "application/json",
                                },
                            )
                            result.raise_for_status()
                            observation_message = f"OBSERVE:\n{json.dumps(result.json(), indent=2)}"
                        except requests.exceptions.RequestException as e:
                            observation_message = f"ERROR: Get events failed - {str(e)}"
                    else:
                        # Check for post_event
                        post_event_match = re.search(POST_EVENT_REGEX, response, re.DOTALL)
                        if post_event_match:
                            token = post_event_match.group(1)
                            calendar_id = post_event_match.group(2)
                            title = post_event_match.group(3)
                            start_time = post_event_match.group(4)
                            end_time = post_event_match.group(5)
                            description = post_event_match.group(6)
                            reminder_minutes_str = post_event_match.group(7)
                            try:
                                event_data = {
                                    "calendar": calendar_id,
                                    "title": title,
                                    "start_time": start_time,
                                    "end_time": end_time,
                                    "description": description,
                                }
                                # Parse reminder_minutes if provided
                                if reminder_minutes_str:
                                    reminder_minutes = [int(x.strip()) for x in reminder_minutes_str.split(",") if x.strip()]
                                    event_data["reminder_minutes"] = reminder_minutes
                                result = requests.post(
                                    f"{CALENDAR_API_BASE}/api/events/",
                                    json=event_data,
                                    headers={
                                        "Authorization": f"Bearer {token}",
                                        "Content-Type": "application/json",
                                    },
                                )
                                result.raise_for_status()
                                response_data = result.json()
                                observation_message = f"OBSERVE:\n{json.dumps(response_data, indent=2)}"
                            except requests.exceptions.RequestException as e:
                                observation_message = f"ERROR: Create event failed - {str(e)}"
                        else:
                            # Check for patch_event
                            patch_event_match = re.search(PATCH_EVENT_REGEX, response, re.DOTALL)
                            if patch_event_match:
                                token = patch_event_match.group(1)
                                event_id = patch_event_match.group(2)
                                try:
                                    updates = {}
                                    if patch_event_match.group(3):
                                        updates["title"] = patch_event_match.group(3)
                                    if patch_event_match.group(4):
                                        updates["start_time"] = patch_event_match.group(4)
                                    if patch_event_match.group(5):
                                        updates["end_time"] = patch_event_match.group(5)
                                    if patch_event_match.group(6):
                                        updates["description"] = patch_event_match.group(6)
                                    if patch_event_match.group(7):
                                        updates["reminder_minutes"] = [int(x.strip()) for x in patch_event_match.group(7).split(",") if x.strip()]

                                    result = requests.patch(
                                        f"{CALENDAR_API_BASE}/api/events/{event_id}/",
                                        json=updates,
                                        headers={
                                            "Authorization": f"Bearer {token}",
                                            "Content-Type": "application/json",
                                        },
                                    )
                                    result.raise_for_status()
                                    response_data = result.json()
                                    observation_message = f"OBSERVE:\n{json.dumps(response_data, indent=2)}"
                                except requests.exceptions.RequestException as e:
                                    observation_message = f"ERROR: Update event failed - {str(e)}"
                            else:
                                # Check for delete_event
                                delete_event_match = re.search(DELETE_EVENT_REGEX, response, re.DOTALL)
                                if delete_event_match:
                                    token = delete_event_match.group(1)
                                    event_id = delete_event_match.group(2)
                                    try:
                                        result = requests.delete(
                                            f"{CALENDAR_API_BASE}/api/events/{event_id}/",
                                            headers={
                                                "Authorization": f"Bearer {token}",
                                            },
                                        )
                                        result.raise_for_status()
                                        observation_message = f"OBSERVE:\nEvent {event_id} deleted successfully"
                                    except requests.exceptions.RequestException as e:
                                        observation_message = f"ERROR: Delete event failed - {str(e)}"
                                else:
                                    observation_message = "ERROR: Invalid ACT format. Please use get_current_datetime(), login(), get_calendars(), get_events(), post_event(), patch_event(), delete_event(), or final_answer()"

    iteration += 1
    return iteration, observation_message, is_final


SYSTEM_PROMPT = """
You are a meticulous calendar assistant that can manage events in a multi-step process using tool calls and reasoning.

## Instructions:
- You will use step-by-step reasoning by
    - THINKING the next steps to take to complete the task and what next tool call to take to get one step closer to the final answer
    - ACTING on the single next tool call to take
- You will always respond with a single THINK/ACT message of the following format:
    THINK:
    [Carry out any reasoning needed to solve the problem not requiring a tool call]
    [Conclusion about what next tool call to take based on what data is needed and what tools are available]
    ACT:
    [Tool to use and arguments]
- As soon as you know the final answer, call the `final_answer` tool in an `ACT` message.
- ALWAYS provide a tool call, after ACT:, else you will fail.

## Available Tools

* `get_current_datetime()`: Get the current date and time to help with relative date calculations (e.g., "tomorrow", "next week")
* `login(email: str = "andrewbrowne161@gmail.com", password: str = "Sierra-Ciara$")`: Authenticate with the calendar API and get access token (credentials are optional, defaults provided)
* `get_calendars(token: str)`: Retrieve all calendars for the authenticated user
* `get_events(token: str)`: Retrieve all events for the authenticated user
* `post_event(token: str, calendar_id: str, title: str, start_time: str, end_time: str, description: str, reminder_minutes: list[int] = [15])`: Create a new calendar event with optional reminders (array of minutes before event, e.g., [5, 15, 60] for 5min, 15min, and 1hr reminders)
* `patch_event(token: str, event_id: str, title: str = "", start_time: str = "", end_time: str = "", description: str = "", reminder_minutes: list[int] = [])`: Update an existing event - only include fields you want to change
* `delete_event(token: str, event_id: str)`: Delete an existing event
* `final_answer(answer: str)`: Return the final answer to the user

## Important Notes:
- For datetime format, use ISO 8601 format: "YYYY-MM-DDTHH:MM:SSZ" (e.g., "2025-10-15T14:00:00Z")
- When the user mentions relative dates like "tomorrow", "next week", etc., call get_current_datetime() first to calculate the correct date
- You must login first to get a token before calling get_calendars, get_events, post_event, patch_event, or delete_event
- After login, get the calendar ID using get_calendars before creating events
- For patch_event, first get_events to find the event_id you want to update
- Keep the token from login response to use in subsequent API calls
- For reminders: use minutes (e.g., 15 = 15 minutes before, 60 = 1 hour before, 1440 = 1 day before)
"""


@app.post("/chat", response_model=CalendarResponse)
async def chat(request: CalendarRequest):
    """Chat with the calendar agent"""
    conversation_id = request.conversation_id or str(hash(request.message))

    # Determine provider and model
    provider = request.provider or DEFAULT_PROVIDER.value
    if provider == "ollama":
        model = request.model or DEFAULT_OLLAMA_MODEL
    else:
        model = request.model or DEFAULT_ANTHROPIC_MODEL.value

    # Initialize or retrieve conversation
    if conversation_id not in conversations:
        conversations[conversation_id] = {
            "messages": [{"role": "assistant", "content": SYSTEM_PROMPT}],
            "session": {"token": None},
            "provider": provider,
            "model": model,
        }

    conv = conversations[conversation_id]
    messages = conv["messages"]
    session = conv["session"]

    # Add credentials context if provided
    user_message = request.message
    if request.email and request.password:
        user_message = f"My credentials are: email={request.email}, password={request.password}. {user_message}"

    messages.append({"role": "user", "content": user_message})

    # Agent loop
    iteration = 1
    max_iterations = 20
    final_response = ""

    while iteration < max_iterations:
        try:
            # Call the appropriate provider
            if provider == "ollama":
                assistant_message = call_ollama(model, messages, SYSTEM_PROMPT)
            else:
                assistant_message = call_anthropic(model, messages)
            messages.append({"role": "assistant", "content": assistant_message})

            # Get observation
            iteration, obs, is_final = get_observation_message(
                iteration, assistant_message, session
            )

            if is_final:
                # Extract final answer
                final_match = re.search(
                    r"Final Answer: (.+)", obs, re.DOTALL | re.IGNORECASE
                )
                if final_match:
                    final_response = final_match.group(1).strip()
                else:
                    final_response = obs
                break
            else:
                # Continue with next tool call
                messages.append({"role": "user", "content": obs})

        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Error: {str(e)}")

    return CalendarResponse(
        response=final_response or "I wasn't able to complete the task.",
        conversation_id=conversation_id,
        completed=is_final,
    )


@app.post("/chat/stream")
async def chat_stream(request: CalendarRequest):
    """Chat with the calendar agent with streaming ReAct steps"""

    async def generate():
        conversation_id = request.conversation_id or str(hash(request.message))

        # Determine provider and model
        provider = request.provider or DEFAULT_PROVIDER.value
        if provider == "ollama":
            model = request.model or DEFAULT_OLLAMA_MODEL
        else:
            model = request.model or DEFAULT_ANTHROPIC_MODEL.value

        # Initialize or retrieve conversation
        if conversation_id not in conversations:
            conversations[conversation_id] = {
                "messages": [{"role": "assistant", "content": SYSTEM_PROMPT}],
                "session": {"token": None},
                "provider": provider,
                "model": model,
            }

        conv = conversations[conversation_id]
        messages = conv["messages"]
        session = conv["session"]

        # Add credentials context if provided
        user_message = request.message
        if request.email and request.password:
            user_message = f"My credentials are: email={request.email}, password={request.password}. {user_message}"

        messages.append({"role": "user", "content": user_message})

        # Send start event with provider info
        yield f"data: {json.dumps({'type': 'start', 'conversation_id': conversation_id, 'message': request.message, 'provider': provider, 'model': model})}\n\n"

        # Agent loop with streaming
        iteration = 1
        max_iterations = 20
        final_response = ""

        while iteration < max_iterations:
            try:
                # Call the appropriate provider
                if provider == "ollama":
                    assistant_message = call_ollama(model, messages, SYSTEM_PROMPT)
                else:
                    assistant_message = call_anthropic(model, messages)
                messages.append({"role": "assistant", "content": assistant_message})

                # Parse and stream THINK/ACT components
                think_match = re.search(r'THINK:(.*?)(?=ACT:|$)', assistant_message, re.DOTALL)
                act_match = re.search(r'ACT:(.*?)$', assistant_message, re.DOTALL)

                if think_match:
                    think_content = think_match.group(1).strip()
                    yield f"data: {json.dumps({'type': 'think', 'content': think_content, 'iteration': iteration})}\n\n"

                if act_match:
                    act_content = act_match.group(1).strip()
                    yield f"data: {json.dumps({'type': 'act', 'content': act_content, 'iteration': iteration})}\n\n"

                # Get observation
                iteration, obs, is_final = get_observation_message(
                    iteration, assistant_message, session
                )

                # Stream observation
                if obs:
                    yield f"data: {json.dumps({'type': 'observe', 'content': obs, 'iteration': iteration})}\n\n"

                if is_final:
                    # Extract final answer
                    final_match = re.search(
                        r"Final Answer: (.+)", obs, re.DOTALL | re.IGNORECASE
                    )
                    if final_match:
                        final_response = final_match.group(1).strip()
                    else:
                        final_response = obs

                    # Send complete event
                    yield f"data: {json.dumps({'type': 'complete', 'response': final_response, 'conversation_id': conversation_id, 'iterations': iteration})}\n\n"
                    break
                else:
                    # Continue with next tool call
                    messages.append({"role": "user", "content": obs})

            except Exception as e:
                yield f"data: {json.dumps({'type': 'error', 'message': str(e)})}\n\n"
                break

        if not final_response:
            error_msg = "I wasn't able to complete the task."
            yield f"data: {json.dumps({'type': 'complete', 'response': error_msg, 'conversation_id': conversation_id, 'iterations': iteration})}\n\n"

    return StreamingResponse(generate(), media_type="text/event-stream")


@app.get("/")
async def root():
    """Health check endpoint"""
    return {"status": "ok", "service": "Calendar Agent API"}


@app.delete("/conversation/{conversation_id}")
async def delete_conversation(conversation_id: str):
    """Delete a conversation"""
    if conversation_id in conversations:
        del conversations[conversation_id]
        return {"status": "deleted"}
    return {"status": "not_found"}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=4012)
