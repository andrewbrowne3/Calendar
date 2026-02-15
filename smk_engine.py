"""
SMK Engine — Progressive disclosure of Skills-Methods-Knowledge.

Like Claude Code's 3-tier skill loading:
  Tier 1 (always loaded):      Skill names + descriptions → agent knows what's available
  Tier 2 (loaded on demand):   Method steps + failure handling → loaded when skill is selected
  Tier 3 (loaded on demand):   Knowledge concepts → loaded alongside the method
"""

import json
from pathlib import Path


class SMKEngine:
    def __init__(self, smk_dir: str = "smk/"):
        self.smk_dir = Path(smk_dir)

        # Load all SMK files at startup (but only serve parts on demand)
        self.skills = self._load_json("skills.json")
        self.methods = self._load_json("methods.json")
        self.knowledge = self._load_json("knowledge.json")

        # Build lookup indexes
        self._skill_index = {s["name"].lower(): s for s in self.skills["skills"]}
        self._method_index = {m["name"].lower(): m for m in self.methods["methods"]}
        self._knowledge_index = {k["name"].lower(): k for k in self.knowledge["concepts"]}

    def _load_json(self, filename: str) -> dict:
        path = self.smk_dir / filename
        with open(path) as f:
            return json.load(f)

    # ── Tier 1: Base prompt (always loaded) ────────────────────────────

    def build_base_prompt(self) -> str:
        """
        Minimal system prompt with skill catalog + available tools + rules.
        Like Claude Code's Tier 1: skill name + description only.
        """
        skill_catalog = self._format_skill_catalog()
        tools_knowledge = self._format_concept("Available Tools")

        return f"""You are a meticulous calendar assistant that manages events using a structured reasoning process.

## Your Skills
{skill_catalog}

## Available Tools
{tools_knowledge}

## How You Work

You reason step-by-step using THINK/ACT/OBSERVE.

**CRITICAL: Your very first ACT must ALWAYS be load_skill_context().** You do NOT know the procedure steps until you load them. Pick the skill that matches the user's request and call:
  load_skill_context(skill_name)

You MUST NOT call login(), get_events(), or any other tool before calling load_skill_context() first. The loaded context will tell you exactly what steps to follow.

After loading context, follow the procedure steps using THINK/ACT/OBSERVE.

Always respond with exactly one THINK/ACT pair:
```
THINK:
[Your reasoning]
ACT:
[tool_call(args)]
```

### Rules
- Your FIRST action must ALWAYS be load_skill_context() — no exceptions
- ALWAYS provide a tool call after ACT:, else you will fail
- When done, call final_answer() with your response to the user
- For datetime format use ISO 8601: "YYYY-MM-DDTHH:MM:SSZ"
- When user mentions relative dates ("tomorrow", "next week"), call get_current_datetime() first
- You must login() before calling any calendar/event tools
- After login, get calendar_id via get_calendars() before creating events
- For patch_event, first get_events() to find the event_id
"""

    def _format_skill_catalog(self) -> str:
        """Tier 1: Just skill names and descriptions — enough to classify."""
        lines = []
        for skill in self.skills["skills"]:
            if not skill.get("parent"):
                lines.append(f"- **{skill['name']}**: {skill['description']}")
                for sub in self.skills["skills"]:
                    if sub.get("parent") == skill["name"]:
                        lines.append(f"  - {sub['name']}: {sub['description']}")
        return "\n".join(lines)

    # ── Tier 2+3: On-demand context loading ────────────────────────────

    def load_skill_context(self, skill_name: str) -> str:
        """
        Load the full method + relevant knowledge for a skill.
        Like Claude Code's Tier 2+3: full instructions + references.
        """
        skill = self._find_skill(skill_name)
        if not skill:
            # Fuzzy match
            skill_lower = skill_name.lower()
            for s in self.skills["skills"]:
                if skill_lower in s["name"].lower() or s["name"].lower() in skill_lower:
                    skill = s
                    break
            if not skill:
                available = ", ".join(s["name"] for s in self.skills["skills"])
                return f"Unknown skill: '{skill_name}'. Available skills: {available}"

        method = self._find_method(skill["method"])
        sections = []

        # Skill details
        sections.append(f"## Skill: {skill['name']}")
        sections.append(f"Description: {skill['description']}")
        sections.append(f"Inputs: {skill['inputs']}")
        sections.append(f"Expected output: {skill['outputs']}")
        sections.append(f"Precondition: {skill['precondition']}")
        sections.append(f"Postcondition: {skill['postcondition']}")

        # Method steps
        if method:
            sections.append(f"\n## Procedure: {method['name']}")
            sections.append(f"{method['description']}")
            sections.append("Steps:")
            for i, step in enumerate(method["steps"], 1):
                sections.append(f"  {i}. {step}")
            if method.get("failure_handling"):
                sections.append("Failure handling:")
                for condition, action in method["failure_handling"].items():
                    sections.append(f"  - If {condition}: {action}")

        # Relevant knowledge
        relevant_knowledge = self._get_relevant_knowledge(skill)
        if relevant_knowledge:
            sections.append("\n## Relevant Knowledge")
            for concept in relevant_knowledge:
                sections.append(f"### {concept['name']}")
                sections.append(f"{concept['description']}")
                for key, value in concept["properties"].items():
                    sections.append(f"  - {key}: {value}")

        sections.append("\nYou now have the procedure. Follow the steps above using THINK/ACT/OBSERVE.")

        return "\n".join(sections)

    def _get_relevant_knowledge(self, skill: dict) -> list:
        """Determine which knowledge concepts are relevant for a skill."""
        relevant = []
        skill_text = json.dumps(skill).lower()

        for concept in self.knowledge["concepts"]:
            name_lower = concept["name"].lower()
            if name_lower == "available tools":
                continue  # Already in base prompt
            if name_lower == "authentication flow" and "authenticat" in skill_text:
                relevant.append(concept)
            elif name_lower == "event" and any(w in skill_text for w in ["event", "create", "update", "delete", "read"]):
                relevant.append(concept)
            elif name_lower == "relative date resolution" and "datetime" in skill_text:
                relevant.append(concept)
            elif name_lower == "calendar api":
                relevant.append(concept)
        return relevant

    # ── Helpers ────────────────────────────────────────────────────────

    def _format_concept(self, name: str) -> str:
        concept = self._knowledge_index.get(name.lower())
        if not concept:
            return ""
        lines = [concept["description"]]
        for key, value in concept["properties"].items():
            lines.append(f"  - {key}: {value}")
        return "\n".join(lines)

    def _find_skill(self, name: str) -> dict | None:
        return self._skill_index.get(name.lower())

    def _find_method(self, name: str) -> dict | None:
        return self._method_index.get(name.lower())
