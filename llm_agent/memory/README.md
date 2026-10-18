# Agent memory (runtime data, local only)

`llm_agent/memory/` is the persistent long-term memory of the DeepSeek game
agent.  It is **runtime data**, not source code, and is excluded from git.

Files created here by the agent:

- `memory.json` — structured memory: current goal + plan, places discovered,
  NPC dialogues, mechanics learned, items, battles, lessons, open questions.
- `journal.jsonl` — append-only chronological log of what the agent observed.
- `reports/memory_latest.md` — human-readable mirror for review.
- `memory.corrupt-<timestamp>.json` — backup if a memory file is ever corrupted.

Everything in this directory is produced from actual game observations only
(screenshots, in-game dialogue, attempts that failed).  It never contains ROM
data, RAM dumps, hidden flags or external walkthrough information.

Delete the contents of this directory to start a fresh agent that knows nothing.
