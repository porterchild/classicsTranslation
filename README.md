# Classics Translation Experiment

Simple side-by-side comparison page for the opening of Plutarch's *Parallel Lives* (Theseus 1.1-1.3), including:
- Greek source text
- Literal English
- Legacy published translations
- AI quorum translations with different user preference prompts
- AI sequential-iteration translations with different user preference prompts

## Current Status

- `index.html` currently shows a 9-column comparison (Greek, literal, two legacy, and multiple AI outputs).
- Translation pipelines are modular in `main.py` (`debate`, `sequential`, `cognitive_user`, `cognitive_dualloop`).
- API key loading is programmatic: scripts read `.env` via `OPENROUTER_API_KEY` (fallback `OPENAI_API_KEY`).
- Run artifacts are stored under `runs/`; markdown result files can be committed, while `.log` files are ignored.
- JSON artifacts were removed; run outputs are markdown plus log files.

## Quick Start

```bash
./view
```

## Quorum Run

```bash
.venv/bin/python main.py --model x-ai/grok-4.1-fast --iterations 2 --verbose --output-prefix runs/quorum_translation --preference "Prioritize short, direct sentences with minimal archaic phrasing." > runs/quorum_run.log 2>&1
```

## Focused Paragraph Run

```bash
.venv/bin/python run_theseus_paragraph3.py --preference "This should be readable by a 7th grader." --iterations 4 --output-prefix runs/theseus_paragraph3 > runs/theseus_paragraph3.log 2>&1
```

## Cognitive Pipelines

```bash
.venv/bin/python main.py --pipeline cognitive_user --iterations 1 --output-prefix runs/cognitive_user_r1 --preference "This should be readable by a 7 year old." > runs/cognitive_user_r1.log 2>&1
```

```bash
.venv/bin/python main.py --pipeline cognitive_dualloop --iterations 1 --output-prefix runs/cognitive_dualloop_r1 --preference "This should be readable by a 7 year old." > runs/cognitive_dualloop_r1.log 2>&1
```

## Flow Chart

```text
+-----------------+
| Greek Paragraph |
+-----------------+
          |
          v
+-------------------------------------------+
| Shared Context                            |
| - goals guidance                          |
| - agent priorities                        |
| - user preference prompt                  |
+-------------------------------------------+
          |
          v
+-------------------------------------------+
| Initial Translation (parallel: 3 agents)  |
| input: Greek + goals + user preference +  |
|        own priority                       |
| output: observations + translation + self_scores |
+-------------------------------------------+
          |
          v
+-------------------------------------------+
| Debate (parallel: 3 agents)               |
| input: Greek + all current translations   |
|        + goals + user preference + own    |
|        priority                           |
| output: round_summary + critiques         |
|         + self_revision_plan              |
+-------------------------------------------+
          |
          v
+-------------------------------------------+
| Revision (parallel: 3 agents)             |
| input: Greek + own previous + all         |
|        translations + debate outputs      |
|        + goals + user preference + own    |
|        priority                           |
| output: revised translation +             |
|         change_summary + self_scores      |
+-------------------------------------------+
          |
          v
+---------------------+
| More Iterations?    |
+---------------------+
   | Yes        | No
   |            v
   +-----> [Back to Debate]
                |
                v
      +-------------------------------------+
      | Finalizer Agent                     |
      | input: Greek + final agent          |
      |        translations + agent/debate  |
      |        summaries + user preference  |
      | output: final_translation +         |
      |         justification +             |
      |         balance_scores              |
      +-------------------------------------+
                |
                v
      +-------------------------------------+
      | Write Files + Logs                  |
      | - runs/quorum_translation.md        |
      | - runs/quorum_run.log               |
      +-------------------------------------+
                |
                v
      +-------------------------------------+
      | Update quorum column in index.html  |
      +-------------------------------------+
```


## Future Ideas
-one of the 'standards' to judge the fidelity of a translation against is whether the main observations/arguments/commentary from classic (and classical) commentary works can be reproduced from the new translation.
