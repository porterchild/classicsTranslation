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


## Translation Feedback Mechanisms

The pipelines currently rely on LLM self-judgment (model scores its own output). The goal is to layer in external closed-loop feedback — deterministic or small-model signals that ground each quality axis independently, so the LLM judge becomes one input among several rather than the sole arbiter.

### Implemented

#### Perplexity Feedback (exact token scoring) — `translation_feedback_mechanisms.py`
`--sequential-feedback-model local_model` uses the local llama.cpp server (`LLAMACPP_BASE_URL`, default `http://localhost:8081`) as an external scorer.
- The scorer computes perplexity token-by-token for the exact candidate text.
- It uses llama.cpp `/completion` with `n_predict=0` and tokenized prefixes.
- It adaptively expands `n_probs` until the exact target token is found (up to full vocab), then accumulates true token logprobs.
- Sequential preflight fails fast if scorer availability is missing.

#### Perplexity Experiment (local_model)
Run artifacts:
- `runs/perplexity_sampling.log`
- `runs/perplexity_sampling.md`
- `runs/perplexity_sampling.json`

Sampled perplexity space (lower = more probable under the local scorer):

| Variant | Perplexity |
|---|---:|
| off_topic_fluent | 24.605 |
| wrong_but_fluent | 60.630 |
| archaic_style | 169.429 |
| smooth_readable | 235.014 |
| smooth_simpler | 294.813 |
| literal_awkward | 320.911 |
| calque_heavy | 417.300 |
| broken_english | 2063.697 |
| word_salad | 5822.976 |

Learnings:
- Perplexity is useful for **tiebreaking between candidate translations** when meaning is already close.
- Perplexity is useful for **red-flagging difficult sections** to allocate more revision effort (`~2000+` looked clearly broken).
- Perplexity provides **no correctness signal**: fluent but semantically wrong text can still score very low (`off_topic_fluent`, `wrong_but_fluent`).
- Practical policy: treat perplexity as a smoothness signal only, and pair it with explicit faithfulness checks (embedding / back-translation / relation checks).

#### Flesch-Kincaid Grade Level (readability) — `more_translation_feedback_mechanisms.py`
Pure Python formula based on syllable count, word count, and sentence count. Returns both grade level and reading ease score.
- **Strengths**: Completely deterministic, zero-cost, zero-dependency. Provides a hard sanity bound: if a "readable by a 7th grader" preference produces grade 16 output, something went wrong, and no amount of LLM self-judgment will catch that. Useful for gating iteration — stop revising when grade level hits the target band.
- **Weaknesses**: Crude. Counts syllables, not comprehension difficulty. "Utilize" and "use" differ by one syllable but vastly in register. Doesn't capture syntactic complexity, conceptual density, or domain-specific difficulty. A sentence full of short archaic words ("lo, the bard hath sung") scores as easy reading. Best as a floor/ceiling check, not a fine-grained quality signal.

#### Embedding Cosine Similarity (faithfulness proxy) — `more_translation_feedback_mechanisms.py`
Multilingual embedding model (qwen/qwen3-embedding-8b via OpenRouter) embeds Greek source and English translation into the same vector space; cosine similarity measures meaning preservation.
- **Strengths**: Completely external to the generating model. Cheap and fast. Strong floor detector: unrelated text scores ~0.06, wrong-paragraph text ~0.50, real translations 0.62–0.72. Also works as a tone filter — cutesy (0.58) and purple prose (0.59) both get penalized vs faithful translations (0.65–0.71).
- **Weaknesses**: Blind to subtle semantic errors. In testing, a translation that *reversed* the meaning ("myths cooperate" instead of "myths resist") scored 0.69 — nearly identical to faithful versions. Close-synonym substitutions ("examine" vs "purify", "archaeology" vs "ancient accounts") are invisible; the misread-archaeology variant actually scored *highest* overall (0.72). Model choice matters: `text-embedding-3-small` gave unusable scores (~0.31 for everything); `qwen3-embedding-8b` gave workable separation.
- **Phrase-level scoring**: Breaking paragraphs into aligned clauses and scoring each pair improves detection of entity-class swaps ("writers" vs "listeners" showed a visible dip) but still can't catch semantic inversions or close-synonym errors. The MIN across phrase scores is more useful than the average as an alarm signal.
- **Practical uses**: (1) **Tracking relative change over iterations** — absolute scores are noisy but direction of change (did similarity go up or down after a revision?) could flag regressions; needs more experimentation to interpret. (2) **Tiebreaking between candidates** — when two translations are otherwise close, higher similarity to source is a reasonable differentiator. (3) **Sanity checking / miswiring detection** — catches wrong-paragraph contamination, completely off-topic output, or garbled pipeline feeds (anything below ~0.55 warrants investigation). (4) **Effort allocation** — paragraphs or clauses with low phrase-level MIN scores identify difficult sections to escalate to a more intensive pipeline or to a human translator.
- **Cannot replace** structured faithfulness checking for subtle errors — those need back-translation or entity/relation extraction. Example: Greek says myths "stubbornly disdain plausibility and refuse mixture with the probable" (αὐθαδῶς τοῦ πιθανοῦ περιφρονῇ); a variant saying myths "readily accept being made plausible" scored 0.69, nearly identical to correct translations, because the same content words appear in both.

#### Back-Translation Divergence (faithfulness) — `translation_feedback_mechanisms.py`
Translates English back to Ancient Greek via LLM, then compares embedding similarity between back-translated Greek and original Greek (same-language comparison).
- **What it catches**: Omissions and truncations clearly. `drops_second_half` (translation covering only half the Greek) scored 0.59–0.63 vs 0.71–0.84 for full translations — clean, actionable gap. Reasoning mode improved separation: without reasoning `drops_second_half` scored 0.676; with reasoning 0.591.
- **What it misses**: Semantic inversions and synonym substitutions. `reverses_stubbornness` ("readily accepts" instead of "disdains") scored *higher* than faithful translations (0.80) because the same content words (πιθανόν, μῖξιν, εἰκός) survive the round-trip unchanged. `misread_archaeology` scored highest overall (0.84) because ἀρχαιολογίαν anchors both versions. The polarity-carrying verbs don't survive; the content nouns do.
- **Verdict**: Good floor/omission detector, not worth the cost (doubles LLM calls per candidate) as a per-iteration signal. Best used as a final sanity pass on the winning translation, not per-candidate per-iteration.

#### Entity/Relation Extraction (faithfulness) — `translation_feedback_mechanisms.py`
LLM extracts entities, relations, contrasts, and modal stance from Greek source (once per paragraph); substring word matching checks each item against candidate translations.
- **Extraction quality**: Good with reasoning. Correctly returns empty entities for paragraphs with no named people/places. Relations and modal stance are precise (e.g. "optative wish (εἴη) for ideal scenario; conditional subjunctive (ὅπου δ᾽ ἂν) leading to future necessity (δεησόμεθα)").
- **What it catches**: Omissions and truncations. `drops_second_half` scored 57% — three items from the conditional clause are simply absent. Legitimate gap.
- **What it misses**: Semantic inversions and entity swaps. `reverses_stubbornness` scores 100% because "plausible" satisfies "it arrogantly despises the plausible" and "mixture"/"probable" satisfy "it rejects the mixture with the probable." `entity_swap` ("writers" instead of "listeners") also scores 100% because common words satisfy every item.
- **Root cause**: The `any(word in translation)` matching checks if *any* content word from a relation phrase appears anywhere in the translation. Common content nouns (plausible, probable, mixture) appear in both faithful and inverted translations. Polarity-carrying verbs (despises, rejects) lose to nouns.
- **Next step for matching**: Filter stopwords and require predicate verbs to match, or replace substring matching with a targeted LLM yes/no check per relation item.

### Not Yet Implemented

#### Archaic Token Lookup (modernity)
Dictionary lookup against a wordlist of archaic/dated terms (unto, thereof, whilst, hath, etc.).
- **Strengths**: Trivial to implement, zero cost, completely deterministic. Catches the most obvious modernity failures instantly. Easy to extend the wordlist over time. Good as a hard gate: any archaic token in a "modern English" translation is a clear defect.
- **Weaknesses**: Only catches lexical archaism, not structural archaism ("it was to him apparent that..." is archaic in structure but uses no archaic words). Binary — a word is on the list or not, with no gradation. Doesn't distinguish between genuinely archaic usage and legitimate modern use of old-origin words. The wordlist requires curation and will never be complete.

#### N-gram Frequency Ratio (modernity)
Compare n-gram frequencies between a modern corpus (recent news/books) and a historical corpus. Phrases common in 1850s text but rare in 2020s text score low on modernity.
- **Strengths**: Captures structural and phrasal archaism that word lists miss. "It seemed not unreasonable" would score as dated even though each individual word is modern. Continuous rather than binary — gives a gradient of how modern/dated phrasing feels. Corpus-grounded rather than opinion-based.
- **Weaknesses**: Requires building or sourcing appropriate reference corpora. Domain mismatch: modern news corpora don't contain much classical-subject-matter text, so perfectly modern phrasing about ancient topics might score oddly. N-gram lookup infrastructure adds complexity. Short n-grams (bigrams, trigrams) may lack context; long n-grams become sparse.

### Considered and Rejected

#### Syntactic Depth (spacy parse tree depth)
Tree depth pointed the wrong direction in testing — a plain modern translation scored deeper (8) than Perrin's formal version (6) due to parser misanalysis. Depth doesn't distinguish well-managed complexity from confusing complexity, and spacy's accuracy on literary prose is unreliable. Flesch-Kincaid covers the readability check better with zero dependencies.

### Architecture Direction

The goal is a composite `ScoreCard` that runs all available mechanisms in parallel against each candidate:

```
candidate text
  ├─ embedding cosine similarity  → faithfulness (external, fast)
  ├─ back-translation divergence  → faithfulness (external, expensive)
  ├─ entity/relation check        → faithfulness (surgical, interpretable)
  ├─ exact local perplexity       → local smoothness signal (never a faithfulness metric)
  ├─ flesch-kincaid grade level   → readability (deterministic, floor/ceiling)
  ├─ archaic token count          → modernity (deterministic, lexical)
  ├─ n-gram frequency ratio       → modernity (corpus-grounded, phrasal)
  └─ LLM judge scores             → holistic (current approach, one voice among many)
```

The LLM judge still steers revision direction, but external signals ground whether iterations are actually moving on each axis. Pipelines can gate on external scores (stop early if they plateau, keep going if the LLM says "done" but Flesch-Kincaid says grade 14).

## Future Ideas
-one of the 'standards' to judge the fidelity of a translation against is whether the main observations/arguments/commentary from classic (and classical) commentary works can be reproduced from the new translation.
