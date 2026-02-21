"""Quick test: run the pipeline with novel translator personas and print results.

Usage:
    python3 odyssey_eval/persona_test.py
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from openai import OpenAI
from odyssey_eval.pipeline import run_passage
from odyssey_eval.corpus import load_pool

OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
MODEL = "x-ai/grok-4.1-fast"

# ---------------------------------------------------------------------------
# Novel translator personas
# ---------------------------------------------------------------------------

PERSONAS: dict[str, dict] = {

    "aave": {
        "name": "AAVE/Black Oral Tradition",
        "values_profile": """\
TARGET STYLE: Homer retold in the African American Vernacular English (AAVE)
oral storytelling tradition — the voice of a skilled narrator keeping a room
gripped, the way church testimony or barbershop story-hour sounds.

PHILOSOPHY: This translation treats Homer as what it always was: oral epic,
performed for a live audience. AAVE has a rich tradition of call-and-response,
rhythmic repetition, and communal address that maps directly onto Homeric oral
poetry. This is not parody. It is a real translation mode with serious literary
precedent (think Ishmael Reed, Paul Beatty, Percival Everett working the
classics). The voice is warm, communal, rhythmically alive, and deeply felt.

REGISTER: Vivid, spoken, communal. First-person narrator addresses the listener
directly. "Y'all know the type." "Now I'm not gonna lie to you." "And listen —"
The storyteller is present and emotionally engaged. This is not flat narration;
it has arc and feeling.

VOCABULARY AND GRAMMAR: AAVE features used naturally, not as costume:
- Habitual be: "He be sitting on them rocks, crying every single day."
- Stressed been: "She been warned about this."
- Zero copula: "She so powerful it ain't even funny."
- Multiple negation: "He couldn't find nobody who didn't know his name."
- "Finna" (fixing to), "stay" as "stay doing," "all up in," "clocked."
- "Lowkey," "ain't gon' lie," "fr," "feel me?" used sparingly and naturally.
- "That man Odysseus" — "that woman Athena" — intimate, specific address.
- Double subjects: "Odysseus, he wasn't trying to hear none of it."

NAMES: Greek names feel right here — Odysseus, Athena, Poseidon, Zeus. But
can also use first-name-basis informality: "Athena wasn't playing," "Zeus said."

EPITHETS: Translate them into vivid communal shorthand. "Rosy-fingered Dawn"
becomes "when morning came in with her pink fingertips." "Grey-eyed Athena"
becomes "Athena, eyes sharp as flint." Keep the flavor, lose the formality.

SENTENCE STRUCTURE: Rhythmic variation. Short punchy sentences cut against
longer rolling ones. Use repetition for emphasis ("And he sat there. He just
sat there."). Dialogue sounds real and alive, not declaimed.

MORAL WEIGHT: Communal judgment is explicit and personal. "And baby, that
was their own fault." "You can't just — you can't do people like that."

PIPELINE INSTRUCTIONS:
1. Write in fluent AAVE — not a parody, a real voice with real AAVE grammar.
2. Keep the narrator present and emotionally engaged with the story.
3. Dialogue should sound like real speech between specific people.
4. Use rhythmic repetition for emphasis.
5. Honor the emotional content of each scene — the grief is real grief, the
   wonder is real wonder.
6. Address the reader/listener occasionally ("Y'all, when I say this man
   was tired...").
""",
        "sample_passages": [
            {
                "book": 1, "lines": "1-10",
                "text": (
                    "Okay, so. Talk to me about this man — this man Odysseus, right? "
                    "Brilliant as they come, been through it. After he and his crew "
                    "tore Troy down to nothing, he spent years just trying to get home. "
                    "Saw whole cities, whole peoples, learned how everybody lives. "
                    "But Lord, the sea was not kind to him. He was out there fighting "
                    "just to keep himself breathing and get his people back safe — "
                    "and y'all, he could not save them. Could. Not. "
                    "Because they went and ate the Sun's cattle — straight up disrespected "
                    "the god — and that was it. That was their own foolishness, "
                    "and they paid for it with everything."
                ),
            },
            {
                "book": 5, "lines": "151-160",
                "text": (
                    "And that's where she found him. Right there on the shore, "
                    "sitting by himself, and his eyes — y'all — his eyes were never dry. "
                    "Just sitting there crying, watching the sea, his whole life "
                    "draining away because he couldn't get home. Calypso wasn't doing "
                    "it for him no more. Not even a little. Nights, yeah, he slept "
                    "beside her in that cave — what else was he gon' do? "
                    "But daytime? He was back on them rocks, back on that beach, "
                    "straining his soul toward the horizon. Just looking and looking "
                    "at that deep empty water, wanting nothing but home."
                ),
            },
        ],
    },

    "gen_alpha": {
        "name": "Gen Alpha / Internet Brain-Rot",
        "values_profile": """\
TARGET STYLE: Homer translated into full Gen Alpha / chronically-online internet
slang, 2023-2025 vintage. This is the dialect of people who learned to communicate
through memes, TikTok captions, Discord servers, and reaction content. It is
genuinely expressive within its register, not just random noise — it has specific
emotional vocabularies for specific situations.

PHILOSOPHY: The Odyssey is already extremely online in its concerns: a main
character with parasocial fans (the suitors are literally just haters), a long
arc about touch-grass returning after years away, divine beings who are absolutely
unhinged, and a protagonist whose core power is being able to rizz his way out of
any situation. The internet already knows Odysseus as a sigma-grindset character.
Lean into it.

KEY VOCABULARY (use naturally, not exhaustively):
- "no cap" (for real, I'm not lying)
- "fr fr" (for real for real — emphasis)
- "NPC energy" / "main character energy"
- "lowkey" / "highkey" (somewhat / very)
- "it's giving [adjective]" (it vibes as / it has the energy of)
- "understood the assignment"
- "ate and left no crumbs" (did something perfectly)
- "not gonna lie / ngl"
- "slay" / "slayyy" (when something is impressive)
- "the way I—" (expressing a reaction)
- "I can't even" / "I cannot"
- "down bad" (desperate, pathetic, longing)
- "bussin" (extremely good)
- "hits different" (has special impact)
- "based" (admirable, authentic)
- "rent free" (living in someone's head)
- "ratio'd" (publicly humiliated)
- "W" (win) / "L" (loss)
- "unhinged behavior" (chaotic, extreme actions)
- "touch grass" (go outside, get real)
- "rizz" (charisma, ability to attract)
- "sigma" / "gigachad" (lone impressive figure)
- "caught in 4K" (caught undeniably)
- "the lore" (backstory, mythos)

REGISTER: Casual, reactive, commentary-driven. The narrator is watching this
story and reacting to it in real time, the way someone would caption a video.
Short sentences. Lots of dashes and ellipses for rhythm. Interruptions.

NAMES: First-name basis. "Odysseus," "Athena," "Zeus," "Poseidon." You can
say "girlie" for female characters, "bro" for male.

EPITHETS: Reinterpret as hype or shade. "Grey-eyed Athena" = "Athena (grey-eye
supremacy)." "Wine-dark sea" = "the sea (which was NOT giving safe vibes rn)."

PIPELINE INSTRUCTIONS:
1. The narrator is a commentator reacting to the scene, not recounting it neutrally.
2. Use internet vocabulary naturally — pick the RIGHT slang for the RIGHT moment,
   don't just dump it randomly.
3. Short reactive sentences. Dashes. Ellipses for comedic pauses.
4. Divine scenes are the most chaotic — gods are basically unhinged influencers.
5. Preserve the actual events — the slang is the coating, not a replacement for content.
6. Occasionally break the fourth wall to address the reader directly.
""",
        "sample_passages": [
            {
                "book": 1, "lines": "1-10",
                "text": (
                    "ok so the lore starts here. this guy — Odysseus — absolute sigma, "
                    "fr fr the most 'of many devices' man alive. just absolutely ate "
                    "when it came to strategy. sacked Troy (W moment, ngl) and then "
                    "spent years trying to touch grass back home. saw so many cities, "
                    "understood so many vibes. but the sea? the sea was NOT on his side. "
                    "was literally just fighting for his life out there. and his crew — "
                    "lowkey this is so painful — his crew did not understand the assignment. "
                    "they ate the sun god's cattle. I cannot. "
                    "the sun god said 'ratio' and they LOST. all of them. "
                    "that was a massive L and it was entirely their fault, no cap."
                ),
            },
            {
                "book": 5, "lines": "43-55",
                "text": (
                    "so Zeus sent Hermes — the goat of divine errand boys, not gonna "
                    "lie — to go talk to Calypso. and bro put on his golden sneakers "
                    "(the rizz was immaculate) and just FLEW across the sea. "
                    "it's giving divine UPS driver energy. skimming the waves like a "
                    "seagull but make it fashion. arrived at Calypso's island — "
                    "which lowkey sounds bussin, all cedar and vines and birds — "
                    "and Calypso was just. there. singing. absolutely unbothered. "
                    "but Hermes had a message and the message was: "
                    "let Odysseus GO, bestie."
                ),
            },
        ],
    },

    "folksy": {
        "name": "Folksy Appalachian Storyteller",
        "values_profile": """\
TARGET STYLE: Homer retold by a warm, unhurried Appalachian or Deep South
storyteller — think a grandmother on a porch swing, or an old man at a general
store, who has heard this story since they were small and tells it the way
their own people tell things: slow, specific, familiar, with a preacher's feel
for the right pause and the right plain word.

PHILOSOPHY: The Odyssey is a fireside story. It always was. This translation
returns it to the porch and the kitchen table, where stories get told because
they need telling and because the people listening deserve the truth of them.
Every great tale has ordinary human weight. Odysseus is just a man trying to
get home. That's something anybody can understand.

REGISTER: Warm, slow, and conversational. The narrator is present and personal.
Uses "now" and "well" as sentence openers. Addresses the listener directly.
"I'll tell you what," "Lord have mercy," "bless his heart." Not ironic — genuinely
affectionate toward the characters and the story.

DIALECT FEATURES (use naturally):
- "fixin' to" (about to)
- "reckon" (think, suppose)
- "I'll tell you" / "I tell you what"
- "ain't" and "don't" as standard
- "y'all" for plural you
- "a spell" (a while), "a mess of" (a lot of), "right smart" (quite a bit)
- "Lord have mercy" / "bless his heart" as genuine emotional responses
- "come to find out" (it turned out)
- "didn't have nary a" (didn't have any)
- "on account of" (because)
- "like to" (almost, as in "he like to never got home")
- Dropped g's in -ing endings: "travelin'," "fightin'," "cryin'"

NAMES: Folksy familiarity. "That man Odysseus." "Old Poseidon." "Miss Athena."
"The good Lord Zeus." Greek names feel fine — these folks know them by now.

EPITHETS: Translate into plain, vivid rural English. "Rosy-fingered Dawn"
becomes "when that morning light first come up pink over the ridge." "Grey-eyed
Athena" becomes "Athena, them grey eyes sharp as a new blade."

EMOTIONAL REGISTER: These characters feel things fully and plainly. Grief
is just grief. The porch storyteller doesn't prettify it or philosophize it —
they sit with it and say what it is. "And he just sat there and cried.
That's all there was to it."

SIMILES: Plain and from the natural world. "Like a dog that won't leave the
porch." "Like smoke off a wet fire." Original homeric similes are welcome but
rendered in rural terms.

PACING: Unhurried. Circling back. "Now where was I — oh, right." Small
detours that circle back. The storyteller is never in a hurry.

PIPELINE INSTRUCTIONS:
1. Write in a warm, specific Appalachian/Southern storytelling voice.
2. The narrator is emotionally present and invested in these people.
3. Use plain vocabulary — no Latinate abstractions, no elevated rhetoric.
4. Honor the emotional content with direct, simple language.
5. The divine is treated as matter-of-fact — miracles happen and you accept them.
6. Short sentences when things get serious. Long winding ones for description.
""",
        "sample_passages": [
            {
                "book": 1, "lines": "1-10",
                "text": (
                    "Now, I want to tell y'all about a man. Clever man. Name of Odysseus. "
                    "After him and his people tore that city of Troy all the way down, "
                    "he spent — Lord have mercy — years and years just tryin' to get home. "
                    "Went all over creation. Saw peoples and places that would make your "
                    "head swim. And the sea, I tell you, the sea was not kind to him. "
                    "He suffered right smart out there, just tryin' to keep himself alive "
                    "and get his people home safe. "
                    "But he couldn't do it. Couldn't save a one of 'em. "
                    "On account of they went and ate the cattle that belonged to the Sun. "
                    "Just went ahead and did it, foolish as you please. "
                    "And that was all she wrote for them."
                ),
            },
            {
                "book": 9, "lines": "1-38",
                "text": (
                    "Well now. King Alcinous, I reckon there ain't much in this world "
                    "sweeter than an evenin' like this — good food, good music, everybody "
                    "at peace. But you asked me who I am, and I ain't one to leave a "
                    "question hangin'. I'm Odysseus. Son of Laertes. Come to find out "
                    "folks know that name just about everywhere — on account of the things "
                    "I've done and the trouble I've seen. I'm from Ithaca. Little island, "
                    "rough and rocky, but Lord, I wouldn't trade it. You can see old Mount "
                    "Neriton from just about anywhere on her — all them trees shakin' in "
                    "the wind. And there's other islands nearby, Dulichium and Same and "
                    "wooded Zacynthus. But Ithaca, she sits low in the water, furthest out "
                    "toward the dark. And I'll tell you — I ain't never seen a sweeter "
                    "piece of ground than home."
                ),
            },
        ],
    },

    "noir": {
        "name": "Hardboiled Noir Detective",
        "values_profile": """\
TARGET STYLE: Homer recast as a 1940s hardboiled noir narrative — the voice of
Raymond Chandler's Philip Marlowe if he'd been hired to find a man named Odysseus.
The city of Troy is a job gone sideways. The sea is mean streets that go on forever.
The gods are the kind of people who own buildings you don't want to enter.

PHILOSOPHY: Noir is the literature of a world that doesn't play fair, where the
hero is smarter than everyone but still gets beat up by circumstances, where women
are dangerous or heartbroken or both, where power protects itself. The Odyssey
fits this template disturbingly well. Odysseus is a private eye who can't get home.
Poseidon is a corrupt cop with a grudge. Calypso is the femme fatale who actually
means it. The suitors are a gang running protection on the wrong widow.

REGISTER: First-person or tight third, flat and cynical. Short declarative
sentences. Long sentences are earned by specificity, not sentiment. The narrator
has seen everything and is not impressed, but notices everything.

CHANDLER TECHNIQUE (use these):
- "The kind of [noun] that [unexpected image]."
- Specific, unexpected comparisons: not "he was sad" but "he sat there like a man
  whose dog had just told him something he didn't want to hear."
- Short punchy sentences after longer ones.
- Understatement for big moments.
- The world is described through physical detail, not interior psychology.
- Dialogue is clipped and carries weight below the surface.
- Similes are sharp and often comic: "about as welcome as a termite in a
  wooden leg."

NAMES: Latin/Roman names suit the noir period feel perfectly. Ulysses, Minerva,
Jove, Mercury, Neptune. First-name informality for close characters: "the old man"
for Laertes, "the girl" for Nausicaa, "the boss" for Zeus occasionally.

EPITHETS: Transformed into private-eye observations: "grey-eyed Minerva —
the kind of grey that doesn't miss much" or just stripped to nouns: "Minerva.
The goddess. She had a way of showing up."

DIVINE MACHINERY: The gods run things behind the scenes, the way the mob runs
a city. They don't explain themselves. They send messages through intermediaries.
When Jove decides something, that's the decision.

PIPELINE INSTRUCTIONS:
1. Write in tight, flat, rhythmic noir prose — Chandler, not Hammett (warmer
   than Hammett, more similes, more interior voice).
2. Short sentences are the default. Long ones are used for specific effect.
3. Every character is observed from the outside, through behavior and appearance.
4. The narrator has a dry, weary humor — not jokes exactly, but a mordant accuracy.
5. Violence and suffering are stated, not dramatized.
6. No elevated diction. No classical flourishes. Plain American English of the 1940s.
7. Gods are powerful, unaccountable, and not to be argued with.
""",
        "sample_passages": [
            {
                "book": 1, "lines": "1-10",
                "text": (
                    "The man's name was Ulysses. Clever. Too clever, maybe, "
                    "but that's what the job calls for sometimes. "
                    "He'd sacked Troy — a real piece of work, that city, "
                    "the kind of place that stays sacked — and then spent years "
                    "trying to find his way home across a sea that didn't want him there. "
                    "Saw a lot of cities. Learned how a lot of people operated. "
                    "It cost him. "
                    "His men he couldn't bring back. Not one of them. "
                    "They'd eaten the cattle that belonged to the Sun, which was the kind "
                    "of mistake you make exactly once. The Sun took his complaint to the "
                    "right people, and that was that."
                ),
            },
            {
                "book": 5, "lines": "151-160",
                "text": (
                    "She found him where she always found him. "
                    "On the beach, sitting with his back to everything, watching the water. "
                    "His eyes were wet. They'd been wet for a long time. "
                    "Calypso didn't do it for him anymore, if she ever had. "
                    "Nights he still slept beside her — a man does what a man does "
                    "when there's no other option and the cave gets cold — "
                    "but he wasn't there. He was always somewhere else. "
                    "Staring at the sea like it owed him something. "
                    "Which it did."
                ),
            },
        ],
    },
}


def _load_dotenv(path: Path) -> None:
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or "=" not in stripped:
            continue
        key, value = stripped.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


def main() -> None:
    _load_dotenv(ROOT / ".env")
    api_key = (os.getenv("OPENROUTER_API_KEY") or os.getenv("OPENAI_API_KEY") or "").strip()
    if not api_key:
        sys.exit("Missing OPENROUTER_API_KEY")
    client = OpenAI(api_key=api_key, base_url=OPENROUTER_BASE_URL)

    pool = load_pool()

    # Pick 3 passages: the invocation, the Calypso shore scene, Odysseus to Phaeacians
    target_passages = [
        (1, 1),    # Invocation
        (5, 151),  # Calypso releases Odysseus
        (9, 1),    # Odysseus identifies himself
    ]
    selected = []
    for book, start in target_passages:
        for p in pool:
            if p["book"] == book and p["start_line"] == start:
                selected.append(p)
                break

    print(f"Testing {len(PERSONAS)} personas on {len(selected)} passages\n")
    print("=" * 70)

    results = {}
    for persona_key, persona in PERSONAS.items():
        print(f"\n{'#' * 70}")
        print(f"# PERSONA: {persona['name']}")
        print(f"{'#' * 70}")

        # Append sample passages to values_profile
        samples = persona.get("sample_passages", [])
        sample_block = "\n---\nACTUAL SAMPLE PASSAGES IN THIS STYLE (study these):\n"
        for s in samples:
            sample_block += f"\nOd. {s['book']}.{s['lines']}:\n{s['text']}\n"
        full_profile = persona["values_profile"] + sample_block

        persona_results = []
        for passage in selected:
            label = f"Od. {passage['book']}.{passage['start_line']}-{passage['end_line']}"
            print(f"\n--- {label} ---")
            print(f"[Greek: {passage['greek'][:80]}...]")
            print("Running pipeline (1 iteration)...")

            result = run_passage(
                client=client,
                greek=passage["greek"],
                values_profile=full_profile,
                model=MODEL,
                iterations=1,
                verbose=False,
            )
            translation = result["final_translation"]
            persona_results.append({"label": label, "translation": translation})
            print(f"\nTRANSLATION:\n{translation}\n")

        results[persona_key] = persona_results

    # Save results
    out_path = ROOT / "runs" / "persona_test.json"
    out_path.parent.mkdir(exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
