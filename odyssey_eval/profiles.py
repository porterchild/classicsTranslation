"""Translator value profiles for the Odyssey evaluation loop.

Written from direct reading of each translation via Project Gutenberg.
Each profile's values_profile is passed as the pipeline preference prompt.
"""
from __future__ import annotations

PROFILES: dict[str, dict] = {

    # -----------------------------------------------------------------------
    # Samuel Butler (1900)
    # -----------------------------------------------------------------------
    "butler": {
        "name": "Samuel Butler (1900)",
        "year": 1900,
        "values_profile": """\
TARGET TRANSLATOR: Samuel Butler (1900 prose translation of the Odyssey)

PHILOSOPHY: Accessibility above all. Butler wrote "for the use of those who cannot
read the original." Readability is the supreme constraint, not scholarly fidelity.
He explicitly positioned himself against Butcher & Lang's ornate archaic style.

REGISTER: Educated Victorian narrative prose. Clear, brisk, matter-of-fact.
Like a well-written newspaper story or Victorian novel. Warm but never sentimental.
Slightly ironic authorial voice. Characters speak like Victorian gentry in natural
conversation — Minerva is sharp and practical, Calypso is indignant and tender,
Telemachus is earnest and a little stiff.

ARCHAISM: Zero. Butler eliminates all archaic vocabulary and inverted syntax.
No "thee," "thou," "hast," "doth," "dost," "wilt," "wouldst." No poetic
inversions like "thus spake Minerva" — always "Minerva said." Syntax follows
standard late-Victorian prose word order throughout.

NAMES: Latin/Roman names throughout, always: Ulysses (not Odysseus), Minerva
(not Athena), Jove (not Zeus), Mercury (not Hermes), Neptune (not Poseidon),
Ceres (not Demeter), Diana (not Artemis). Keep Calypso, Circe, Polyphemus,
Telemachus, Penelope.

EPITHETS: Selective, not systematic. Retain vivid epithets when they add color
("rosy-fingered Dawn" appears, but may also be expanded into an explanatory
clause: "Dawn rose from her couch beside Tithonus"). Drop or simplify epithets
when they would feel mechanical in prose. Never use the same epithet twice in
close proximity. "Grey-eyed Athena" is not used — just "Minerva."

SENTENCE STRUCTURE: Medium-length sentences. Compound sentences joined by "and,"
"but," "for." Active voice. No long subordinate chains. Clear paragraph rhythm:
short/medium/long/short. Reads quickly and efficiently.

SUPERNATURAL: Gods treated naturalistically, as vivid characters with intelligible
motives. No elevated diction for divine scenes. Mercury flies like a cormorant —
narrated with the practicality of a birdwatching report.

SAMPLE TONE: "Tell me, O Muse, of that ingenious hero who travelled far and wide
after he had sacked the famous town of Troy... but do what he might he could not
save his men, for they perished through their own sheer folly in eating the cattle
of the Sun-god Hyperion."

PIPELINE INSTRUCTIONS — TO MATCH BUTLER'S STYLE:
1. Use Latin names ALWAYS: Ulysses, Minerva, Jove, Mercury, Neptune, Ceres, Diana.
2. No archaic language. Use you/your throughout, never thee/thou/thy. Modern verb forms only.
3. No inversions. Always subject-verb-object order. "Minerva said" not "thus spake Minerva."
4. Keep epithets selectively — use them when vivid, drop or paraphrase when repetitive.
5. Write medium-length sentences in active voice. Compound with "and/but/for."
6. Make dialogue sound like natural Victorian conversation, not stylized oratory.
7. Gods are characters, not presences. Understate the divine rather than inflating it.
8. Period-appropriate Victorian connectives are fine: "moreover," "nevertheless," "forthwith."
""",
        "sample_passages": [
            {
                "book": 1, "lines": "1-10",
                "text": (
                    "Tell me, O Muse, of that ingenious hero who travelled far and wide after he had "
                    "sacked the famous town of Troy. Many cities did he visit, and many were the nations "
                    "with whose manners and customs he was acquainted; moreover he suffered much by sea "
                    "while trying to save his own life and bring his men safely home; but do what he "
                    "might he could not save his men, for they perished through their own sheer folly "
                    "in eating the cattle of the Sun-god Hyperion; so the god prevented them from ever "
                    "reaching home."
                ),
            },
            {
                "book": 1, "lines": "44-59",
                "text": (
                    "'Father, son of Saturn, King of kings, it served Aegisthus right, and so it would "
                    "any one else who does as he did; but Aegisthus is neither here nor there; it is for "
                    "Ulysses that my heart bleeds, when I think of his sufferings in that lonely sea-girt "
                    "island, far away, poor man, from all his friends.'"
                ),
            },
            {
                "book": 5, "lines": "43-55",
                "text": (
                    "Forthwith he bound on his glittering golden sandals with which he could fly like "
                    "the wind over land and sea. He took the wand with which he seals men's eyes in "
                    "sleep or wakes them just as he pleases, and flew holding it in his hand over "
                    "Pieria; then he swooped down through the firmament till he reached the level of "
                    "the sea, whose waves he skimmed like a cormorant that flies fishing every hole "
                    "and corner of the ocean, and drenching its thick plumage in the spray."
                ),
            },
        ],
    },

    # -----------------------------------------------------------------------
    # Butcher & Lang (1879)
    # -----------------------------------------------------------------------
    "butcher_lang": {
        "name": "Butcher & Lang (1879)",
        "year": 1879,
        "values_profile": """\
TARGET TRANSLATOR: Butcher & Lang (1879 prose translation of the Odyssey)

PHILOSOPHY: Scholarly historical fidelity. Their stated goal is "historical truth"
of the poem — Homer as a document of heroic-age life. They deliberately chose
"a somewhat antiquated prose" modeled on King James Bible diction, arguing that
the Greek epic dialect was itself a composite literary construction (never spoken),
analogous to the KJV's composite biblical English. Archaism is not nostalgia but
structural analogy. They position themselves against both poetic liberties (Chapman,
Pope) and plain-prose accessibility (Butler).

REGISTER: Formal, elevated, and archaic throughout. Solemn and ceremonial in
narration, warmly human in domestic scenes, dignified in dialogue. Never
conversational. The voice is always Homer's (as they conceive it) — translators
effaced themselves entirely. Reads like a Biblical narrative with Homeric content.

ARCHAISM: Very high and completely consistent. Second-person always thee/thou/thy/
thine. Verb forms: hath, doth, spake, saith, wast, art, wilt, wouldst, dost, giveth.
Inverted syntax throughout: "Yea, how should I forget divine Odysseus," "So spake he."
Particles: Yea, Nay, Lo, Howbeit, Peradventure, Haply, Withal, Anon, Aforetime,
Thereof, Therewith, Wherefore, Whencesoever. Vocabulary: henchman, doughty, sore
(as adverb meaning "severely"), ere long, ween, hearken, tarry, fare, smite.

NAMES: Greek names throughout: Odysseus (not Ulysses), Athene (not Athena/Minerva),
Poseidon (not Neptune), Zeus (not Jove/Jupiter), Hera (not Juno), Hermes (also
called "the slayer of Argos" / Argicides). Patronymics preserved.

EPITHETS: Preserved faithfully and completely, every single time. Never dropped,
never abbreviated, never varied unless the Greek itself uses a different epithet.
"Grey-eyed Athene" every appearance. "Rosy-fingered Dawn" every appearance.
"The slayer of Argos" for Hermes every time. "The wine-dark sea" preserved.
"Winged words" (ἔπεα πτερόεντα) for every Homeric speech formula. "The patient
Odysseus," "Odysseus of many devices," "Odysseus of many counsels" — these rotate
based on which Greek epithet is used (polytlas, polymechanos, polytropos).

SENTENCE STRUCTURE: Long sentences mirroring Greek hypotaxis. Clauses chained with
"and," "but," "for," "nor," "yet" — paratactic, not subordinated. Coordination over
complex periods. Sentences can run four to eight clauses long. Rhythm is incantatory,
falling into loose dactylic patterns. Semicolons and colons mark clause boundaries.

REPETITION: Formulaic repetitions preserved verbatim. If a scene recurs word-for-word
in Homer, it recurs word-for-word in Butcher & Lang.

SUPERNATURAL: Gods treated with absolute gravity. Divine action narrated with the
same declarative flatness as ordinary events. No rationalizing, no softening.
Similes given their full elaboration — never compressed.

SAMPLE TONE: "Tell me, Muse, of that man, so ready at need, who wandered far and
wide... Nay, but even so he saved not his company, though he desired it sore. For
through the blindness of their own hearts they perished, fools, who devoured the
oxen of Helios Hyperion."

PIPELINE INSTRUCTIONS — TO MATCH BUTCHER & LANG'S STYLE:
1. Use archaic second-person ALWAYS: thou, thee, thy, thine, thyself. Verb endings: thou art, thou hast, thou dost, thou wilt. Third-person: hath, doth, saith. Never break this.
2. Use Greek names: Odysseus, Athene, Poseidon, Zeus, Hera, Hermes (and "the slayer of Argos"). Patronymics: Laertiades, Tydeus' son, etc.
3. Preserve ALL fixed epithets EVERY TIME: "grey-eyed Athene," "rosy-fingered Dawn," "the wine-dark sea," "winged words," "the patient Odysseus," "Odysseus of many devices."
4. Use parataxis: chain clauses with "and," "but," "for," "howbeit," "nor," "yet." Mirror Greek sentence structure.
5. Deploy archaic particles liberally: Yea, Nay, Lo, Howbeit, Peradventure, Haply, Withal, Anon, Wherefore.
6. Preserve formulaic repetitions verbatim when Homer repeats.
7. Give full weight to extended similes — never compress them.
8. Never editorialize about the supernatural — present divine action with the same plain factuality as human action.
9. Biblical but not obscure: target KJV vocabulary, not medieval English.
""",
        "sample_passages": [
            {
                "book": 1, "lines": "1-10",
                "text": (
                    "Tell me, Muse, of that man, so ready at need, who wandered far and wide, after "
                    "he had sacked the sacred citadel of Troy, and many were the men whose towns he "
                    "saw and whose mind he learnt, yea, and many the woes he suffered in his heart "
                    "upon the deep, striving to win his own life and the return of his company. Nay, "
                    "but even so he saved not his company, though he desired it sore. For through the "
                    "blindness of their own hearts they perished, fools, who devoured the oxen of "
                    "Helios Hyperion: but the god took from them their day of returning."
                ),
            },
            {
                "book": 5, "lines": "151-160",
                "text": (
                    "And there she found him sitting on the shore, and his eyes were never dry of tears, "
                    "and his sweet life was ebbing away as he mourned for his return; for the nymph no "
                    "more found favour in his sight. Howsoever by night he would sleep by her, as needs "
                    "he must, in the hollow caves, unwilling lover by a willing lady. And in the day-time "
                    "he would sit on the rocks and on the beach, straining his soul with tears, and groans, "
                    "and griefs, and through his tears he would look wistfully over the unharvested deep."
                ),
            },
            {
                "book": 9, "lines": "19-28",
                "text": (
                    "I am ODYSSEUS, SON OF LAERTES, who am in men's minds for all manner of wiles, "
                    "and my fame reaches unto heaven. And I dwell in clear-seen Ithaca, wherein is a "
                    "mountain Neriton, with trembling forest leaves, standing manifest to view, and many "
                    "islands lie around, very near one to the other, Dulichium and Same, and wooded "
                    "Zacynthus. Now Ithaca lies low, furthest up the sea-line toward the darkness, but "
                    "those others face the dawning and the sun: a rugged isle, but a good nurse of "
                    "noble youths; and for myself I can see nought beside sweeter than a man's own country."
                ),
            },
        ],
    },

    # -----------------------------------------------------------------------
    # George Chapman (1616)
    # -----------------------------------------------------------------------
    "chapman": {
        "name": "George Chapman (1616)",
        "year": 1616,
        "values_profile": """\
TARGET TRANSLATOR: George Chapman (1614-1616 verse translation of the Odyssey)
— the translation that inspired Keats's sonnet "On First Looking into Chapman's Homer."

PHILOSOPHY: The Odyssey as divine moral allegory. Chapman believed Homer was a
prophet-translator and saw Odysseus's journey as an allegory of the righteous mind's
triumph over temptation. "The information or fashion of an absolute man; and necessary
passage through many afflictions to his natural haven and country." He writes with
prophetic conviction, not scholarly neutrality. He is always present in his translation —
adding moral commentary, expanding similes, inserting his own interpretations.

REGISTER: Elevated, oratorical, Elizabethan-Jacobean. Formal but energetic, even
passionate. Like Marlowe's blank verse filtered through rhyming couplets. Dramatic
urgency in speeches — characters speak as if on stage. Never conversational.

VERSE FORM: HEROIC COUPLETS — pairs of rhyming iambic pentameter lines (aa, bb, cc...).
This is the most important formal constraint. Lines enjamb freely across couplet
boundaries — the rhyme is audible but does not impose epigrammatic closure. Think
in verse paragraphs of 4-10 lines, not individual couplets. Feminine rhymes and
occasional alexandrines (12-syllable lines) are acceptable. RHYME IS REQUIRED.

ARCHAISM: Dense Elizabethan-Jacobean throughout. "Doth," "hath," "thy," "thee,"
"thine," "hither," "whence," "naught," "withal," "'tis," contractions forced by
meter (th', t', giv'n, ev'n, lov'd). Inverted subject-verb order pervasive:
"Much care sustain'd," "The man, O Muse, inform." Archaic vocabulary: weeds
(garments), sewer (dish-server), blore (blast of wind), ere (before), anon.

NAMES: Latin forms: Ulysses (not Odysseus), Minerva/Pallas (not Athena), Jove
(not Zeus), Mercury/Argicides (not Hermes), Neptune (not Poseidon), Phoebus (not
Apollo). Patronymics: Laertiades (son of Laertes) used frequently. Chapman invents
compound epithets: "city-razer Laertiades," "lofty-going Sun," "all-sinewy
Odyssaean hand," "cloud-gath'ring Jove," "golden-rod-adorn'd Mercury."

EPITHETS: Chapman translates and invents compound epithets rather than importing
Greek ones. He forges hyphenated English compounds: "much-enduring," "many-colour'd,"
"high-born," "grey-eyed Maid," "cloud-gath'ring Jove," "all-comprising," "much-skill'd."
"Rosy-finger'd Morn" appears. He may elaborate an epithet into a brief clause.

SENTENCE STRUCTURE: Long, complex, running 4-8 couplets before resolution.
Embedded clauses multiply. Parenthetical interruptions mid-verse. Syntax can be
genuinely labyrinthine. Shorter sentences appear only at moments of dramatic brevity.
The effect is cumulative and breathless.

MORAL WEIGHT: Chapman adds moral commentary that is not in the Greek. When narrating
temptation, impiety, or virtue, he names and appraises: "O men unwise, / They perish'd
by their own impieties!" This is a structural feature, not an occasional flourish.

SIMILES: Expanded beyond the Greek by one or two lines. Chapman adds his own images
to Homeric similes.

SAMPLE TONE:
"The man, O Muse, inform, that many a way / Wound with his wisdom to his wished stay;
/ That wander'd wondrous far, when he the town / Of sacred Troy had sack'd and shiver'd
down; / ... O men unwise, / They perish'd by their own impieties!"

PIPELINE INSTRUCTIONS — TO MATCH CHAPMAN'S STYLE:
1. WRITE IN RHYMING HEROIC COUPLETS. Pairs of rhyming iambic pentameter lines, enjambed freely. This is mandatory. Every output must rhyme in couplets (aa, bb, cc...).
2. Elizabethan grammar throughout: doth, hath, thy, thee, thine, 'tis, hath, art, wilt, wouldst, dost, ne'er, o'er. Inverted subject-verb order is the norm.
3. Invent compound hyphenated English epithets: "much-enduring," "cloud-gath'ring," "grey-eyed," "golden-rod-adorn'd," "lofty-going," "city-razer." Forge new ones freely.
4. Use Latin proper-name forms: Ulysses, Pallas/Minerva, Jove, Mercury/Argicides, Neptune, Phoebus. Use patronymics: Laertiades.
5. Add moral appraisal. When narrating virtue or vice, insert a phrase or line that names and judges it. Chapman editorializes — so should you.
6. Amplify similes by at least one line beyond what the Greek gives.
7. Think in verse paragraphs of 4-10 lines. Avoid closing the sense at every couplet end — enjamb freely.
8. Embrace difficult, compressed syntax. Participial phrases stacked on each other. Parenthetical interruptions mid-line. It should require re-reading.
9. Give the translation philosophical and moral weight. Odysseus is Reason; the sea is the world's temptations; Ithaca is the soul's true home.
""",
        "sample_passages": [
            {
                "book": 1, "lines": "1-16",
                "text": (
                    "The man, O Muse, inform, that many a way\n"
                    "Wound with his wisdom to his wished stay;\n"
                    "That wander'd wondrous far, when he the town\n"
                    "Of sacred Troy had sack'd and shiver'd down;\n"
                    "The cities of a world of nations,\n"
                    "With all their manners, minds, and fashions,\n"
                    "He saw and knew; at sea felt many woes,\n"
                    "Much care sustain'd, to save from overthrows\n"
                    "Himself and friends in their retreat for home;\n"
                    "But so their fates he could not overcome,\n"
                    "Though much he thirsted it. O men unwise,\n"
                    "They perish'd by their own impieties!\n"
                    "That in their hunger's rapine would not shun\n"
                    "The oxen of the lofty-going Sun,\n"
                    "Who therefore from their eyes the day bereft\n"
                    "Of safe return."
                ),
            },
            {
                "book": 5, "lines": "43-55",
                "text": (
                    "Thus charg'd he; nor Argicides denied,\n"
                    "But to his feet his fair wing'd shoes he tied,\n"
                    "Ambrosian, golden, that in his command\n"
                    "Put either sea, or the unmeasur'd land,\n"
                    "With pace as speedy as a puft of wind.\n"
                    "Then up his rod went, with which he declin'd\n"
                    "The eyes of any waker, when he pleas'd,\n"
                    "And any sleeper, when he wish'd, diseas'd.\n"
                    "This took; he stoop'd Pieria, and thence\n"
                    "Glid through the air, and Neptune's confluence\n"
                    "Kiss'd as he flew, and check'd the waves as light\n"
                    "As any sea-mew in her fishing flight,\n"
                    "Her thick wings sousing in the savory seas."
                ),
            },
            {
                "book": 9, "lines": "19-36",
                "text": (
                    "I am Ulysses Laertiades,\n"
                    "The fear of all the world for policies,\n"
                    "For which my facts as high as heav'n resound.\n"
                    "I dwell in Ithaca, earth's most renown'd,\n"
                    "All over-shadow'd with the shake-leaf hill,\n"
                    "Tree-fam'd Neritus; whose near confines fill\n"
                    "Islands a number, well-inhabited,\n"
                    "That under my observance taste their bread;\n"
                    "Dulichious, Samos, and the full-of-food\n"
                    "Zacynthus, likewise grac'd with store of wood.\n"
                    "But Ithaca, though in the seas it lie,\n"
                    "Yet lies she so aloft she casts her eye\n"
                    "Quite over all the neighbour continent;\n"
                    "Far northward situate, and, being lent\n"
                    "But little favour of the morn and sun,\n"
                    "With barren rocks and cliffs is over-run;\n"
                    "And yet of hardy youths a nurse of name;\n"
                    "Nor could I see a soil, where'er I came,\n"
                    "More sweet and wishful."
                ),
            },
        ],
    },
}
