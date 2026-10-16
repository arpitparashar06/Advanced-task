# Manga transcription with consistent speakers (BYTE Applied ML, Task 2)

Given three consecutive manga pages, the task is to write out the story text in reading order and say who
speaks each line, with the same character keeping the same name on all three pages.

My final system uses Magiv2 to read the pages, then fixes the three things I saw it get wrong:
the same character gets a different name on every page, text floating in the air gets linked to the
nearest face, and narration gets linked to a character.

| text_order_score | balanced_joint_f1 | speaker_accuracy_on_matched | joint_f1 | matched_token_coverage |
|---|---|---|---|---|
| 0.929 | 0.510 (≈0.48 cross-validated) | 0.712 | 0.673 | 0.963 |

*80 development sequences. T, CAP and the narration classifier were picked on these same sequences,
so the cross-validated number is the honest one (section 3).*

**Run it:** `python predict.py --data dataset` → `predictions.jsonl` (images in, predictions out; section 6).

---

## 1. Final pipeline

Everything below runs in one go in `predict.py`, straight from the page images.

| Step | What it does | Built in | Tuned / trained on data? |
|---|---|---|---|
| 1 | Magiv2, as it comes: finds text, reads it, puts it in reading order, drops sound effects, links each text to a character box, groups the characters on each page, finds speech-bubble tails | `baseline magi.ipynb` | no |
| 2 | Fingerprints: Magiv2's own character embedder turns every character crop into 768 numbers | `Fingerprints/` | no |
| 3 | Is every text inside a bubble / box, or floating in the air? (OpenCV, no training) | `confirmed speaker/` | no |
| 4 | Narration probability of every line: a logistic regression on simple clues | `Narration/` | yes: trained on the dev labels |
| 5 | Confirmed-speaker rule: text inside a bubble, cloud or box keeps Magiv2's link; text in the air gets the last confirmed speaker | `confirmed speaker/` | no |
| 6 | Join Magiv2's per-page groups into characters (also across pages) while their average fingerprints are more alike than T; lines linked to nobody get the previous speaker; keep the CAP most talkative speakers | `Fingerprints/` | yes: T = 0.75, CAP = 3 |
| 7 | If the sequence seems to have a narrator, lines that look like narration become NARRATION | `Narration/` | yes: G = 0.6, K = 1, P = 0.15 |

The learned / tuned parts are saved in `grouping_settings.json` and `narration_model.json`.

---

## 2. How I got there

### Magiv2 as it is: good text, bad speakers

I started with Magiv2 ([ragavsachdeva/magiv2](https://huggingface.co/ragavsachdeva/magiv2)), a model made
for exactly this kind of manga transcription (`baseline magi.ipynb`, code from Magiv2's documentation).
Speaker = Magiv2's character group, text linked to nobody = NARRATION.
Its text was already good (text_order_score 0.941 on my 16-sequence validation split), but
balanced_joint_f1 was only **0.273** there (0.342 on all 80 dev sequences).

### Trying a VLM for everything

The obvious modern answer was to give the pages to a vision-language model and fine-tune it.
I followed the Hugging Face cookbook (TRL, 4-bit QLoRA) for Qwen2-VL-7B and trained it on 64 dev sequences
to write the whole answer: every line, in order, with its speaker (`QWEN for everything/qwen-only.ipynb`).
Before training it did not even follow the format. After training it scored **0.042**
balanced_joint_f1 and only 0.198 text_order_score on the 16 validation sequences: to fit three pages on a
T4 they had to be shrunk to 640 px, too small to read the text, and it called almost every line NARRATION.

### A VLM only for the speakers

Since Magiv2 already reads the text well, I kept Magiv2's lines and asked Qwen only who says each one
(`qwen with magi support/qwen-with-magi-support.ipynb`). The first try scored 0.207: Qwen could not tell
which line was which bubble. So I drew each line's number in a red box on the page, where Magiv2 found the
text. That gave **0.264**, still below Magiv2 alone (0.273). The answers often did not follow the
"number: speaker" format, so many lines fell back to the previous speaker.

With only 64 training sequences, fine-tuning a 7B VLM was not going to learn who is who, so I stepped back
and looked closely at Magiv2 itself.

### Looking at what actually goes wrong

On a Kaguya-sama page (`confirmed speaker/confirmed-chars.ipynb`) three things were wrong:

1. The same girl is two different groups on one page, and on the next page the group numbers start
   again. Magiv2 only groups characters inside one page.
2. Kaguya's thoughts float in the air next to the boy, and Magiv2 gives them to the boy.
3. The narration boxes ("Shinomiya thought for an instant...") are linked to a face. Magiv2 has no
   narration output at all: on the dev set it links 72 of the 80 narration lines to some character.

### Fix 1: fingerprints

Reading the Magiv2 paper and code, I found that it has a character embedder inside (it uses it to name
characters when you give it pictures of them). It works as a fingerprint: two boxes from the same Magiv2
group are 0.85 alike on average, boxes from different groups 0.48. I gave every Magiv2 group a fingerprint
(the average over its boxes), then kept joining the two most alike groups, also across pages, while they
were more alike than a threshold T. Keeping only the 3 most talkative speakers helps too, because small
extra speakers cost a lot in balanced_joint_f1. With T and CAP picked on the dev set this gave **0.468**
cross-validated, up from 0.342, the biggest jump of the project (`Fingerprints/magi-fingerprint.ipynb`).

### Fix 2: confirmed speakers

A speaker is **confirmed** when the text sits inside a speech bubble, thought cloud or box. Text floating
in the air is not confirmed and belongs to the last confirmed speaker. To find bubbles I used OpenCV: the
white paper around the text must be a small closed region, or rays sent out from the text must hit an
outline all around it; a speech-bubble tail also counts. It calls 96% of the texts that have a tail
"inside", and finds 85 of 1851 lines in the air. It fixes the Kaguya monologue, but adds only
**+0.004** (0.468 → 0.472, better in 80% of the splits), because only about 5% of lines float in the air
(`confirmed speaker/confirmed-chars.ipynb`).

### Fix 3: narration

A logistic regression on simple clues: narration never says "I / me / you" (0% of narration lines vs 45%
of character lines), almost never has a tail (1% vs 57%), and more often uses he/she, past tense and names.
It finds narration lines well (AUC 0.93), but adds only about **+0.009** (to ≈0.48), because few
sequences have narration and every wrong NARRATION counts as an extra speaker
(`Narration/narration.ipynb`).

---

## 3. Results

### Step by step (balanced_joint_f1, 80 development sequences)

Cross-validated = 5 folds, settings picked on 4 and scored on the 5th, averaged over 200 different splits.
With only 80 sequences one split can move the score by about ±0.02, as much as most of my changes, so I
checked everything this way. "All 80" is what `score.py` gives on all dev sequences with the settings
picked on those same sequences.

| Pipeline | Cross-validated | All 80 (optimistic) | Notebook |
|---|---|---|---|
| Magiv2 as it is | 0.342 | 0.342 | `Fingerprints/` |
| + fingerprints | 0.468 | 0.482 | `Fingerprints/` |
| + confirmed-speaker rule | 0.472 | 0.485 | `confirmed speaker/` |
| + narration | ≈0.48 (+0.009, between 0.000 and +0.015 over 6 repeats) | 0.510 | `Narration/` |

### VLM experiments (16-sequence validation split)

The same 16 dev sequences for every row; the other 64 were used for training.

| System | text_order_score | balanced_joint_f1 |
