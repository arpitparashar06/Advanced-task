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
Its text was already good (text_order_score 0.929), but balanced_joint_f1 was only **0.342**
(0.273 on the 16-sequence validation split I used for the VLM experiments).

### Trying a VLM

- First I fine-tuned Qwen2-VL-7B (Hugging Face cookbook recipe, TRL + 4-bit QLoRA) to do the whole task:
  every line, in order, with its speaker. The results were far below Magiv2: balanced_joint_f1 **0.042**
  and text_order_score 0.198 on the validation split. To fit three pages on a T4 they had to be shrunk to
  640 px, too small to read the text, and it called almost every line NARRATION
  (`QWEN for everything/qwen-only.ipynb`).

Next the obvious thing to do was to give Qwen Magiv2's text and ask it only who says each line
(`qwen with magi support/qwen-with-magi-support.ipynb`):

- The same cookbook recipe, with Magiv2's lines numbered in reading order: 0.207. Qwen could not tell
  which line belonged to which bubble.
- 0.264 with each line's number drawn in a red box on the page, so it could see which line was which.
  Still below Magiv2 alone (0.273): the answers often did not follow the "number: speaker" format, so many
  lines fell back to the previous speaker.

So I stepped back and researched more about Magiv2 itself, with the help of an LLM.

### Looking at what actually goes wrong

On a Kaguya-sama page (`confirmed speaker/confirmed-chars.ipynb`) three things were wrong:

1. The same girl is two different groups on one page, and on the next page the group numbers start
   again. Magiv2 only groups characters inside one page.
2. Kaguya's thoughts float in the air next to the boy, and Magiv2 gives them to the boy.
3. The narration boxes ("Shinomiya thought for an instant...") are linked to a face. Magiv2 has no
   narration output at all: on the dev set it links 72 of the 80 narration lines to some character.

Scored page by page Magiv2 gets 0.492 on the validation split, but across the three pages only 0.273,
so problem 1 is the big one.

### Fix 1: fingerprints

Reading the Magiv2 paper and code, I found that it has a character embedder inside (it uses it to name
characters when you give it pictures of them). I used it to give every Magiv2 group a fingerprint (the
average over its boxes), then kept joining the two most alike groups, also across pages, while their
similarity was above a threshold T (cosine similarity 0.75 in my case). Keeping only the 3 most talkative
speakers helps too, because small extra speakers cost a lot in balanced_joint_f1. With T and CAP picked on
the dev set this gave **0.468** cross-validated, the biggest jump of the project
(`Fingerprints/magi-fingerprint.ipynb`).

### Fix 2: confirmed speakers

A speaker is **confirmed** when the text sits inside a speech bubble, thought cloud or box. Text floating
in the air is not confirmed and belongs to the last confirmed speaker. To find bubbles I used OpenCV: the
white paper around the text must be a small closed region, or rays sent out from the text must hit an
outline all around it; a speech-bubble tail also counts. It calls 96% of the texts that have a tail
"inside", and finds 85 of 1851 lines in the air. It fixes the Kaguya monologue, but adds only
**+0.004** (0.468 → 0.472, better in 80% of the splits), because only about 5% of lines float in the air
(`confirmed speaker/confirmed-chars.ipynb`).

### Fix 3: narration

A logistic regression on simple clues: narration never says "I / me / you", almost never has a tail, and
often uses he/she, past tense and names. It finds narration lines well (AUC 0.93), but adds only about
**+0.01** (to ≈0.48), because only 12 of the 80 dev sequences have narration and every wrong NARRATION
counts as an extra speaker (`Narration/narration.ipynb`).

### The real problem is data

The only external data I could find was the [PopManga](https://huggingface.co/datasets/ragavsachdeva/popmanga_test)
dataset. Fine-tuning a character recogniser on its character crops helped, but only a tiny bit, not
enough to beat Magiv2's own fingerprints, so I kept those (this was a quick test, its notebook is not in
this repo). To get better results a good dataset is
required. [Manga109](http://www.manga109.org/en/) could help, but it is restricted to academic use and
needs a college email id. I don't have one yet; as soon as I do, my plan is to get this dataset and
fine-tune Magiv2 itself on it.

---

## 3. Results

### Step by step (balanced_joint_f1, development set)

Cross-validated = 5 folds, settings picked on 4 and scored on the 5th, averaged over 200 different splits.
With only 80 sequences one split can move the score by about ±0.02, as much as most of my changes, so I
checked everything this way. "All 80" is what `dataset/score.py` gives on all dev sequences with the
settings picked on those same sequences.

| Pipeline | Cross-validated | All 80 (optimistic) | Notebook |
|---|---|---|---|
| Magiv2 as it is | 0.342 | 0.342 | `Fingerprints/` |
| + fingerprints | 0.468 | 0.482 | `Fingerprints/` |
| + confirmed-speaker rule | 0.472 | 0.485 | `confirmed speaker/` |
| + narration | ≈0.48 (+0.009, between 0.000 and +0.015 over 6 repeats) | 0.510 | `Narration/` |

### The 16-sequence validation split (VLM experiments)

The same 16 dev sequences for every row; the other 64 were used for training.

| System | balanced_joint_f1 | Notebook |
|---|---|---|
| Magiv2 alone, across the 3 pages | 0.273 | `baseline magi.ipynb` |
| Magiv2 alone, every page on its own | 0.492 | quick check, not saved |
| Qwen2-VL-7B QLoRA, whole transcription | 0.042 | `QWEN for everything/` |
| Qwen2-VL-7B QLoRA, speakers for Magiv2's lines | 0.207 | an earlier version of `qwen with magi support/` (not saved) |
| Qwen2-VL-7B QLoRA, + numbered boxes on the pages | 0.264 | `qwen with magi support/` |
| Magiv2 + fingerprints | 0.408 | quick check, not saved |

---

## 4. What I would try next

- **A better character recogniser.** Fine-tune Magiv2's own embedder on many more labelled characters,
  e.g. Manga109. This is where most of the remaining points are.
- **Narration from the story.** Narration is easy to spot line by line but expensive to get wrong; a model
  that reads the whole sequence's text might decide better whether there is a narrator at all.

---

## 5. Models, data and rules

- **Models** (all open-weight, run on Kaggle):
  [Magiv2](https://huggingface.co/ragavsachdeva/magiv2) (Sachdeva, Shin, Zisserman, *Tails Tell Tales:
  Chapter-Wide Manga Transcriptions with Character Names*, ACCV 2024; pinned to commit `fbc890f` in
  `predict.py` and the notebooks),
  [Qwen2-VL-7B-Instruct](https://huggingface.co/Qwen/Qwen2-VL-7B-Instruct) with QLoRA following the
  [Hugging Face cookbook](https://huggingface.co/learn/cookbook/en/fine_tuning_vlm_trl).
- **Data:** the task's dataset; [PopManga](https://huggingface.co/datasets/ragavsachdeva/popmanga_test)
  only for the character recogniser test above.
- **No hosted AI inference APIs** were used for training targets or test predictions. The test predictions
  come straight from the code; nothing was edited by hand and no test answers were used.
- **My part:** the confirmed-speaker rule and the bubble detector, the fingerprint grouping and its tuning,
  the narration classifier, the Qwen fine-tuning runs and the evaluation. Magiv2, Qwen, the cookbook
  recipe and the scorer are other people's work. I used an AI assistant (Claude) to help write and debug
  code.

---

## 6. How to run

```
pip install -r requirements.txt
```

Put the task data (the folder with `sequences.json`, `development/`, `test/`, `score.py`) in `./dataset`.
It is not included because it is not mine to publish.

**The whole system** (a GPU is much faster, CPU works):
```
python predict.py --data dataset                          # -> predictions.jsonl (15 test sequences)
python predict.py --data dataset --split development      # -> dev_predictions.jsonl
python dataset/score.py --references dataset/development/labels.jsonl --predictions dev_predictions.jsonl --output dev_scores.json
python predict.py --pages page1.png page2.png page3.png   # -> one JSON for any 3 pages
```

**The notebooks** (Kaggle, GPU T4, internet on, task data as input), in order:
1. `baseline magi.ipynb` → `magi_output.json` and Magiv2's score.
2. `QWEN for everything/qwen-only.ipynb` → Qwen2-VL fine-tuned for the whole task (adapter in the same folder).
3. `qwen with magi support/qwen-with-magi-support.ipynb` → Qwen2-VL for the speakers only (adapter in the same folder). Needs `magi_output.json`.
4. `Fingerprints/magi-fingerprint.ipynb` → `char_embeddings.json`, picks T and CAP (`grouping_settings.json`). Needs `magi_output.json`.
5. `confirmed speaker/confirmed-chars.ipynb` (CPU) → bubble detector, `confirmed.json`, the rule. Needs `magi_output.json` and `char_embeddings.json`.
6. `Narration/narration.ipynb` (CPU) → the narration classifier, `narration_model.json` and `predictions.jsonl`. Needs all three json files.
7. `check predict.ipynb` → runs `predict.py` straight from the images on test and dev, scores dev, and compares with `predictions.jsonl`.

On Kaggle, add the json files the earlier notebooks save as inputs; they are also in this repo.

---

## 7. Files

| File | What it is |
|---|---|
| `predict.py` | The whole system: images in, predictions out |
| `predictions.jsonl` | Final test predictions (15 sequences) |
| `grouping_settings.json`, `narration_model.json` | T and CAP, and the narration classifier (learned on dev) |
| `magi_output.json` | Magiv2's output for every page (from `baseline magi.ipynb`) |
| `baseline magi.ipynb` | Magiv2 as it is |
| `QWEN for everything/` | Qwen2-VL QLoRA for the whole task: notebook and adapter |
| `qwen with magi support/` | Qwen2-VL QLoRA for the speakers of Magiv2's lines: notebook and adapter |
| `Fingerprints/` | Fingerprints and joining groups: notebook and `char_embeddings.json` |
| `confirmed speaker/` | Bubble detector and confirmed-speaker rule: notebook and `confirmed.json` |
| `Narration/` | Narration classifier and final predictions |
| `check predict.ipynb` | `predict.py` run from the images, compared with `predictions.jsonl` |
| `requirements.txt` | Packages |
