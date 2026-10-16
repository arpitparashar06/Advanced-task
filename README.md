# Manga transcription with consistent speakers (BYTE Applied ML, Task 2)

Given three consecutive manga pages, the task is to write out the story text in reading order and say who
speaks each line, with the same character keeping the same name on all three pages.

My final system uses Magiv2 to read the pages, then fixes the three things I saw it get wrong:
the same character gets a different name on every page, text floating in the air gets linked to the
nearest face, and narration gets linked to a character.

| text_order_score | balanced_joint_f1 | speaker_accuracy_on_matched | joint_f1 | matched_token_coverage |
|---|---|---|---|---|
| 0.929 | 0.510 | 0.712 | 0.673 | 0.963 |


**Run it:** `python predict.py --data dataset` → `predictions.jsonl` (images in, predictions out; section 6).

---

## 1. Final pipeline

Everything below runs in one go in `predict.py` (or `notebooks/11_predict.ipynb`), straight from the page images.

| Step | What it does | Built in | Tuned / trained on data? |
|---|---|---|---|
| 1 | Magiv2, as it comes: finds text, reads it, puts it in reading order, drops sound effects, links each text to a character box, groups the characters on each page, finds speech-bubble tails | notebook 1 | no |
| 2 | Fingerprints: Magiv2's own character embedder turns every character crop into 768 numbers | notebook 8 | no |
| 3 | Is every text inside a bubble / box, or floating in the air? (OpenCV, no training) | notebook 9 | no |
| 4 | Narration probability of every line: a logistic regression on simple clues | notebook 10 | yes: trained on the dev labels |
| 5 | Confirmed-speaker rule: text inside a bubble, cloud or box keeps Magiv2's link; text in the air gets the last confirmed speaker | notebook 9 | no |
| 6 | Join Magiv2's per-page groups into characters (also across pages) while their average fingerprints are more alike than T; lines linked to nobody get the previous speaker; keep the CAP most talkative speakers | notebook 8 | yes: T = 0.75, CAP = 3 |
| 7 | If the sequence seems to have a narrator, lines that look like narration become NARRATION | notebook 10 | yes: G, K, P picked on dev |

The learned / tuned parts are saved in `grouping_settings.json` and `narration_model.json`.

---

## 2. How I got there

### Magiv2 as it is: good text, bad speakers

I started with Magiv2 ([ragavsachdeva/magiv2](https://huggingface.co/ragavsachdeva/magiv2)), a model made
for exactly this kind of manga transcription (`notebooks/1_magi_baseline.ipynb`). Its text was already
good (text_order_score 0.929), but balanced_joint_f1 was only **0.342** (0.273 on the 16-sequence
validation split I used for the VLM experiments).

### Trying a VLM

The obvious modern answer was to give the pages to a vision-language model, with Magiv2's text as hints.
On the 16-sequence validation split (`notebooks/5_finetune_qwen.ipynb`):

- Qwen3-VL-8B zero-shot: 0.310. Adding a "face sheet" of character crops did not help (0.305).
- Qwen3-VL-8B fine-tuned with LoRA on the other 64 dev sequences: **0.400**.
- The same, with PopManga added as extra training data: 0.395, no gain.
- It never predicted NARRATION once.

I also tried the plain Hugging Face cookbook recipe (TRL + QLoRA) on Qwen2-VL-7B, giving it Magiv2's
lines and asking only for the speakers: 0.207 (`notebooks/6_finetune_qwen_trl.ipynb`), and 0.264 with
numbered boxes drawn on the text so it could see which line was which
(`notebooks/7_finetune_qwen_trl_boxes.ipynb`).

None of this got far past what Magiv2 already knew, so I stepped back and looked closely at Magiv2 itself.

### Looking at what actually goes wrong

On a Kaguya-sama page (`notebooks/9_confirmed_speakers.ipynb`) three things were wrong:

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
average over its boxes), then kept joining the two most alike groups, also across pages, while they were
more alike than a threshold T. Keeping only the 3 most talkative speakers helps too, because small extra
speakers cost a lot in balanced_joint_f1. With T and CAP picked on the dev set this gave **0.468**
cross-validated, the biggest jump of the project (`notebooks/8_fingerprints.ipynb`).

### Fix 2: confirmed speakers

A speaker is **confirmed** when the text sits inside a speech bubble, thought cloud or box. Text floating
in the air is not confirmed and belongs to the last confirmed speaker. To find bubbles I used OpenCV: the
white paper around the text must be a small closed region, or rays sent out from the text must hit an
outline all around it; a speech-bubble tail also counts. It calls 96% of the texts that have a tail
"inside", and finds 85 of 1851 lines in the air. It fixes the Kaguya monologue, but adds only
**+0.004** (0.468 → 0.472, better in 80% of the splits), because only about 5% of lines float in the air
(`notebooks/9_confirmed_speakers.ipynb`).

### Fix 3: narration

A logistic regression on simple clues: narration never says "I / me / you", almost never has a tail, and
often uses he/she, past tense and names. It finds narration lines well (AUC 0.93), but adds only about
**+0.01** (to ≈0.48), because only 12 of the 80 dev sequences have narration and every wrong NARRATION
counts as an extra speaker (`notebooks/10_narration.ipynb`).

### Trying better fingerprints

Since the fingerprints were the important part, I fine-tuned DINOv2-base on PopManga character crops
(crop → character name, `notebooks/4_finetune_dino.ipynb`). On held-out PopManga series it got much better
at telling characters apart (AUC 0.569 → 0.790), but on the task's pages it stayed below Magiv2's own
embedder (0.426 vs 0.455).

### The real problem is data

Even if every Magiv2 group were matched to the right character, the score would only reach 0.710. The
fingerprints are right about "same person or not" only about 73% of the time across pages. Black-and-white
manga characters differ by small details, every sequence is a new series, and big labelled character
datasets are locked behind copyright (Manga109 needs an institutional licence). That, not the model, is
what stopped the bigger jumps.

---

## 3. Results

### Step by step (balanced_joint_f1, development set)

Cross-validated = 5 folds, settings picked on 4 and scored on the 5th, averaged over 200 different splits.
With only 80 sequences one split can move the score by about ±0.02, as much as most of my changes, so I
checked everything this way. "All 80" is what `dataset/score.py` gives on all dev sequences with the
settings picked on those same sequences.

| Pipeline | Cross-validated | All 80 (optimistic) | Notebook / log |
|---|---|---|---|
| Magiv2 as it is | 0.342 | 0.342 | notebook 1 |
| Magiv2 + simple rules (same group number = same person, top 3 speakers) | 0.392 | – | `logs/tune_grouping.log` |
| + fingerprints | 0.468 | 0.482 | notebook 8, `logs/repeated_cv.log` |
| + confirmed-speaker rule | 0.472 | 0.485 | notebook 9, `logs/tail_rule.log` |
| + narration | ≈0.48 (+0.009, between 0.000 and +0.015 over 6 splits) | 0.510 | notebook 10, `logs/narration_clues.log` |

### The 16-sequence validation split (VLM experiments)

The same 16 dev sequences for every row; the other 64 were used for training.

| System | balanced_joint_f1 | Where |
|---|---|---|
| Magiv2 alone, across the 3 pages | 0.273 | notebook 1 |
| Magiv2 alone, every page on its own | 0.492 | notebook 5 |
| Qwen3-VL-8B zero-shot (Magiv2's text as hints) | 0.310 | notebook 5 |
| Qwen3-VL-8B zero-shot + face sheet | 0.305 | notebook 5 |
| Plain DINOv2 crops in a character array | 0.270 | notebook 5 |
| Qwen3-VL-8B + LoRA | 0.400 | notebook 5, `qwen_lora_adapter/` |
| Qwen3-VL-8B + LoRA, also trained on PopManga | 0.395 | – |
| Qwen2-VL-7B, cookbook QLoRA, speakers for Magiv2's lines | 0.207 | notebook 6 |
| Qwen2-VL-7B, cookbook QLoRA, + numbered boxes on the text | 0.264 | notebook 7 |
| Fingerprints (Magiv2 embedder) | 0.408 | `logs/compare_fingerprints.log` |

### Which fingerprints are best?

Same-person AUC = how well the fingerprint similarity separates "same character" from "different
character" for pairs of Magiv2 groups on different pages (0.5 = guessing).

| Fingerprints | balanced_joint_f1 (cross-validated, one split) | Same-person AUC |
|---|---|---|
| Magiv2's embedder | **0.455** | **0.729** |
| Plain DINOv2-base | 0.411 | 0.540 |
| DINOv2-base fine-tuned on PopManga | 0.426 | 0.636 |
| Magiv2 + fine-tuned DINOv2 | 0.453 | – |
| Magiv2 + random numbers (control) | 0.422 | – |

I checked the DINOv2 numbers were not a bug (crop alignment, shuffle test, big boxes only):
`logs/sanity_checks.log`.

### Things that did not help

| Idea | Result | Log |
|---|---|---|
| Learned "same person?" classifier instead of plain similarity | 0.440 vs 0.455 | `logs/train_same_person_classifier.log` |
| Every text → nearest character box | 0.410 | `logs/link_experiments.log` |
| Nearest box only for texts Magiv2 linked to nobody | 0.470 vs 0.468 | `logs/repeated_cv.log` |
| Merge until exactly K speakers are left | 0.466 vs 0.468 | `logs/repeated_cv.log` |
| Minimum lines per speaker, same-page penalty | no gain | `logs/rule_experiments.log` |
| Every text with no tail → previous speaker | 0.489, but swings a lot between sequences | `logs/tail_rule.log` |
| First narration classifier (box shape and position only) | AUC 0.698, never helped | `logs/narration_step.log` |

### Where the points are lost

| What if … | balanced_joint_f1 |
|---|---|
| Magiv2's groups are never linked across pages | 0.299 |
| Groups linked perfectly, but only across pages | 0.489 |
| Every Magiv2 group matched to the right character | 0.710 |
| Perfect speakers on my (Magiv2) text | 0.828 |

(`logs/oracle_analysis.log`, `logs/error_breakdown.log`.) Lost points by gold speaker: narration 7%,
biggest speaker 15%, 2nd 19%, 3rd 18%, 4th and smaller 17%, extra predicted speakers 24%.
UNKNOWN lines are 2% of the gold words; the scorer never scores their speaker, so I don't predict them.

---

## 4. What I would try next

- **A better character recogniser.** Fine-tune Magiv2's own embedder (not DINOv2) on many more labelled
  characters, e.g. Manga109 with a licence. This is where most of the remaining points are.
- **Use the tail directions.** Magiv2 finds the tails; checking where each tail points could catch the
  bubbles it links to the wrong face.
- **Narration from the story.** Narration is easy to spot line by line but expensive to get wrong; a model
  that reads the whole sequence's text might decide better whether there is a narrator at all.

---

## 5. Data, models and rules

- **Models** (all open-weight, run locally or on Kaggle):
  [Magiv2](https://huggingface.co/ragavsachdeva/magiv2) (Sachdeva, Shin, Zisserman, *Tails Tell Tales:
  Chapter-Wide Manga Transcriptions with Character Names*, ACCV 2024),
  [DINOv2-base](https://huggingface.co/facebook/dinov2-base),
  [Qwen3-VL-8B-Instruct](https://huggingface.co/Qwen/Qwen3-VL-8B-Instruct) with LoRA (PEFT),
  [Qwen2-VL-7B-Instruct](https://huggingface.co/Qwen/Qwen2-VL-7B-Instruct) with QLoRA following the
  [Hugging Face cookbook](https://huggingface.co/learn/cookbook/en/fine_tuning_vlm_trl).
- **External data:** [PopManga](https://huggingface.co/datasets/ragavsachdeva/popmanga_test) labels, with
  the page images fetched from MANGA Plus using the `mangaplus` package (`popmanga/`).
  Leakage check (`popmanga/check_overlap.py`): 24 dev sequences also appear in PopManga, no test
  sequences; the DINOv2 training skipped those series (Chainsaw Man, Jujutsu Kaisen, Tokyo Ghoul).
- **No hosted AI inference APIs** were used for training targets or test predictions. The test predictions
  come straight from the code; nothing was edited by hand and no test answers were used.
- **My part:** the confirmed-speaker rule and the bubble detector, the fingerprint grouping and its tuning,
  the narration classifier, the DINOv2 and Qwen fine-tuning runs, and all the evaluation and analysis.
  Magiv2, DINOv2, Qwen, PopManga and the scorer are other people's work.
- I used an AI assistant (Claude) to help write and debug code; the ideas, experiments and decisions were mine.

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
On Kaggle: `notebooks/11_predict.ipynb` (GPU T4, internet on; inputs: the task data,
`grouping_settings.json` and `narration_model.json`).

**How each step was built** (Kaggle, GPU T4, internet on, task data as input):
1. `notebooks/1_magi_baseline.ipynb` → `magi_output.json` and Magiv2's score.
2. `notebooks/5_finetune_qwen.ipynb`, `6_finetune_qwen_trl.ipynb`, `7_finetune_qwen_trl_boxes.ipynb` → the VLM experiments.
3. `notebooks/8_fingerprints.ipynb` → `char_embeddings.json`, picks T and CAP.
4. `notebooks/9_confirmed_speakers.ipynb` (CPU) → bubble detector, `confirmed.json`, the rule.
5. `notebooks/10_narration.ipynb` (CPU) → the narration classifier and `predictions.jsonl`.
6. `notebooks/4_finetune_dino.ipynb` → the DINOv2 fine-tune.

Each notebook needs the files the earlier ones save (add them as inputs on Kaggle). They are also in `data/`.
The experiment scripts (section 7) print the numbers in their logs in `logs/`.
Hardware: Kaggle T4 GPUs for Magiv2, Qwen and DINOv2; everything else on a laptop CPU.

---

## 7. Files

| File | What it is |
|---|---|
| `predict.py`, `notebooks/11_predict.ipynb` | The whole system: images in, predictions out |
| `predictions.jsonl` | Final test predictions (15 sequences) |
| `grouping_settings.json`, `narration_model.json` | T and CAP, and the narration classifier (learned on dev) |
| `notebooks/1_magi_baseline.ipynb` | Magiv2 as it is |
| `notebooks/5_finetune_qwen.ipynb` | Qwen3-VL zero-shot, face sheet, DINOv2 array, LoRA fine-tune |
| `notebooks/6_finetune_qwen_trl.ipynb`, `notebooks/7_finetune_qwen_trl_boxes.ipynb` | Qwen2-VL cookbook QLoRA |
| `notebooks/8_fingerprints.ipynb` | Fingerprints and joining groups |
| `notebooks/9_confirmed_speakers.ipynb` | What goes wrong, bubble detector, confirmed-speaker rule |
| `notebooks/10_narration.ipynb` | Narration classifier, final predictions |
| `notebooks/4_finetune_dino.ipynb` | DINOv2 fine-tune on PopManga crops |
| `link_speakers.py`, `embed_characters.py`, `grouping.py`, `bubbles.py`, `narration.py` | The same pipeline as scripts working from the saved files in `data/`, used by the experiment scripts |
| `data/` | Magiv2 output and fingerprints (Magiv2, plain and fine-tuned DINOv2) |
| `qwen_lora_adapter/`, `dino_finetuned/` | Trained weights |
| `tune_grouping.py`, `repeated_cv.py`, `tail_rule.py`, `narration_clues.py`, `compare_fingerprints.py`, `link_experiments.py`, `rule_experiments.py`, `train_same_person_classifier.py`, `narration_step.py` | Experiments |
| `error_breakdown.py`, `oracle_analysis.py`, `scorer_analysis.py`, `sanity_checks.py` | Analysis |
| `fast_scoring.py` | Makes repeated scoring faster (same scores) |
| `popmanga/` | Getting PopManga, overlap check |
| `logs/` | Output of every experiment |
| `predictions_v1.jsonl` | First version, before the confirmed-speaker rule and narration |
