# predict.py
# The whole system in one file: three manga pages in -> story text in reading order with speakers out.
# Same code as the notebooks in Fingerprints/, confirmed speaker/ and Narration/, put together:
#   1. Magiv2 reads the pages: text boxes, OCR, reading order, sound effects dropped, links to characters, tails
#   2. Magiv2's crop embedder gives every character box a fingerprint
#   3. OpenCV checks if every text is inside a bubble / box or floating in the air
#   4. the narration classifier gives every line a "narration" probability
#   5. my confirmed-speaker rule: text in the air -> last confirmed speaker
#   6. Magi's groups are joined into characters by fingerprint, CAP most talkative kept
#   7. lines that look like narration become NARRATION
#
# The learned / tuned parts are loaded from files made on the 80 development sequences:
#   grouping_settings.json - T and CAP (Fingerprints/magi-fingerprint.ipynb)
#   narration_model.json   - the narration classifier and its G, K, P (Narration/narration.ipynb)
#
# usage (needs transformers==4.49.0; a GPU is much faster but CPU works):
#   python predict.py --data dataset                          -> predictions.jsonl for the 15 test sequences
#   python predict.py --data dataset --split development      -> dev_predictions.jsonl, then score it with
#       python dataset/score.py --references dataset/development/labels.jsonl --predictions dev_predictions.jsonl --output dev_scores.json
#   python predict.py --pages page1.png page2.png page3.png   -> prints one JSON object for these 3 pages

import os
import re
import sys
import json
import math
import argparse
import numpy as np
import cv2
import torch
from PIL import Image
from collections import Counter

HERE = os.path.dirname(os.path.abspath(__file__))
settings = json.load(open(os.path.join(HERE, "grouping_settings.json")))
narration_model = json.load(open(os.path.join(HERE, "narration_model.json")))
T = settings["T"]
CAP = settings["CAP"]

device = "cuda" if torch.cuda.is_available() else "cpu"
model = None           # Magiv2, loaded at the bottom
# Magiv2 runs its own code from Hugging Face (trust_remote_code), so pin the exact version:
# its latest commit (7 Mar 2025), the one all my notebooks downloaded
MAGI_REVISION = "fbc890fec52977142e8ee00bfe26e9458b65517c"


# ======================= 1. Magiv2 =======================

def read_image(path):
    # grey then back to 3 channels, like Magiv2's model card
    return np.array(Image.open(path).convert("L").convert("RGB"))


def run_magi(paths):
    # Magiv2 on the 3 pages together
    pages = []
    for path in paths:
        pages.append(read_image(path))
    with torch.no_grad():
        results = model.do_chapter_wide_prediction(pages, {"images": [], "names": []}, use_tqdm=False, do_ocr=True)

    saved = []
    for r in results:
        saved.append({
            "text_boxes": [[float(v) for v in b] for b in r["texts"]],
            "char_boxes": [[float(v) for v in b] for b in r["characters"]],
            "links": [[int(a), int(b)] for a, b in r["text_character_associations"]],
            "tail_links": [[int(a), int(b)] for a, b in r["text_tail_associations"]],
            "clusters": [int(c) for c in r["character_cluster_labels"]],
            "keep": [bool(k) for k in r["is_essential_text"]],
            "ocr": [str(t) for t in r["ocr"]],
        })
    return saved


# ======================= 2. fingerprints (Fingerprints/) =======================

MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def crop(img, b):
    # same box fixing as magi: whole pixels, inside the page, at least 10 px
    H, W = img.shape[0], img.shape[1]
    x1 = max(0, min(int(min(b[0], b[2])), W))
    x2 = max(0, min(int(max(b[0], b[2])), W))
    y1 = max(0, min(int(min(b[1], b[3])), H))
    y2 = max(0, min(int(max(b[1], b[3])), H))
    if x2 - x1 < 10:
        if W - x1 > 10:
            x2 = x1 + 10
        else:
            x1 = x2 - 10
    if y2 - y1 < 10:
        if H - y1 > 10:
            y2 = y1 + 10
        else:
            y1 = y2 - 10
    return img[y1:y2, x1:x2]


def fingerprints(path, boxes):
    img = read_image(path)
    batch = []
    for b in boxes:
        c = Image.fromarray(crop(img, b)).resize((224, 224), Image.BILINEAR)
        x = (np.asarray(c).astype(np.float32) / 255.0 - MEAN) / STD
        batch.append(x.transpose(2, 0, 1))
    with torch.no_grad():
        out = model.crop_embedding_model(pixel_values=torch.tensor(np.stack(batch)).to(device)).last_hidden_state[:, 0]
    return out.float().cpu().numpy().round(4).tolist()


def read_sequence(magi, emb):
    # lines = every story line in reading order: [page, text, group]
    # group = (page, magi group) or None if magi linked it to nobody
    lines = []
    groups = {}
    for p in range(3):
        r = magi[p]
        link = {a: b for a, b in r["links"]}
        for i in range(len(r["ocr"])):
            if r["keep"][i] == True and r["ocr"][i].strip() != "":
                group = None
                c = link.get(i, -1)
                if 0 <= c < len(r["clusters"]):
                    group = (p, r["clusters"][c])
                    if group not in groups:
                        groups[group] = {"lines": []}
                    groups[group]["lines"].append(len(lines))
                lines.append([p, r["ocr"][i].strip(), group])

        # every group gets one fingerprint: the average of its boxes
        if p not in emb or len(emb[p]) == 0:
            continue
        E = np.array(emb[p], dtype=np.float32)
        E = E / (np.linalg.norm(E, axis=1, keepdims=True) + 1e-8)
        for (gp, g) in groups:
            if gp != p:
                continue
            rows = [c for c in range(min(len(r["clusters"]), len(E))) if r["clusters"][c] == g]
            if len(rows) == 0:
                continue
            v = E[rows].mean(0)
            groups[(p, g)]["vec"] = v / (np.linalg.norm(v) + 1e-8)
    return lines, groups


def merge_groups(groups, T):
    # start with every magi group alone, keep joining the two most alike while they are above T
    keys = sorted([k for k in groups if "vec" in groups[k]])
    clusters = [[k] for k in keys]
    while True:
        best = None
        for i in range(len(clusters)):
            for j in range(i + 1, len(clusters)):
                s = float(np.mean([groups[a]["vec"] @ groups[b]["vec"] for a in clusters[i] for b in clusters[j]]))
                if s > T and (best is None or s > best[0]):
                    best = (s, i, j)
        if best is None:
            break
        clusters[best[1]] = clusters[best[1]] + clusters[best[2]]
        del clusters[best[2]]
    person = {}
    for n in range(len(clusters)):
        for k in clusters[n]:
            person[k] = "c" + str(n)
    return person


def finish(lines, person, CAP):
    rows = []
    for page, text, g in lines:
        spk = None
        if g is not None:
            spk = person[g] if g in person else "p" + str(g[0]) + "_g" + str(g[1])
        rows.append([page, text, spk])

    # magi linked it to nobody -> previous speaker (or the next one at the start)
    for i in range(len(rows)):
        if rows[i][2] is None:
            spk = None
            for x in range(i - 1, -1, -1):
                if rows[x][2] is not None:
                    spk = rows[x][2]
                    break
            if spk is None:
                for x in range(i + 1, len(rows)):
                    if rows[x][2] is not None:
                        spk = rows[x][2]
                        break
            rows[i][2] = spk if spk is not None else "NARRATION"

    # keep the CAP most talkative, the rest -> previous kept speaker
    count = Counter(r[2] for r in rows if r[2] != "NARRATION")
    keep = [name for name, n in count.most_common(CAP)]
    last = None
    for r in rows:
        if r[2] != "NARRATION" and r[2] not in keep:
            r[2] = last if last is not None else keep[0]
        if r[2] != "NARRATION":
            last = r[2]

    names = {}
    pages = [[], [], []]
    for page, text, spk in rows:
        if spk != "NARRATION":
            if spk not in names:
                names[spk] = "character" + str(len(names) + 1)
            spk = names[spk]
        pages[page].append({"speaker": spk, "text": text})
    return pages


# ======================= 3. confirmed speakers (confirmed speaker/) =======================

def outline_rays(gray, box):
    H, W = gray.shape
    x1, y1, x2, y2 = [float(v) for v in box]
    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
    reach = max(60.0, 0.8 * max(x2 - x1, y2 - y1))
    dists = []
    for k in range(24):
        a = 2 * math.pi * k / 24
        dx, dy = math.cos(a), math.sin(a)
        to_x = (x2 - x1) / 2 / abs(dx) if abs(dx) > 1e-6 else 1e9
        to_y = (y2 - y1) / 2 / abs(dy) if abs(dy) > 1e-6 else 1e9
        start = min(to_x, to_y) + 2          # just outside the text box
        d = 0
        while d <= reach:
            x = int(round(cx + dx * (start + d)))
            y = int(round(cy + dy * (start + d)))
            if x < 0 or y < 0 or x >= W or y >= H:
                break
            if gray[y, x] < 140:
                dists.append(d)
                break
            d += 1
    hit_share = len(dists) / 24
    padding = float(np.median(dists)) if len(dists) > 0 else -1
    return hit_share, padding


def inside_bubble_or_box(gray, text_boxes):
    H, W = gray.shape
    dark = (gray <= 180).astype(np.uint8)
    dark = cv2.dilate(dark, np.ones((7, 7), np.uint8))     # close small gaps (dotted thought clouds)
    for b in text_boxes:                                     # rub out the letters
        x1, y1, x2, y2 = [int(round(v)) for v in b]
        dark[max(0, y1):y2 + 1, max(0, x1):x2 + 1] = 0
    n, labels, stats, _ = cv2.connectedComponentsWithStats(1 - dark, connectivity=4)
    result = []
    for b in text_boxes:
        x1, y1, x2, y2 = [int(round(v)) for v in b]
        letters = gray[max(0, y1):y2 + 1, max(0, x1):x2 + 1]
        white = (letters > 200).mean()
        black = (letters < 60).mean()
        grey = 1 - white - black
        label = labels[min(H - 1, (y1 + y2) // 2), min(W - 1, (x1 + x2) // 2)]
        sx, sy, sw, sh, area = stats[label]
        touches_edge = sx <= 1 or sy <= 1 or sx + sw >= W - 1 or sy + sh >= H - 1
        closed = (not touches_edge) and area <= 0.08 * H * W
        hit_share, padding = outline_rays(gray, b)
        if black > 0.5 and grey < 0.15:
            inside = True
        elif white < 0.6:
            inside = False
        elif closed:
            inside = True
        elif hit_share >= 0.7 and (4 <= padding <= 35 or hit_share >= 0.95):
            inside = True
        else:
            inside = False
        result.append(inside)
    return result


def line_info(magi, paths):
    info = []
    for p in range(3):
        r = magi[p]
        has_tail = [a for a, b in r["tail_links"]]
        gray = np.array(Image.open(paths[p]).convert("L"))
        inside = inside_bubble_or_box(gray, r["text_boxes"])
        for i in range(len(r["ocr"])):
            if r["keep"][i] == True and r["ocr"][i].strip() != "":
                tail = i in has_tail
                info.append({"tail": tail, "inside": inside[i], "confirmed": tail or inside[i]})
    return info


def confirmed_speaker_rule(lines, inf):
    last = None
    for n in range(len(lines)):
        if not inf[n]["confirmed"] and last is not None:
            lines[n][2] = last                 # in the air -> last confirmed speaker
        if inf[n]["confirmed"] and lines[n][2] is not None:
            last = lines[n][2]
    return lines


# ======================= 4. narration (Narration/) =======================

FIRST = re.compile(r"\b(i|i'm|i've|i'll|i'd|me|my|mine|we|we're|us|our|you|you're|your|yours|ya)\b")
THIRD = re.compile(r"\b(he|she|his|her|him|they|their|them|its)\b")
PAST = re.compile(r"\b(was|were|had|did)\b|\w{3,}ed\b")


def clues(magi, lines, info, paths):
    # lines: from read_sequence, before the rule changes the links
    rows = []
    n = 0
    for p in range(3):
        r = magi[p]
        W, H = Image.open(paths[p]).size
        for i in range(len(r["ocr"])):
            if r["keep"][i] == True and r["ocr"][i].strip() != "":
                text = lines[n][1]
                low = text.lower()
                letters = [c for c in text if c.isalpha()]
                names = re.findall(r"(?<![.!?]\s)(?<!^)\b[A-Z][a-z]{2,}", text)
                b = r["text_boxes"][i]
                rows.append([1.0 if info[n]["tail"] else 0.0,
                             1.0 if info[n]["inside"] else 0.0,
                             1.0 if lines[n][2] is not None else 0.0,
                             1.0 if FIRST.search(low) else 0.0,
                             1.0 if THIRD.search(low) else 0.0,
                             1.0 if PAST.search(low) else 0.0,
                             1.0 if "?" in text else 0.0,
                             1.0 if "!" in text else 0.0,
                             float(min(len(names), 3)),
                             1.0 if re.search(r"\d|:", text) else 0.0,
                             sum(c.isupper() for c in letters) / max(1, len(letters)),
                             float(np.log1p(len(low.split()))),
                             (b[2] - b[0]) / W, (b[3] - b[1]) / H, (b[0] + b[2]) / 2 / W, (b[1] + b[3]) / 2 / H])
                n += 1
    return rows


def prob(m, rows):
    if len(rows) == 0:
        return []
    z = ((np.array(rows) - np.array(m["mean"])) / np.array(m["spread"])) @ np.array(m["weights"]) + m["bias"]
    return list(1 / (1 + np.exp(-z)))


def add_narration(pages, probs, G, K, P):
    pages = [[dict(x) for x in page] for page in pages]
    if sum(1 for p in probs if p >= G) >= K:          # does this sequence have a narrator?
        n = 0
        for page in pages:
            for x in page:
                if probs[n] >= P:                     # which lines are narration?
                    x["speaker"] = "NARRATION"
                n += 1
    names = {}
    for page in pages:
        for x in page:
            if x["speaker"] != "NARRATION":
                if x["speaker"] not in names:
                    names[x["speaker"]] = "character" + str(len(names) + 1)
                x["speaker"] = names[x["speaker"]]
    return pages


# ======================= all together =======================

def predict_sequence(paths):
    magi = run_magi(paths)

    emb = {}
    for p in range(3):
        if len(magi[p]["char_boxes"]) > 0:
            emb[p] = fingerprints(paths[p], magi[p]["char_boxes"])

    lines, groups = read_sequence(magi, emb)
    info = line_info(magi, paths)
    probs = prob(narration_model, clues(magi, lines, info, paths))
    lines = confirmed_speaker_rule(lines, info)
    pages = finish(lines, merge_groups(groups, T), CAP)
    return add_narration(pages, probs, narration_model["G"], narration_model["K"], narration_model["P"])


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="dataset", help="the task's dataset folder (with sequences.json)")
    parser.add_argument("--split", default="test", help="test or development")
    parser.add_argument("--out", default=None, help="output jsonl file")
    parser.add_argument("--pages", nargs=3, default=None, help="or: just 3 page images")
    args = parser.parse_args()

    from transformers import AutoModel
    model = AutoModel.from_pretrained("ragavsachdeva/magiv2", revision=MAGI_REVISION, trust_remote_code=True).to(device).eval()
    model.crop_embedding_model.config.mask_ratio = 0.0      # it's a ViT-MAE, by default it hides 75% of the crop

    if args.pages is not None:
        print(json.dumps({"sequence_id": "sequence", "pages": predict_sequence(args.pages)}))
        sys.exit()

    out_path = args.out
    if out_path is None:
        out_path = "predictions.jsonl" if args.split == "test" else "dev_predictions.jsonl"

    sequences = []
    for s in json.load(open(os.path.join(args.data, "sequences.json"))):
        if s["split"] == args.split:
            sequences.append(s)

    out_file = open(out_path, "w")
    for n in range(len(sequences)):
        s = sequences[n]
        paths = [os.path.join(args.data, p) for p in s["images"]]
        pages = predict_sequence(paths)
        out_file.write(json.dumps({"sequence_id": s["sequence_id"], "pages": pages}) + "\n")
        print(n + 1, "/", len(sequences), s["sequence_id"], flush=True)
    out_file.close()
    print("wrote", len(sequences), "sequences to", out_path)
