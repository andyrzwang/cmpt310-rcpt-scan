# src/textExtraction/cross_validate.py

import os
import sys
import re
import json
import random

import cv2
import numpy as np
import pandas as pd
import progressbar
from sklearn.model_selection import KFold, ParameterGrid

# ─── make preprocess modules importable ────────────────────────────────────────
sys.path.append(
    os.path.abspath(
        os.path.join(os.path.dirname(__file__), '..', 'preprocess')
    )
)
from file_full_path     import folder_file_path
from grayscale          import grayscale
from binarization       import apply_binarization
from rescale            import rescale

# ─── import your OCR + total-extraction routine ───────────────────────────────
from textDetection      import text_detection


def cross_validate(max_files=50, n_splits=5, sample_size=None):
    """
    Args:
      max_files   – max receipts to use (randomly sampled if more exist)
      n_splits    – number of CV folds
      sample_size – number of hyper-param combos to test (None → all)
    """
    # 1) Gather all available receipts
    imgs, sols = [], []
    for i in range(1, 1000):
        fn = f"{i:03d}.jpg"
        ip = folder_file_path('images', fn)
        sp = folder_file_path('gdt',    fn.replace('.jpg', '.json'))
        if os.path.exists(ip) and os.path.exists(sp):
            imgs.append(ip)
            sols.append(sp)

    if not imgs:
        print("⚠️  No images found. Check your folder_file_path settings.")
        return

    # 1️⃣ Fewer images: randomly sample max_files receipts
    if len(imgs) > max_files:
        combined = list(zip(imgs, sols))
        random.seed(42)
        sampled = random.sample(combined, max_files)
        imgs, sols = zip(*sampled)
    num_receipts = len(imgs)
    print(f"Using {num_receipts} receipts for CV (max_files={max_files})")

    # 2) Hyper-parameter grid
    grid = {
        'clip_limit':       [1.0, 2.0, 3.0],
        'tile_grid_size':   [(8, 8), (16, 16)],
        'bin_method':       ['otsu', 'adaptive'],
        'block_size':       [11, 15, 21],
        'C':                [2, 5, 10],
        'font_size_thresh': [12, 14, 16],
        'small_scale':      [1.5, 2.0, 2.5],
        'large_scale':      [0.75, 1.0, 1.25],
    }
    full_param_list = list(ParameterGrid(grid))

    # 3️⃣ Sample hyper-params if requested
    if sample_size is not None and sample_size < len(full_param_list):
        random.seed(42)
        param_list = random.sample(full_param_list, sample_size)
        print(f"🔀 Sampling {sample_size} of {len(full_param_list)} total combos")
    else:
        param_list = full_param_list
        print(f"✅ Testing all {len(param_list)} parameter combos")

    # Total steps = combos × folds × receipts
    total_steps = len(param_list) * n_splits * num_receipts

    # ─── progress bar (fullScaleTest.py style) ─────────────────────────────────
    widgets = [
        progressbar.Percentage(), ' ',
        progressbar.Counter(format='(%(value)d of %(max_value)d)'), ' ',
        progressbar.Bar(marker='=', left='|', right='|'), ' ',
        progressbar.Timer(format='Elapsed Time: %(elapsed)s'), ' ',
        progressbar.ETA()
    ]
    bar = progressbar.ProgressBar(maxval=total_steps, widgets=widgets).start()

    # 2️⃣ Fewer folds: use n_splits
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)

    results = []
    step = 0

    for params in param_list:
        fold_accs = []

        for _, val_idx in kf.split(imgs):
            correct = 0

            # per-receipt evaluation & progress update
            for j in val_idx:
                ip, sp = imgs[j], sols[j]
                img = cv2.imread(ip)

                # a) grayscale + CLAHE
                g = grayscale(
                    img,
                    clip_limit=params['clip_limit'],
                    tile_grid_size=params['tile_grid_size']
                )

                # b) binarization
                b = apply_binarization(
                    g,
                    method=params['bin_method'],
                    block_size=params['block_size'],
                    C=params['C']
                )

                # c) dynamic rescale
                r = rescale(
                    b,
                    font_size_thresh=params['font_size_thresh'],
                    small_scale=params['small_scale'],
                    large_scale=params['large_scale']
                )

                # d) OCR total extraction
                total, _ = text_detection(r, r, ip)

                # e) load ground-truth total
                with open(sp, 'r') as f:
                    sol = json.load(f)
                sol_val = re.sub(r'[^\d.]', '', str(sol.get('total', 0.0))) or "0"
                sol_total = float(sol_val)

                # f) compare (None→0.0)
                pred = float(total) if total is not None else 0.0
                if pred == sol_total:
                    correct += 1

                # g) update bar
                step += 1
                bar.update(step)
                sys.stdout.flush()

            fold_accs.append(correct / len(val_idx))

        results.append({**params, 'accuracy': np.mean(fold_accs)})

    bar.finish()

    # ─── Report the best setting ────────────────────────────────────────────────
    df = (
        pd.DataFrame(results)
          .sort_values('accuracy', ascending=False)
          .reset_index(drop=True)
    )
    best = df.iloc[0].to_dict()

    print("\n▶ Best hyper-parameters (mean accuracy):")
    for k, v in best.items():
        if isinstance(v, float):
            print(f"   • {k:17s}: {v:.4f}")
        else:
            print(f"   • {k:17s}: {v}")


if __name__ == "__main__":
    # Sample size represents how many hyper-parameter combinations to test.
    cross_validate(max_files=20, n_splits=3, sample_size=10)