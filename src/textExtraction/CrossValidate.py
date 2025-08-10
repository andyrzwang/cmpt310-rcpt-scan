import os
import sys
import re
import json
import random

import cv2
import numpy as np
import pandas as pd
import progressbar
from sklearn.model_selection import KFold

# make preprocess modules importable
sys.path.append(
    os.path.abspath(
        os.path.join(os.path.dirname(__file__), '..', 'preprocess')
    )
)
from file_full_path     import folder_file_path
from grayscale          import grayscale
from binarization       import apply_binarization
from fix_tilted_image  import fix_tilted_image
# NOTE: rescale removed

#  import OCR + total extraction routine 
from textDetection      import text_detection


def evaluate_single_config(imgs, sols, params, kf, progress_callback=None):
    """Evaluate a single parameter configuration using cross validation"""
    fold_accs = []
    
    for _, val_idx in kf.split(imgs):
        correct = 0
        
        for j in val_idx:
            ip, sp = imgs[j], sols[j]
            img = cv2.imread(ip)
            if img is None:
                # Skip unreadable images (counts against denominator via len(val_idx))
                if progress_callback:
                    progress_callback()
                continue
            
            # a) grayscale + CLAHE
            g = grayscale(
                img,
                clip_limit=params['clip_limit'],
                tile_grid_size=params['tile_grid_size']
            )
            
            # b) binarization (conditional parameters)
            if params['bin_method'] == 'adaptive':
                b = apply_binarization(
                    g,
                    method='adaptive',
                    block_size=params['block_size'],
                    C=params['C']
                )
            else:  # otsu
                b = apply_binarization(g, method='otsu')
            
            # c) deskew AFTER binarization
            _, r = fix_tilted_image(b)

            # d) OCR total extraction
            total, _ = text_detection(r, r, ip)
            
            # e) load ground truth total
            with open(sp, 'r') as f:
                sol = json.load(f)
            sol_val = re.sub(r'[^\d.]', '', str(sol.get('total', 0.0))) or "0"
            sol_total = float(sol_val)
            
            # f) compare (strict equality to match fullScaleTest behavior)
            pred = float(total) if total is not None else 0.0
            if pred == sol_total:
                correct += 1
            
            # Update progress bar after each image
            if progress_callback:
                progress_callback()
        
        fold_accs.append(correct / len(val_idx) if len(val_idx) > 0 else 0.0)
    
    return float(np.mean(fold_accs)) if fold_accs else 0.0


def cross_validate_sequential(max_files=50, n_splits=5):
    
    # 1) Gather all available receipts
    imgs, sols = [], []
    for i in range(1, 1000):
        fn = f"{i:03d}.jpg"
        ip = folder_file_path('images', fn)
        sp = folder_file_path('gdt', fn.replace('.jpg', '.json'))
        if os.path.exists(ip) and os.path.exists(sp):
            imgs.append(ip)
            sols.append(sp)

    if not imgs:
        print("No images found. Check your folder_file_path settings.")
        return None

    # Sample if needed
    if len(imgs) > max_files:
        combined = list(zip(imgs, sols))
        random.seed(42)
        sampled = random.sample(combined, max_files)
        imgs, sols = zip(*sampled)
    
    imgs, sols = list(imgs), list(sols)
    num_receipts = len(imgs)
    print(f"Using {num_receipts} receipts for sequential CV (max_files={max_files})")

    # 2) Define parameter search spaces (rescale options removed)
    param_options = {
        'clip_limit':       [1.0, 2.0, 3.0, 4.0],
        'tile_grid_size':   [(8, 8), (16, 16), (32, 32)],
        'bin_method':       ['otsu', 'adaptive'],
        'block_size':       [11, 15, 21, 25],  # only for adaptive
        'C':                [2, 5, 10, 15],    # only for adaptive  
    }
    
    # 3) Start with baseline parameters (middle values)
    best_params = {
        'clip_limit':       2.0,
        'tile_grid_size':   (16, 16),
        'bin_method':       'otsu',
        'block_size':       15,  # default, only used if adaptive
        'C':                5,   # default, only used if adaptive
    }
    
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    
    # Calculate total evaluations for progress bar
    n_clip     = len(param_options['clip_limit'])
    n_tile     = len(param_options['tile_grid_size'])
    n_otsu     = 1
    n_adaptive = len(param_options['block_size']) * len(param_options['C'])
    n_baseline = 1

    total_configs = n_baseline + n_clip + n_tile + n_otsu + n_adaptive
    total_evaluations = total_configs * num_receipts  # NOT multiplied by n_splits

    widgets = [
        progressbar.Percentage(), ' ',
        progressbar.Counter(format='(%(value)d of %(max_value)d)'), ' ',
        progressbar.Bar(marker='=', left='|', right='|'), ' ',
        progressbar.Timer(format='Elapsed Time: %(elapsed)s'), ' ',
        progressbar.ETA()
    ]
    bar = progressbar.ProgressBar(maxval=total_evaluations, widgets=widgets).start()

    step = 0
    def progress_callback():
        nonlocal step
        step += 1
        if step <= total_evaluations:
            bar.update(step)
        sys.stdout.flush()
    
    # Get baseline performance
    baseline_score = evaluate_single_config(imgs, sols, best_params, kf, progress_callback)
    print(f"Baseline accuracy: {baseline_score:.4f}")
    print(f"Baseline params: {best_params}")
    
    # 4) Sequential optimization (with special handling for bin_method)
    optimization_history = []

    #      Optimize clip_limit    
    param_name = 'clip_limit'
    print(f"\nOptimizing {param_name}...")
    best_score_for_param = -1
    best_value_for_param = best_params[param_name]
    param_results = []
    for value in param_options[param_name]:
        test_params = best_params.copy()
        test_params[param_name] = value
        score = evaluate_single_config(imgs, sols, test_params, kf, progress_callback)
        param_results.append((value, score))
        if score > best_score_for_param:
            best_score_for_param = score
            best_value_for_param = value
        print(f"   {param_name} = {value}: {score:.4f}")
    if best_score_for_param > baseline_score:
        print(f"   Improved! {param_name}: {best_params[param_name]} -> {best_value_for_param}")
        print(f"   Score: {baseline_score:.4f} -> {best_score_for_param:.4f}")
        best_params[param_name] = best_value_for_param
        baseline_score = best_score_for_param
    else:
        print(f"   No improvement. Keeping {param_name} = {best_params[param_name]}")
    optimization_history.append({
        'parameter': param_name,
        'tested_values': param_results,
        'best_value': best_value_for_param,
        'best_score': best_score_for_param,
        'kept_value': best_params[param_name]
    })

    # Optimize tile_grid_size
    param_name = 'tile_grid_size'
    print(f"\nOptimizing {param_name}...")
    best_score_for_param = -1
    best_value_for_param = best_params[param_name]
    param_results = []
    for value in param_options[param_name]:
        test_params = best_params.copy()
        test_params[param_name] = value
        score = evaluate_single_config(imgs, sols, test_params, kf, progress_callback)
        param_results.append((value, score))
        if score > best_score_for_param:
            best_score_for_param = score
            best_value_for_param = value
        print(f"   {param_name} = {value}: {score:.4f}")
    if best_score_for_param > baseline_score:
        print(f"   Improved! {param_name}: {best_params[param_name]} -> {best_value_for_param}")
        print(f"   Score: {baseline_score:.4f} -> {best_score_for_param:.4f}")
        best_params[param_name] = best_value_for_param
        baseline_score = best_score_for_param
    else:
        print(f"   No improvement. Keeping {param_name} = {best_params[param_name]}")
    optimization_history.append({
        'parameter': param_name,
        'tested_values': param_results,
        'best_value': best_value_for_param,
        'best_score': best_score_for_param,
        'kept_value': best_params[param_name]
    })

    # -    Special: method selection with joint (block_size, C) tuning for Adaptive     
    print(f"\nSelecting binarization method with Adaptive tuning...")
    # 1) Otsu as is
    params_otsu = best_params.copy()
    params_otsu['bin_method'] = 'otsu'
    score_otsu = evaluate_single_config(imgs, sols, params_otsu, kf, progress_callback)
    print(f"   Otsu score: {score_otsu:.4f}")

    # 2) Adaptive: search grid over (block_size, C) and keep the best
    best_adapt_score = -1
    best_adapt_bs, best_adapt_C = best_params['block_size'], best_params['C']
    adapt_results = []
    for bs in param_options['block_size']:
        for C in param_options['C']:
            params_adapt = best_params.copy()
            params_adapt['bin_method'] = 'adaptive'
            params_adapt['block_size'] = bs
            params_adapt['C'] = C
            score = evaluate_single_config(imgs, sols, params_adapt, kf, progress_callback)
            adapt_results.append(((bs, C), score))
            print(f"   Adaptive block_size={bs:2d}, C={C:2d}: {score:.4f}")
            if score > best_adapt_score:
                best_adapt_score = score
                best_adapt_bs, best_adapt_C = bs, C

    # Decide method
    if best_adapt_score > score_otsu and best_adapt_score > baseline_score:
        print(f"   → Choosing ADAPTIVE with block_size={best_adapt_bs}, C={best_adapt_C}")
        print(f"     Score: {baseline_score:.4f} -> {best_adapt_score:.4f}")
        best_params['bin_method'] = 'adaptive'
        best_params['block_size'] = best_adapt_bs
        best_params['C'] = best_adapt_C
        baseline_score = best_adapt_score
        kept_value = ('adaptive', best_adapt_bs, best_adapt_C)
    else:
        print(f"   → Keeping OTSU")
        kept_value = ('otsu',)

    optimization_history.append({
        'parameter': 'bin_method_with_joint_adaptive_grid',
        'tested_values': [('otsu', score_otsu)] + [(f'adaptive(bs={bs},C={C})', s) for (bs, C), s in adapt_results],
        'best_value': kept_value,
        'best_score': baseline_score,
        'kept_value': kept_value
    })

    print("\nSkipping separate block_size and C optimization (already tuned jointly during method selection).")

    bar.finish()
    
    # 5) Final results
    final_score = baseline_score
    
    print(f"\n{'='*50}")
    print(f"SEQUENTIAL OPTIMIZATION RESULTS")
    print(f"{'='*50}")
    print(f"Final accuracy: {final_score:.4f}")
    print(f"\nBest hyperparameters:")
    for k, v in best_params.items():
        if k in ['block_size', 'C'] and best_params['bin_method'] != 'adaptive':
            print(f"   {k:17s}: {v} (unused   bin_method is '{best_params['bin_method']}')")
        else:
            print(f"   {k:17s}: {v}")
    
    return best_params, final_score, optimization_history


if __name__ == "__main__":
    best_params, final_score, history = cross_validate_sequential(
        max_files=20, 
        n_splits=3
    )
    
    # Optional: Print detailed optimization history
    print(f"\nOptimization History:")
    for step in history:
        param = step['parameter']
        best_val = step['best_value'] 
        kept_val = step['kept_value']
        best_score = step['best_score'] if isinstance(step['best_score'], float) else None
        status = "KEPT" if best_val == kept_val else "REJECTED"
        print(f"   {param:35s}: best={best_val} -> {status}")