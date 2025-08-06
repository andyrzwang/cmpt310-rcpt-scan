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


def evaluate_single_config(imgs, sols, params, kf, progress_callback=None):
    """Evaluate a single parameter configuration using cross-validation"""
    fold_accs = []
    
    for _, val_idx in kf.split(imgs):
        correct = 0
        
        for j in val_idx:
            ip, sp = imgs[j], sols[j]
            img = cv2.imread(ip)
            
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
                    method=params['bin_method'],
                    block_size=params['block_size'],
                    C=params['C']
                )
            else:  # otsu
                b = apply_binarization(
                    g,
                    method=params['bin_method']
                    # block_size and C are ignored for otsu
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
            
            # Update progress bar after each image
            if progress_callback:
                progress_callback()
        
        fold_accs.append(correct / len(val_idx))
    
    return np.mean(fold_accs)


def cross_validate_sequential(max_files=50, n_splits=5):
    """
    Sequential hyperparameter optimization:
    1. Start with baseline parameters
    2. Optimize one parameter at a time
    3. Keep the best value and move to next parameter
    4. Handle conditional parameters (block_size, C only for adaptive)
    """
    
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

    # 2) Define parameter search spaces
    param_options = {
        'clip_limit':       [1.0, 2.0, 3.0, 4.0],
        'tile_grid_size':   [(8, 8), (16, 16), (32, 32)],
        'bin_method':       ['otsu', 'adaptive'],
        'block_size':       [11, 15, 21, 25],  # only for adaptive
        'C':                [2, 5, 10, 15],     # only for adaptive  
        'font_size_thresh': [10, 12, 14, 16, 18],
        'small_scale':      [1.0, 1.5, 2.0, 2.5, 3.0],
        'large_scale':      [0.5, 0.75, 1.0, 1.25, 1.5],
    }
    
    # 3) Start with baseline parameters (middle values)
    best_params = {
        'clip_limit':       2.0,
        'tile_grid_size':   (16, 16),
        'bin_method':       'otsu',
        'block_size':       15,  # default, only used if adaptive
        'C':                5,   # default, only used if adaptive
        'font_size_thresh': 14,
        'small_scale':      2.0,
        'large_scale':      1.0,
    }
    
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    
    # Calculate total evaluations for progress bar
    total_evaluations = 0
    for param_name in ['clip_limit', 'tile_grid_size', 'bin_method', 'block_size', 
                       'C', 'font_size_thresh', 'small_scale', 'large_scale']:
        total_evaluations += len(param_options[param_name]) * n_splits * num_receipts
    
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
        bar.update(step)
        sys.stdout.flush()
    
    # Get baseline performance
    baseline_score = evaluate_single_config(imgs, sols, best_params, kf, progress_callback)
    print(f"Baseline accuracy: {baseline_score:.4f}")
    print(f"Baseline params: {best_params}")
    
    # 4) Sequential optimization
    optimization_order = [
        'clip_limit',
        'tile_grid_size', 
        'bin_method',
        'block_size',      # conditional on bin_method
        'C',               # conditional on bin_method  
        'font_size_thresh',
        'small_scale',
        'large_scale'
    ]
    
    optimization_history = []
    
    for param_name in optimization_order:
        print(f"\nOptimizing {param_name}...")
        
        # Skip conditional parameters if not applicable
        if param_name in ['block_size', 'C'] and best_params['bin_method'] != 'adaptive':
            print(f"   Skipping {param_name} (bin_method is not 'adaptive')")
            # Still update progress bar for skipped steps
            skip_steps = len(param_options[param_name]) * n_splits * num_receipts
            for _ in range(skip_steps):
                progress_callback()
            continue
        
        best_score_for_param = -1
        best_value_for_param = best_params[param_name]
        param_results = []
        
        # Test each value for this parameter
        for value in param_options[param_name]:
            # Create test configuration
            test_params = best_params.copy()
            test_params[param_name] = value
            
            # Evaluate this configuration
            score = evaluate_single_config(imgs, sols, test_params, kf, progress_callback)
            param_results.append((value, score))
            
            if score > best_score_for_param:
                best_score_for_param = score
                best_value_for_param = value
            
            print(f"   {param_name} = {value}: {score:.4f}")
        
        # Update best parameters if we found improvement
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
            print(f"   {k:17s}: {v} (unused - bin_method is '{best_params['bin_method']}')")
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
        best_score = step['best_score']
        
        status = "KEPT" if best_val == kept_val else "REJECTED"
        print(f"   {param:15s}: best={best_val} ({best_score:.4f}) -> {status}")