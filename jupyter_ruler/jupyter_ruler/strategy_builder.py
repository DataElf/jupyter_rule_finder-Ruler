#!/usr/bin/env python
# coding: utf-8

import numpy as np
import pandas as pd
from itertools import combinations


def greedy_select(masks, target, max_rules=5, min_hit_rate=0.02):
    n = len(masks)
    selected = []
    remaining = list(range(n))
    current_mask = np.zeros(len(target), dtype=bool)
    while len(selected) < max_rules and remaining:
        best_gain = -np.inf
        best_idx = None
        for idx in remaining:
            new_mask = current_mask | masks[idx]
            hit_rate = new_mask.mean()
            if hit_rate < min_hit_rate:
                continue
            gain = (target[new_mask].mean() - target.mean()) * hit_rate
            if gain > best_gain:
                best_gain = gain
                best_idx = idx
        if best_idx is None:
            break
        selected.append(best_idx)
        current_mask = current_mask | masks[best_idx]
        remaining.remove(best_idx)
    return selected, current_mask


def greedy_lift_select(masks, target, max_rules=10, lift_threshold=0.05,
                       hit_increment_threshold=0.002, hit_ratio_cap=0.35):
    overall_bad_rate = target.mean()
    sorted_indices = sorted(
        range(len(masks)),
        key=lambda i: (target[masks[i]].mean() / overall_bad_rate if overall_bad_rate > 0 and masks[i].sum() > 0 else 0),
        reverse=True
    )
    selected = []
    current_mask = np.zeros(len(target), dtype=bool)
    current_lift = 1.0
    current_hit_rate = 0.0

    for idx in sorted_indices[:max_rules * 3]:
        if len(selected) >= max_rules:
            break
        candidate_mask = current_mask | masks[idx]
        candidate_hit_rate = candidate_mask.mean()
        if candidate_hit_rate < 0.005:
            continue
        if candidate_hit_rate > hit_ratio_cap and len(selected) > 0:
            continue
        candidate_bad_rate = target[candidate_mask].mean()
        candidate_lift = candidate_bad_rate / overall_bad_rate if overall_bad_rate > 0 else 0
        if candidate_lift <= 1.0:
            continue

        hit_increase = candidate_hit_rate - current_hit_rate
        lift_increase = candidate_lift - current_lift

        if len(selected) == 0:
            if candidate_lift > 1.5:
                selected.append(idx)
                current_mask = candidate_mask
                current_lift = candidate_lift
                current_hit_rate = candidate_hit_rate
        elif lift_increase >= lift_threshold and hit_increase <= hit_ratio_cap * 0.5:
            selected.append(idx)
            current_mask = candidate_mask
            current_lift = candidate_lift
            current_hit_rate = candidate_hit_rate
        elif lift_increase >= 0 and hit_increase >= hit_increment_threshold:
            selected.append(idx)
            current_mask = candidate_mask
            current_lift = candidate_lift
            current_hit_rate = candidate_hit_rate

    return selected, current_mask


def random_path_search(masks, target, n_combinations=50, max_rules_per_set=6,
                       hit_ratio_threshold=0.35, min_hit_rate=0.02, top_k=5):
    n = len(masks)
    overall_bad_rate = target.mean()
    results = []

    rng = np.random.RandomState(42)
    for _ in range(n_combinations):
        k = rng.randint(1, min(max_rules_per_set, n) + 1)
        indices = rng.choice(n, size=min(k, n), replace=False)
        combined_mask = np.any(masks[indices], axis=0)
        hit_rate = combined_mask.mean()
        if hit_rate < min_hit_rate or hit_rate > hit_ratio_threshold:
            continue
        bad_rate_hit = target[combined_mask].mean()
        lift = bad_rate_hit / overall_bad_rate if overall_bad_rate > 0 else 0
        results.append({
            'indices': list(indices),
            'mask': combined_mask,
            'hit_rate': hit_rate,
            'bad_rate': bad_rate_hit,
            'lift': lift
        })

    if not results:
        return [], np.zeros(len(target), dtype=bool)

    results.sort(key=lambda x: x['lift'], reverse=True)
    top_results = results[:top_k]
    best = top_results[0]
    return best['indices'], best['mask']


def compute_strategy_stats(mask, target):
    hit = mask
    pass_cnt = hit.sum()
    total = len(target)
    bad_hit = target[hit].sum()
    bad_rate = target.mean()

    metrics = {
        '通过率': pass_cnt / total if total > 0 else 0,
        '拒绝率': (total - pass_cnt) / total if total > 0 else 0,
        '命中占比': pass_cnt / total if total > 0 else 0,
        'Lift': (target[hit].mean() / bad_rate) if bad_rate > 0 and pass_cnt > 0 else 0,
        '命中坏账率': target[hit].mean() if pass_cnt > 0 else 0,
        '整体坏账率': bad_rate,
        '通过样本量': int(pass_cnt),
        '拒绝样本量': int(total - pass_cnt),
    }
    return metrics
