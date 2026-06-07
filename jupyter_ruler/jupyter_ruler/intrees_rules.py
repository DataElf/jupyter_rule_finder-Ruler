#!/usr/bin/env python
# coding: utf-8

"""inTrees-style rule extraction utilities.

This module keeps the inTrees workflow focused on rule production:
extract full path rules from tree ensembles, score them, prune weak
conditions, remove redundant coverage, and select a compact final rule set.
It intentionally does not implement the STEL prediction model.
"""

import numpy as np
import pandas as pd


def extract_rules(model, feature_names=None, max_depth=None):
    """Extract root-to-leaf path rules from a fitted sklearn tree ensemble."""
    trees = _iter_trees(model)
    rows = []

    if feature_names is None:
        try:
            n_features = trees[0].n_features_in_
        except Exception:
            n_features = trees[0].tree_.n_features
        feature_names = [f"x{i}" for i in range(n_features)]

    for estimator in trees:
        tree = estimator.tree_
        classes = getattr(estimator, "classes_", None)

        def recurse(node_id, conditions, depth):
            left = tree.children_left[node_id]
            right = tree.children_right[node_id]
            is_leaf = left == right

            if (max_depth is not None) and (depth >= max_depth):
                is_leaf = True

            if is_leaf:
                rows.append({
                    "conditions": _rule_to_string(conditions),
                    "pred": _leaf_prediction(tree, node_id, classes),
                })
                return

            feat_idx = tree.feature[node_id]
            threshold = tree.threshold[node_id]
            feat_name = feature_names[feat_idx]

            recurse(left, conditions + [(feat_name, "<=", threshold)], depth + 1)
            recurse(right, conditions + [(feat_name, ">", threshold)], depth + 1)

        recurse(0, [], 0)

    return pd.DataFrame(rows)


def get_rule_metrics(rules, X, y, feature_names=None, positive_label=1):
    """Compute risk-rule metrics and keep rules predicting the positive label."""
    df = _to_dataframe(X, feature_names)
    y_arr = np.asarray(y)
    n_total = len(df)
    total_bad = int(np.sum(y_arr == positive_label))
    base_bad_rate = total_bad / n_total if n_total > 0 else 0.0

    rows = []
    rules_iter = rules.iterrows() if isinstance(rules, pd.DataFrame) else enumerate(rules)

    for _, rule in rules_iter:
        conditions = _get_conditions_from_rule(rule)
        if not conditions:
            continue

        mask = rule_mask(df, conditions)
        hit_samples = int(mask.sum())
        if hit_samples == 0:
            continue

        pred = rule.get("pred") if isinstance(rule, dict) else rule.get("pred")
        if pred is None or (pd.isna(pred) if not isinstance(pred, str) else False):
            pred = _majority_class(y_arr[mask])
        if pred != positive_label:
            continue

        bad_hits = int(np.sum(y_arr[mask] == positive_label))
        bad_rate = bad_hits / hit_samples
        hit_rate = hit_samples / n_total
        recall_rate = bad_hits / total_bad if total_bad > 0 else 0.0
        lift = bad_rate / base_bad_rate if base_bad_rate > 0 else 0.0

        rows.append({
            "len": len(conditions),
            "hit_rate": hit_rate,
            "bad_rate": bad_rate,
            "recall_rate": recall_rate,
            "lift": lift,
            "condition": _get_condition_string(rule, conditions),
            "pred": pred,
            "hit_samples": hit_samples,
            "conditions": conditions,
        })

    return pd.DataFrame(rows)


def prune_rules(rule_metric_df, X, y, max_decay=0.05, type_decay=2,
                feature_names=None, positive_label=1):
    """Remove weak conditions while limiting bad-rate decay."""
    df = _to_dataframe(X, feature_names)
    y_arr = np.asarray(y)
    n_total = len(df)
    total_bad = int(np.sum(y_arr == positive_label))
    base_bad_rate = total_bad / n_total if n_total > 0 else 0.0

    pruned_rows = []
    for _, row in rule_metric_df.iterrows():
        conditions = list(_get_conditions_from_rule(row))
        pred = row["pred"]
        if pred != positive_label or not conditions:
            continue

        base_mask = rule_mask(df, conditions)
        base_bad = int(np.sum(y_arr[base_mask] == positive_label))
        base_bad_rate_rule = base_bad / max(int(base_mask.sum()), 1)
        base_err = 1.0 - base_bad_rate_rule

        i = len(conditions) - 1
        while i >= 0 and len(conditions) > 1:
            new_conditions = conditions[:i] + conditions[i + 1:]
            mask = rule_mask(df, new_conditions)
            if mask.sum() == 0:
                i -= 1
                continue

            new_bad = int(np.sum(y_arr[mask] == positive_label))
            new_bad_rate = new_bad / max(int(mask.sum()), 1)
            new_err = 1.0 - new_bad_rate
            if type_decay == 1:
                decay = (new_err - base_err) / max(base_err, 1e-6)
            else:
                decay = new_err - base_err

            if decay <= max_decay:
                conditions = new_conditions
                base_err = new_err
                i = len(conditions) - 1
            else:
                i -= 1

        mask = rule_mask(df, conditions)
        hit_samples = int(mask.sum())
        if hit_samples == 0:
            continue

        bad_hits = int(np.sum(y_arr[mask] == positive_label))
        bad_rate = bad_hits / hit_samples
        hit_rate = hit_samples / n_total
        recall_rate = bad_hits / total_bad if total_bad > 0 else 0.0
        lift = bad_rate / base_bad_rate if base_bad_rate > 0 else 0.0

        pruned_rows.append({
            "len": len(conditions),
            "hit_rate": hit_rate,
            "bad_rate": bad_rate,
            "recall_rate": recall_rate,
            "lift": lift,
            "condition": _rule_to_string(conditions),
            "pred": pred,
            "hit_samples": hit_samples,
            "conditions": conditions,
        })

    return pd.DataFrame(pruned_rows)


def select_rules(rule_metric_df, min_samples=1, min_bad_rate=0.5,
                 max_len=None, top_k=None):
    """Select rules by simple thresholds and metric ranking."""
    df = rule_metric_df.copy()
    if df.empty:
        return df
    if min_samples is not None:
        df = df[df["hit_samples"] >= min_samples]
    if min_bad_rate is not None:
        df = df[df["bad_rate"] >= min_bad_rate]
    if max_len is not None:
        df = df[df["len"] <= max_len]

    if df.empty:
        return df.reset_index(drop=True)

    df = df.drop_duplicates(subset=["condition", "pred"])
    df = df.sort_values(by=["bad_rate", "hit_rate", "len"],
                        ascending=[False, False, True])
    if top_k is not None:
        df = df.head(top_k)
    return df.reset_index(drop=True)


def remove_redundant_rules(rule_metric_df, X, y=None, feature_names=None,
                           jaccard_thresh=0.85):
    """Remove rules with highly overlapping coverage and the same prediction."""
    df = rule_metric_df.copy()
    if df.empty:
        return df

    df = df.sort_values(by=["bad_rate", "len", "hit_rate"],
                        ascending=[False, True, False]).reset_index(drop=True)
    features = _to_dataframe(X, feature_names)
    masks = [rule_mask(features, _get_conditions_from_rule(row))
             for _, row in df.iterrows()]

    kept_rows = []
    kept_masks = []
    kept_preds = []
    for idx, row in df.iterrows():
        mask = masks[idx]
        pred = row["pred"]
        redundant = False
        for kept_mask, kept_pred in zip(kept_masks, kept_preds):
            if pred != kept_pred:
                continue
            union = (mask | kept_mask).sum()
            if union == 0:
                continue
            jaccard = (mask & kept_mask).sum() / union
            if jaccard >= jaccard_thresh:
                redundant = True
                break
        if not redundant:
            kept_rows.append(row)
            kept_masks.append(mask)
            kept_preds.append(pred)

    return pd.DataFrame(kept_rows).reset_index(drop=True)


def select_rules_rrf(rule_metric_df, X, y, feature_names=None, max_rules=50,
                     min_gain=0.0, lambda_len=0.02, gamma_overlap=0.5,
                     min_samples=1):
    """Select rules with an RRF-style gain, length, and overlap score."""
    df = rule_metric_df.copy()
    if min_samples is not None:
        df = df[df["hit_samples"] >= min_samples].reset_index(drop=True)
    if df.empty:
        return df

    features = _to_dataframe(X, feature_names)
    y_arr = np.asarray(y)
    masks = [rule_mask(features, _get_conditions_from_rule(row))
             for _, row in df.iterrows()]

    base_impurity = _entropy(y_arr)
    gains = []
    for mask in masks:
        p1 = mask.mean()
        if p1 <= 0.0 or p1 >= 1.0:
            gains.append(0.0)
            continue
        gain = base_impurity - (
            p1 * _entropy(y_arr[mask]) + (1 - p1) * _entropy(y_arr[~mask])
        )
        gains.append(float(gain))

    candidates = list(range(len(df)))
    selected = []
    selected_masks = []
    selected_scores = []

    while candidates and len(selected) < max_rules:
        best_idx = None
        best_score = min_gain
        for idx in candidates:
            length_penalty = max(
                0.0, 1.0 - lambda_len * max(df.loc[idx, "len"] - 1, 0)
            )
            max_jaccard = 0.0
            for selected_mask in selected_masks:
                union = (masks[idx] | selected_mask).sum()
                if union == 0:
                    continue
                jaccard = (masks[idx] & selected_mask).sum() / union
                if jaccard > max_jaccard:
                    max_jaccard = jaccard

            overlap_penalty = max(0.0, 1.0 - gamma_overlap * max_jaccard)
            score = gains[idx] * length_penalty * overlap_penalty
            if score > best_score:
                best_score = score
                best_idx = idx

        if best_idx is None:
            break
        selected.append(best_idx)
        selected_masks.append(masks[best_idx])
        selected_scores.append(best_score)
        candidates.remove(best_idx)

    out = df.iloc[selected].copy().reset_index(drop=True)
    out["rrf_score"] = selected_scores
    return out


def rule_mask(df, conditions):
    """Return a boolean mask for a structured condition list."""
    mask = np.ones(len(df), dtype=bool)
    for feat, op, threshold in conditions:
        if op == "<=":
            mask &= df[feat] <= threshold
        elif op == "<":
            mask &= df[feat] < threshold
        elif op == ">":
            mask &= df[feat] > threshold
        elif op == ">=":
            mask &= df[feat] >= threshold
        else:
            raise ValueError(f"Unsupported operator: {op}")
    return mask


def _iter_trees(model):
    if hasattr(model, "tree_"):
        return [model]
    if hasattr(model, "estimators_"):
        estimators = model.estimators_
        if isinstance(estimators, np.ndarray):
            return [estimator for estimator in estimators.ravel()]
        return list(estimators)
    raise ValueError("Unsupported model type for rule extraction")


def _to_dataframe(X, feature_names=None):
    if isinstance(X, pd.DataFrame):
        return X.copy()
    if feature_names is None:
        raise ValueError("X is not a DataFrame; pass feature_names.")
    return pd.DataFrame(X, columns=feature_names)


def _leaf_prediction(tree, node_id, classes=None):
    value = tree.value[node_id]
    if value.ndim == 3:
        value = value[0]
    if value.shape[1] == 1:
        return float(value[0, 0])
    idx = int(np.argmax(value[0]))
    return idx if classes is None else classes[idx]


def _get_conditions_from_rule(rule):
    if isinstance(rule, dict):
        value = rule.get("conditions")
    else:
        value = rule.get("conditions") if "conditions" in rule else None
    if isinstance(value, (list, tuple)):
        return list(value)
    if isinstance(value, str):
        return _parse_condition_string(value)
    raise ValueError("Rule must contain structured or string conditions.")


def _get_condition_string(rule, conditions):
    if isinstance(rule, dict):
        value = rule.get("condition") or rule.get("conditions")
    else:
        value = rule.get("condition") if "condition" in rule else None
        value = value or (rule.get("conditions") if "conditions" in rule else None)
    if isinstance(value, str) and value.strip():
        return value
    return _rule_to_string(conditions)


def _parse_condition_string(condition_str):
    if condition_str is None:
        return []
    text = str(condition_str).strip()
    if text == "" or text.upper() == "TRUE":
        return []
    conditions = []
    for part in text.split("&"):
        clean = part.strip().strip("()").strip()
        matched = False
        for op in ("<=", ">=", "<", ">"):
            if op in clean:
                feat, threshold = clean.split(op, 1)
                conditions.append((feat.strip(), op, float(threshold.strip())))
                matched = True
                break
        if not matched:
            raise ValueError(f"Invalid condition: {part}")
    return conditions


def _rule_to_string(conditions):
    if not conditions:
        return "TRUE"
    parts = []
    for feat, op, threshold in conditions:
        parts.append(f"({feat} {op} {threshold:.6g})")
    return " & ".join(parts)


def _majority_class(y):
    values, counts = np.unique(y, return_counts=True)
    return values[int(np.argmax(counts))]


def _entropy(y):
    if len(y) == 0:
        return 0.0
    _, counts = np.unique(y, return_counts=True)
    p = counts / counts.sum()
    return float(-np.sum(p * np.log2(p + 1e-12)))
