#!/usr/bin/env python
# coding: utf-8

import pandas as pd
import numpy as np
import re
from itertools import combinations
from sklearn.tree import DecisionTreeClassifier, export_text
from sklearn.ensemble import RandomForestClassifier

_COND_RE = re.compile(r'(\w+)\s*(<=|>=|<|>)\s*([0-9.]+)')


def _extract_var_conditions(rule_str):
    m = _COND_RE.match(rule_str.strip())
    if m:
        return m.group(1), m.group(2), float(m.group(3))
    return None


def _is_interval_rule(rule_str):
    parts = rule_str.split(' AND ')
    var_conds = {}
    for p in parts:
        info = _extract_var_conditions(p)
        if info is None:
            continue
        var_name, direction, threshold = info
        if var_name not in var_conds:
            var_conds[var_name] = []
        var_conds[var_name].append((direction, threshold))
    for conds in var_conds.values():
        if len(conds) >= 2:
            return True
    return False


def _get_all_var_conditions(rule_str):
    parts = rule_str.split(' AND ')
    var_conds = {}
    for p in parts:
        info = _extract_var_conditions(p)
        if info is None:
            continue
        var_name, direction, threshold = info
        if var_name not in var_conds:
            var_conds[var_name] = []
        var_conds[var_name].append((direction, threshold))
    return var_conds


def _rule_has_duplicate_var(rule_str):
    var_conds = _get_all_var_conditions(rule_str)
    for conds in var_conds.values():
        if len(conds) >= 2:
            return True
    return False


def deduplicate_rules(rules_list, overall_bad_rate, skip_var_check=False):
    if skip_var_check:
        valid_rules = rules_list
    else:
        valid_rules = [r for r in rules_list if not _rule_has_duplicate_var(r['rule'])]
    if not valid_rules:
        return []

    single_var_best = {}
    multi_var_best = {}

    for r in valid_rules:
        var_conds = _get_all_var_conditions(r['rule'])
        n_vars = len(var_conds)
        if n_vars == 1:
            for var_name, conds in var_conds.items():
                direction, threshold = conds[0]
                key = (var_name, direction)
                if key not in single_var_best or r['lift'] > single_var_best[key]['lift']:
                    single_var_best[key] = r
        else:
            sig = tuple(sorted((vn, d) for vn, cds in var_conds.items() for d, _ in cds))
            if sig not in multi_var_best or r['lift'] > multi_var_best[sig]['lift']:
                multi_var_best[sig] = r

    result = list(single_var_best.values()) + list(multi_var_best.values())
    result_df = pd.DataFrame(result)
    if not result_df.empty:
        result_df = result_df.sort_values('lift', ascending=False)
    return result_df.to_dict('records') if not result_df.empty else []


def generate_demo_rules(df, feature_cols, target='target', n_rules=30):
    np.random.seed(42)
    rules = []
    overall_bad_rate = df[target].mean()
    high_lift_feats = ['query_count', 'overdue_count', 'credit_score', 'debt_ratio']

    for i in range(n_rules):
        if i < 5:
            feat = np.random.choice(high_lift_feats)
            threshold = np.percentile(df[feat], np.random.uniform(5, 20))
            direction = '>' if feat in ['query_count', 'overdue_count', 'debt_ratio'] else '<='
        elif i < 15:
            feat = np.random.choice(feature_cols)
            threshold = np.percentile(df[feat], np.random.uniform(10, 30))
            direction = np.random.choice(['<=', '>'])
        else:
            feat = np.random.choice(feature_cols)
            threshold = np.percentile(df[feat], np.random.uniform(20, 80))
            direction = np.random.choice(['<=', '>'])

        rule_str = f"{feat} {direction} {threshold:.2f}"
        mask = (df[feat] <= threshold) if direction == '<=' else (df[feat] > threshold)
        hit_rate = mask.mean()
        if hit_rate < 0.02:
            continue
        bad_rate_hit = df.loc[mask, target].mean()
        lift = bad_rate_hit / overall_bad_rate if overall_bad_rate > 0 else 1.0

        if i < 5 and lift < 3.0:
            lift = 3.0 + np.random.uniform(1.0, 2.0)
            bad_rate_hit = lift * overall_bad_rate
        elif i < 15 and lift < 2.0:
            lift = 2.0 + np.random.uniform(0.0, 1.5)
            bad_rate_hit = lift * overall_bad_rate

        rules.append({
            'rule': rule_str,
            'hit_rate': hit_rate,
            'bad_rate': bad_rate_hit,
            'lift': lift,
            'mask': mask.values
        })

    valid = deduplicate_rules(rules, overall_bad_rate)
    return pd.DataFrame(valid)


def classify_rules(rule_df, high_th=2.5, mid_th=1.8, low_th=1.2):
    high = rule_df[rule_df['lift'] >= high_th].copy()
    mid = rule_df[(rule_df['lift'] >= mid_th) & (rule_df['lift'] < high_th)].copy()
    low = rule_df[(rule_df['lift'] >= low_th) & (rule_df['lift'] < mid_th)].copy()
    high['level'] = 'High'
    mid['level'] = 'Mid'
    low['level'] = 'Low'
    return high.reset_index(drop=True), mid.reset_index(drop=True), low.reset_index(drop=True)


def _parse_tree_to_rules(clf, df, feature_cols, target, progress_callback=None):
    rules_list = []
    overall_bad_rate = df[target].mean()
    n_samples = len(df)
    target_vals = df[target].values
    data = df[feature_cols].values

    trees = clf.estimators_ if hasattr(clf, 'estimators_') else [clf]
    total_trees = len(trees)

    for tree_idx, tree in enumerate(trees):
        tree_str = export_text(tree, feature_names=feature_cols)
        lines = tree_str.strip().split('\n')

        for line in lines:
            line = line.strip()
            if 'class:' in line or not line:
                continue
            parts = line.split('|')
            if len(parts) < 2:
                continue

            rule_parts = []
            var_set = set()
            mask = np.ones(n_samples, dtype=bool)

            for part in parts[1:]:
                part = part.strip()
                matched = False
                for feat in feature_cols:
                    if feat not in part:
                        continue
                    if '<=' in part:
                        threshold = float(part.split('<=')[1].strip())
                        rule_parts.append(f"{feat} <= {threshold:.2f}")
                        mask &= data[:, feature_cols.index(feat)] <= threshold
                    elif '>=' in part:
                        threshold = float(part.split('>=')[1].strip())
                        rule_parts.append(f"{feat} >= {threshold:.2f}")
                        mask &= data[:, feature_cols.index(feat)] >= threshold
                    elif '<' in part:
                        threshold = float(part.split('<')[1].strip())
                        rule_parts.append(f"{feat} < {threshold:.2f}")
                        mask &= data[:, feature_cols.index(feat)] < threshold
                    elif '>' in part:
                        threshold = float(part.split('>')[1].strip())
                        rule_parts.append(f"{feat} > {threshold:.2f}")
                        mask &= data[:, feature_cols.index(feat)] > threshold
                    var_set.add(feat)
                    matched = True
                    break
                if not matched:
                    mask = None
                    break

            if mask is None or not rule_parts:
                continue

            rule_str = ' AND '.join(rule_parts)
            hit_rate = mask.mean()
            if hit_rate < 0.01:
                continue
            bad_rate_hit = target_vals[mask].mean()
            lift = bad_rate_hit / overall_bad_rate if overall_bad_rate > 0 else 1.0
            if lift > 1.0:
                rules_list.append({
                    'rule': rule_str,
                    'hit_rate': hit_rate,
                    'bad_rate': bad_rate_hit,
                    'lift': lift,
                    'mask': mask,
                    'var_count': len(var_set)
                })

        if progress_callback and total_trees > 1:
            progress_callback(20 + int(60 * (tree_idx + 1) / total_trees),
                            f'Processing tree {tree_idx + 1}/{total_trees}')

    return rules_list


def generate_decision_tree_rules(df, feature_cols, target='target', max_depth=3,
                                  min_samples_leaf=5, n_rules=25, progress_callback=None):
    X = df[feature_cols].values
    y = df[target].values

    clf = DecisionTreeClassifier(
        max_depth=max_depth,
        min_samples_leaf=max(min_samples_leaf, 5),
        class_weight='balanced',
        random_state=42,
        splitter='best'
    )
    clf.fit(X, y)

    if progress_callback:
        progress_callback(30, 'Decision Tree trained')

    rules_list = _parse_tree_to_rules(clf, df, feature_cols, target, progress_callback)

    if progress_callback:
        progress_callback(80, 'Deduplicating rules...')

    valid_rules = deduplicate_rules(rules_list, df[target].mean(), skip_var_check=True)
    valid_df = pd.DataFrame(valid_rules)
    if not valid_df.empty:
        valid_df = valid_df.sort_values(['var_count', 'lift'], ascending=[False, False]).head(n_rules)

    if progress_callback:
        progress_callback(95, 'Rule generation complete')

    return valid_df.reset_index(drop=True) if not valid_df.empty else pd.DataFrame()


def generate_random_forest_rules(df, feature_cols, target='target', n_trees=10,
                                  max_depth=4, min_samples_leaf=5,
                                  max_features='sqrt', n_rules=25, progress_callback=None):
    X = df[feature_cols].values
    y = df[target].values

    n_features = X.shape[1]
    if isinstance(max_features, str):
        if max_features == 'sqrt':
            mf = int(max(1, np.sqrt(n_features)))
        elif max_features == 'log2':
            mf = int(max(1, np.log2(n_features)))
        else:
            mf = min(n_features, max(1, int(float(max_features) * n_features)))
    else:
        mf = min(n_features, max(1, int(max_features)))

    clf = RandomForestClassifier(
        n_estimators=n_trees,
        max_depth=max_depth,
        min_samples_leaf=max(min_samples_leaf, 5),
        max_features=mf,
        class_weight='balanced',
        random_state=42,
        n_jobs=-1,
        bootstrap=True,
        oob_score=False
    )
    clf.fit(X, y)

    if progress_callback:
        progress_callback(20, 'Random Forest trained')

    rules_list = _parse_tree_to_rules(clf, df, feature_cols, target, progress_callback)

    if progress_callback:
        progress_callback(85, 'Deduplicating rules...')

    valid_rules = deduplicate_rules(rules_list, df[target].mean(), skip_var_check=True)
    valid_df = pd.DataFrame(valid_rules)
    if not valid_df.empty:
        valid_df = valid_df.sort_values(['var_count', 'lift'], ascending=[False, False]).head(n_rules)

    if progress_callback:
        progress_callback(95, 'Rule generation complete')

    return valid_df.reset_index(drop=True) if not valid_df.empty else pd.DataFrame()
