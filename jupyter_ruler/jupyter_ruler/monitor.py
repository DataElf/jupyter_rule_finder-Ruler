#!/usr/bin/env python
# coding: utf-8

import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def aggregate_stats(df, date_col, mask, target, freq='D'):
    df = df.copy()
    df['date'] = pd.to_datetime(df[date_col])
    df['hit'] = mask.astype(bool)
    df['is_bad'] = df[target].astype(int)
    resampled = df.set_index('date').resample(freq)
    grouped = pd.DataFrame()
    grouped['date'] = resampled['hit'].count().index
    grouped['total'] = resampled['hit'].count().values
    grouped['hit_count'] = resampled['hit'].sum().values
    grouped['bad_total'] = resampled[target].sum().values

    bad_hit_vals = []
    for label, grp in df.set_index('date').resample(freq):
        if len(grp) == 0:
            bad_hit_vals.append(np.nan)
            continue
        hit_grp = grp[grp['hit']]
        bad_hit_vals.append(hit_grp[target].sum() if len(hit_grp) > 0 else 0)
    grouped['bad_hit'] = bad_hit_vals

    grouped['pass_rate'] = grouped['hit_count'] / grouped['total'].replace(0, np.nan)
    grouped['hit_ratio'] = grouped['hit_count'] / grouped['total'].replace(0, np.nan)
    grouped['bad_rate_pass'] = grouped['bad_hit'] / grouped['hit_count'].replace(0, np.nan)
    grouped['bad_rate_all'] = grouped['bad_total'] / grouped['total'].replace(0, np.nan)
    return grouped


def swap_analysis(base_mask, new_mask, target):
    both_pass = base_mask & new_mask
    base_pass_new_reject = base_mask & ~new_mask
    base_reject_new_pass = ~base_mask & new_mask
    both_reject = ~base_mask & ~new_mask

    total = len(target)

    def _quadrant_stats(mask):
        cnt = int(mask.sum())
        bad_cnt = int(target[mask].sum()) if cnt > 0 else 0
        good_cnt = cnt - bad_cnt
        bad_rate = bad_cnt / cnt if cnt > 0 else 0.0
        return {
            'cnt': cnt,
            'ratio': cnt / total if total > 0 else 0,
            'good_cnt': good_cnt,
            'bad_cnt': bad_cnt,
            'bad_rate': bad_rate
        }

    def _strategy_bad_rate(mask):
        cnt = int(mask.sum())
        if cnt > 0:
            return float(target[mask].mean())
        return None

    return {
        'both_pass': _quadrant_stats(both_pass),
        'swap_out': _quadrant_stats(base_pass_new_reject),
        'swap_in': _quadrant_stats(base_reject_new_pass),
        'both_reject': _quadrant_stats(both_reject),
        'base_total_passes': int(base_mask.sum()),
        'new_total_passes': int(new_mask.sum()),
        'base_bad_rates': {
            'both_pass': _strategy_bad_rate(both_pass),
            'swap_out': _strategy_bad_rate(base_pass_new_reject),
            'swap_in': None,
            'both_reject': None,
        },
        'new_bad_rates': {
            'both_pass': _strategy_bad_rate(both_pass),
            'swap_out': None,
            'swap_in': _strategy_bad_rate(base_reject_new_pass),
            'both_reject': None,
        },
    }


def _hex_to_rgb(hex_color):
    hex_color = hex_color.lstrip('#')
    return tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))


def plot_strategy_monitor(base_stats, new_stats, freq_name, swap_stats, scatter_data=None):
    freq_labels = {'D': 'Day', 'W': 'Week', 'M': 'Month'}
    freq_label = freq_labels.get(freq_name, freq_name)

    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=(
            f'{freq_label} Pass Rate & Bad Rate',
            'Pass Rate vs Bad Rate',
            'Swap In/Out Quadrant Analysis',
            ''
        ),
        specs=[
            [{'secondary_y': True}, {'type': 'xy'}],
            [{'secondary_y': True}, {'type': 'xy'}],
        ],
        row_heights=[0.48, 0.52],
        column_widths=[0.55, 0.45],
        horizontal_spacing=0.08,
        vertical_spacing=0.12
    )

    C_BLUE = '#1f77b4'
    C_ORANGE = '#ff7f0e'
    C_RED = '#d62728'
    C_GREEN = '#2ca02c'
    C_GRAY = '#7f7f7f'
    C_PURPLE = '#9467bd'

    all_bad_rates_line = []

    if base_stats is not None and not base_stats.empty:
        fig.add_trace(
            go.Scatter(
                x=base_stats['date'], y=base_stats['pass_rate'],
                name='Base Pass Rate', mode='lines+markers',
                line=dict(color=C_BLUE, width=2),
                marker=dict(size=4, symbol='circle', color=C_BLUE),
                legendgroup='base', legendgrouptitle_text='Base Strategy'
            ),
            row=1, col=1, secondary_y=False
        )
        valid_br = base_stats['bad_rate_pass'].dropna()
        if not valid_br.empty:
            all_bad_rates_line.extend(valid_br.tolist())
        fig.add_trace(
            go.Scatter(
                x=base_stats['date'], y=base_stats['bad_rate_pass'],
                name='Base Bad Rate', mode='lines+markers',
                line=dict(color=C_BLUE, width=1.6, dash='dash'),
                marker=dict(size=3, symbol='diamond', color=C_BLUE),
                legendgroup='base'
            ),
            row=1, col=1, secondary_y=True
        )

    if new_stats is not None and not new_stats.empty:
        fig.add_trace(
            go.Scatter(
                x=new_stats['date'], y=new_stats['pass_rate'],
                name='New Pass Rate', mode='lines+markers',
                line=dict(color=C_ORANGE, width=2),
                marker=dict(size=4, symbol='circle', color=C_ORANGE),
                legendgroup='new', legendgrouptitle_text='New Strategy'
            ),
            row=1, col=1, secondary_y=False
        )
        valid_br = new_stats['bad_rate_pass'].dropna()
        if not valid_br.empty:
            all_bad_rates_line.extend(valid_br.tolist())
        fig.add_trace(
            go.Scatter(
                x=new_stats['date'], y=new_stats['bad_rate_pass'],
                name='New Bad Rate', mode='lines+markers',
                line=dict(color=C_ORANGE, width=1.6, dash='dash'),
                marker=dict(size=3, symbol='diamond', color=C_ORANGE),
                legendgroup='new'
            ),
            row=1, col=1, secondary_y=True
        )

    if all_bad_rates_line:
        bad_line_min = max(0.0, min(all_bad_rates_line) - 0.02)
        bad_line_max = max(all_bad_rates_line) + 0.02
    else:
        bad_line_min, bad_line_max = 0.0, 0.1

    scatter_bad_rates = []

    if base_stats is not None and not base_stats.empty:
        valid_base = base_stats.dropna(subset=['pass_rate', 'bad_rate_pass'])
        if not valid_base.empty:
            n_pts = len(valid_base)
            colors = [
                f'rgba(31,119,180,{0.25 + 0.75 * i / max(n_pts - 1, 1):.2f})'
                for i in range(n_pts)
            ]
            fig.add_trace(
                go.Scatter(
                    x=valid_base['pass_rate'], y=valid_base['bad_rate_pass'],
                    mode='markers', name='Base',
                    marker=dict(color=colors, size=9, symbol='circle',
                               line=dict(color='white', width=0.8)),
                    legendgroup='base', showlegend=False
                ),
                row=1, col=2
            )
            scatter_bad_rates.extend(valid_base['bad_rate_pass'].tolist())

    if new_stats is not None and not new_stats.empty:
        valid_new = new_stats.dropna(subset=['pass_rate', 'bad_rate_pass'])
        if not valid_new.empty:
            n_pts = len(valid_new)
            colors = [
                f'rgba(255,127,14,{0.25 + 0.75 * i / max(n_pts - 1, 1):.2f})'
                for i in range(n_pts)
            ]
            fig.add_trace(
                go.Scatter(
                    x=valid_new['pass_rate'], y=valid_new['bad_rate_pass'],
                    mode='markers', name='New',
                    marker=dict(color=colors, size=9, symbol='triangle-up',
                               line=dict(color='white', width=0.8)),
                    legendgroup='new', showlegend=False
                ),
                row=1, col=2
            )
            scatter_bad_rates.extend(valid_new['bad_rate_pass'].tolist())

    if scatter_bad_rates:
        s_bad_min = max(0.0, min(scatter_bad_rates) - 0.02)
        s_bad_max = max(scatter_bad_rates) + 0.02
    else:
        s_bad_min, s_bad_max = 0.0, 0.1

    if swap_stats is not None:
        quadrants = [
            ('Q\u2160 Both Pass', 'both_pass', C_GREEN),
            ('Q\u2161 Base\u2192Reject', 'swap_out', C_RED),
            ('Q\u2162 New\u2192Pass', 'swap_in', C_PURPLE),
            ('Q\u2163 Both Reject', 'both_reject', C_GRAY),
        ]

        base_total_passes = swap_stats.get('base_total_passes', 0)
        new_total_passes = swap_stats.get('new_total_passes', 0)

        x_labels = []
        bar_offset = 0

        base_ratios = []
        new_ratios = []

        for _qi, (_qlabel, qkey, _qcolor) in enumerate(quadrants):
            q = swap_stats[qkey]
            cnt = q['cnt']

            if base_total_passes > 0 and qkey in ('both_pass', 'swap_out'):
                base_ratios.append(cnt / base_total_passes)
            else:
                base_ratios.append(0)

            if new_total_passes > 0 and qkey in ('both_pass', 'swap_in'):
                new_ratios.append(cnt / new_total_passes)
            else:
                new_ratios.append(0)

        base_bar_x = []
        new_bar_x = []
        all_bar_x_for_bad = []

        for _qi, (qlabel, qkey, _qcolor) in enumerate(quadrants):
            q = swap_stats[qkey]

            bl = bar_offset
            nl = bar_offset + 0.95
            mid = bar_offset + 0.475

            base_bar_x.append(bl)
            new_bar_x.append(nl)
            all_bar_x_for_bad.append(mid)
            x_labels.append(qlabel)

            r, g, b = _hex_to_rgb(C_BLUE)
            fig.add_trace(
                go.Bar(
                    x=[bl], y=[base_ratios[_qi]],
                    name='Base Strategy' if _qi == 0 else '',
                    marker=dict(color=f'rgba({r},{g},{b},0.8)',
                               line=dict(color=C_BLUE, width=1.2)),
                    width=0.88, showlegend=(_qi == 0),
                    legendgroup='swap_base',
                    text=[f"{base_ratios[_qi]:.1%}"],
                    textposition='outside', textfont=dict(color=C_BLUE, size=8, family='Georgia,serif'),
                    hovertemplate=f'<b>Base</b><br>Ratio: {base_ratios[_qi]:.1%}<br>Count: {q["cnt"]:,}<extra></extra>'
                ),
                row=2, col=1, secondary_y=False
            )

            rr, gg, bb = _hex_to_rgb(C_ORANGE)
            fig.add_trace(
                go.Bar(
                    x=[nl], y=[new_ratios[_qi]],
                    name='New Strategy' if _qi == 0 else '',
                    marker=dict(color=f'rgba({rr},{gg},{bb},0.8)',
                               line=dict(color=C_ORANGE, width=1.2)),
                    width=0.88, showlegend=(_qi == 0),
                    legendgroup='swap_new',
                    text=[f"{new_ratios[_qi]:.1%}"],
                    textposition='outside', textfont=dict(color=C_ORANGE, size=8, family='Georgia,serif'),
                    hovertemplate=f'<b>New</b><br>Ratio: {new_ratios[_qi]:.1%}<br>Count: {q["cnt"]:,}<extra></extra>'
                ),
                row=2, col=1, secondary_y=False
            )

            bar_offset += 2.4

        swap_bad_rates_all = []
        for _qi, (_qlabel, qkey, _qcolor) in enumerate(quadrants):
            bbr = swap_stats.get('base_bad_rates', {}).get(qkey)
            nbr = swap_stats.get('new_bad_rates', {}).get(qkey)

            mid_x = all_bar_x_for_bad[_qi]

            pair_x = []
            pair_y = []
            pair_vars = ('base_bad', bbr, C_BLUE, 'circle'), ('new_bad', nbr, C_ORANGE, 'triangle-up')

            if bbr is not None and nbr is not None:
                pair_x = [base_bar_x[_qi] + 0.44, new_bar_x[_qi] + 0.44]
                pair_y = [bbr, nbr]
                swap_bad_rates_all.extend([bbr, nbr])
            elif bbr is not None:
                pair_x = [mid_x]
                pair_y = [bbr]
                swap_bad_rates_all.append(bbr)
            elif nbr is not None:
                pair_x = [mid_x]
                pair_y = [nbr]
                swap_bad_rates_all.append(nbr)

            if not pair_x:
                continue

            if bbr is not None:
                fig.add_trace(
                    go.Scatter(
                        x=[pair_x[0]], y=[pair_y[0]],
                        mode='markers', name='Base Bad Rate' if _qi == 0 else '',
                        marker=dict(size=10, symbol='circle', color=C_BLUE,
                                   line=dict(color='white', width=1.5)),
                        showlegend=(_qi == 0),
                        legendgroup='swap_bad_base', legendgrouptitle_text='Bad Rate by Quadrant',
                        hovertemplate=f'<b>Base Bad</b>: {pair_y[0]:.2%}<extra></extra>'
                    ),
                    row=2, col=1, secondary_y=True
                )

            if nbr is not None:
                fig.add_trace(
                    go.Scatter(
                        x=[pair_x[-1]], y=[pair_y[-1]],
                        mode='markers', name='New Bad Rate' if _qi == 0 else '',
                        marker=dict(size=10, symbol='triangle-up', color=C_ORANGE,
                                   line=dict(color='white', width=1.5)),
                        showlegend=(_qi == 0),
                        legendgroup='swap_bad_new',
                        hovertemplate=f'<b>New Bad</b>: {pair_y[-1]:.2%}<extra></extra>'
                    ),
                    row=2, col=1, secondary_y=True
                )

            if bbr is not None and nbr is not None:
                mid_bad_x = base_bar_x[_qi] + 0.44
                mid_bad_x2 = new_bar_x[_qi] + 0.44
                fig.add_trace(
                    go.Scatter(
                        x=[mid_bad_x, mid_bad_x, mid_bad_x2, mid_bad_x2],
                        y=[bbr, bbr, nbr, nbr],
                        mode='lines', name='' if _qi > 0 else 'Swap Differential',
                        line=dict(color=C_RED, width=1.8, dash='dot'),
                        showlegend=(_qi == 0),
                        legendgroup='swap_diff',
                        hoverinfo='skip'
                    ),
                    row=2, col=1, secondary_y=True
                )

        tick_vals = [(base_bar_x[i] + new_bar_x[i]) / 2 for i in range(len(x_labels))]

        if swap_bad_rates_all:
            sw_bad_min = max(0.0, min(swap_bad_rates_all) - 0.02)
            sw_bad_max = max(swap_bad_rates_all) + 0.02
        else:
            sw_bad_min, sw_bad_max = 0.0, 0.1

        fig.update_xaxes(
            tickvals=tick_vals, ticktext=x_labels, tickfont=dict(size=7, color='#444'),
            row=2, col=1
        )
        fig.update_yaxes(
            title_text="Through Ratio per Strategy", row=2, col=1, secondary_y=False,
            tickformat='.0%', gridcolor='#e5e5e5', title_font=dict(size=9),
            rangemode='nonnegative'
        )
        fig.update_yaxes(
            title_text="Bad Rate", row=2, col=1, secondary_y=True,
            tickformat='.1%', gridcolor='#f0f0f0', title_font=dict(size=9),
            range=[sw_bad_min, sw_bad_max]
        )

    fig.update_layout(
        height=700,
        showlegend=True,
        hovermode='closest',
        legend=dict(
            orientation="h", yanchor="bottom", y=1.03,
            xanchor="center", x=0.5,
            font=dict(size=8),
            bgcolor='rgba(255,255,255,0.9)',
            bordercolor='#e0e0e0', borderwidth=1
        ),
        font=dict(family='Georgia, serif', size=10, color='#333333'),
        paper_bgcolor='white',
        plot_bgcolor='#fafafa',
        margin=dict(t=50, b=50, l=50, r=20),
    )

    fig.update_yaxes(
        title_text="Pass Rate", secondary_y=False, row=1, col=1,
        tickformat='.0%', gridcolor='#e5e5e5', title_font=dict(size=9),
        range=[0, 1]
    )
    fig.update_yaxes(
        title_text="Bad Rate", secondary_y=True, row=1, col=1,
        tickformat='.1%', gridcolor='#f0f0f0', title_font=dict(size=9),
        range=[bad_line_min, bad_line_max]
    )
    fig.update_xaxes(
        title_text="Date", row=1, col=1, gridcolor='#e5e5e5', title_font=dict(size=9)
    )
    fig.update_xaxes(
        title_text="Pass Rate", row=1, col=2, tickformat='.0%',
        gridcolor='#e5e5e5', title_font=dict(size=9),
        rangemode='tozero'
    )
    fig.update_yaxes(
        title_text="Bad Rate", row=1, col=2, tickformat='.1%',
        gridcolor='#e5e5e5', title_font=dict(size=9),
        range=[s_bad_min, s_bad_max]
    )

    return fig
