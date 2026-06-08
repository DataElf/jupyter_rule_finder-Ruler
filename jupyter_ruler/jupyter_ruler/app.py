#!/usr/bin/env python
# coding: utf-8

import ipywidgets as w
from ipywidgets import VBox, HBox, Output, HTML, Button, Layout
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from IPython.display import clear_output, display as ipydisplay

from .rule_mining import (
    generate_demo_rules,
    classify_rules,
    generate_decision_tree_rules,
    generate_random_forest_rules,
    generate_intrees_rules,
)
from .strategy_builder import greedy_lift_select, random_path_search, compute_strategy_stats
from .monitor import aggregate_stats, swap_analysis, plot_strategy_monitor

STYLE = {
    'bg': '#ffffff',
    'section_bg': '#fafbfc',
    'border': '#e1e4e8',
    'accent': '#1a365d',
    'accent_light': '#2b6cb0',
    'text': '#2d3748',
    'text_muted': '#718096',
    'red': '#c53030',
    'orange': '#c05621',
    'green': '#2f855a',
    'gold': '#b7791f',
    'font': '"Segoe UI", "Noto Sans SC", -apple-system, BlinkMacSystemFont, sans-serif',
}

SECTION_CSS = f"""
<style>
.jp-RuleStrategyApp-section {{
    background: {STYLE['section_bg']};
    border: 1px solid {STYLE['border']};
    border-radius: 10px;
    padding: 18px 22px;
    margin: 10px 0;
    box-shadow: 0 1px 3px rgba(0,0,0,0.04);
}}
.jp-RuleStrategyApp-section h3 {{
    font-family: {STYLE['font']};
    font-weight: 600;
    font-size: 15px;
    color: {STYLE['accent']};
    margin: 0 0 12px 0;
    padding-bottom: 8px;
    border-bottom: 2px solid {STYLE['accent_light']};
    letter-spacing: 0.5px;
}}
</style>
"""


class RuleStrategyApp:
    def __init__(self, df, feature_cols, target='target', date_col=None, sample_splits=None,
                 select_cols=None):
        self.df = df
        self.feature_cols = feature_cols
        self.target = target
        self.date_col = date_col
        self.select_cols = select_cols or ['apply_time', 'customer_rating', 'pass_label']
        self.samples = sample_splits if sample_splits else {'all': df.copy()}
        self.current_sample_key = 'all'

        self.rule_df = generate_demo_rules(df, feature_cols, target, n_rules=40)
        self.high_rules, self.mid_rules, self.low_rules = classify_rules(
            self.rule_df, high_th=2.5, mid_th=1.8, low_th=1.2)

        self.all_masks = np.array([row['mask'] for _, row in self.rule_df.iterrows()])
        self.target_vals = df[target].values
        self.current_df = df.copy()
        self.current_target_vals = df[target].values

        self.strategy_indices = []
        self.strategy_mask = None

        self.ui = self._create_ui()
        self._init_time_dropdowns()
        self._update_ui()

    def _s_header(self, icon, title):
        return f'<h3>{icon} {title}</h3>'

    def _create_ui(self):
        header_html = f"""
        {SECTION_CSS}
        <div style='text-align:center;padding:18px 0 8px 0;'>
            <span style='font-family:{STYLE["font"]};font-size:22px;font-weight:700;
            color:{STYLE["accent"]};letter-spacing:2px;'>
            Ruler — AntiRisk Lab
            </span>
            <div style='font-family:{STYLE["font"]};font-size:12px;color:{STYLE["text_muted"]};
            margin-top:4px;letter-spacing:1px;'>
            Interactive Rule Mining &amp; Strategy Monitoring for Risk Control
            </div>
        </div>
        <hr style='border:none;border-top:2px solid {STYLE["accent_light"]};opacity:0.3;margin:8px 0 16px 0;'>
        """

        return VBox([
            HTML(header_html),
            self._build_dataset_tab(),
            self._build_rule_search_tab(),
            self._build_strategy_tab(),
            self._build_monitor_tab(),
        ], layout=Layout(width='100%', padding='8px 16px', background=STYLE['bg']))

    def _build_dataset_tab(self):
        field_options = []
        col_name_map = {
            'apply_time': 'Application Time',
            'customer_rating': 'Customer Rating',
            'pass_label': 'Pass Label',
            'manual_review': 'Manual Review',
            'product_name': 'Product',
            'term': 'Term',
        }
        for col in self.select_cols:
            field_options.append((col_name_map.get(col, col), col))

        self.split_btn_time = Button(
            description='Execute Split',
            layout=Layout(width='110px', margin='0 0 0 10px'),
            style={'button_color': STYLE['accent_light'], 'text_color': 'white',
                   'font_weight': 'bold', 'border': 'none', 'border_radius': '6px'}
        )
        self.split_btn_time.on_click(self._perform_split)

        self.split_btn_label = Button(
            description='Execute Split',
            layout=Layout(width='110px', margin='0 0 0 10px'),
            style={'button_color': STYLE['accent_light'], 'text_color': 'white',
                   'font_weight': 'bold', 'border': 'none', 'border_radius': '6px'}
        )
        self.split_btn_label.on_click(self._perform_split)

        self.split_field_time = w.Dropdown(
            options=field_options,
            value=self.select_cols[0] if self.select_cols else 'apply_time',
            description='Field:',
            style={'description_width': 'initial'},
            layout=Layout(width='160px')
        )
        self.split_field_time.observe(self._on_split_field_change, names='value')

        self.split_field_label = w.Dropdown(
            options=field_options,
            value=self.select_cols[0] if self.select_cols else 'apply_time',
            description='Field:',
            style={'description_width': 'initial'},
            layout=Layout(width='160px')
        )
        self.split_field_label.observe(self._on_split_field_change, names='value')

        self.dev_label_dropdown = w.Dropdown(
            options=[], description='Dev:',
            style={'description_width': 'initial'}, layout=Layout(width='160px')
        )
        self.val_label_dropdown = w.Dropdown(
            options=[], description='Val:',
            style={'description_width': 'initial'}, layout=Layout(width='160px')
        )
        self.oot_label_dropdown = w.Dropdown(
            options=[], description='OOT:',
            style={'description_width': 'initial'}, layout=Layout(width='160px')
        )

        self.dev_start_dropdown = w.Dropdown(
            options=[], description='Dev Start:',
            style={'description_width': 'initial'}, layout=Layout(width='160px')
        )
        self.dev_end_dropdown = w.Dropdown(
            options=[], description='Dev End:',
            style={'description_width': 'initial'}, layout=Layout(width='160px')
        )
        self.val_end_dropdown = w.Dropdown(
            options=[], description='OOT Start:',
            style={'description_width': 'initial'}, layout=Layout(width='160px')
        )

        self.time_controls = HBox([
            self.split_field_time, self.dev_start_dropdown,
            self.dev_end_dropdown, self.val_end_dropdown, self.split_btn_time
        ], layout=Layout(margin='6px 0', align_items='center', flex_wrap='wrap'))

        self.label_controls = HBox([
            self.split_field_label, self.dev_label_dropdown,
            self.val_label_dropdown, self.oot_label_dropdown, self.split_btn_label
        ], layout=Layout(margin='6px 0', align_items='center', flex_wrap='wrap'))
        self.label_controls.layout.display = 'none'

        self.dataset_table_html = HTML(value='')
        dataset_base_fig = go.Figure()
        dataset_base_fig.update_layout(
            height=280, font=dict(family='Georgia, serif', size=10, color='#333'),
            paper_bgcolor='white', plot_bgcolor='#fafafa',
            margin=dict(t=10, r=50, b=50, l=50),
            xaxis=dict(title='Month', tickfont=dict(size=9), gridcolor='#eee'),
            yaxis=dict(title='Volume', side='left', gridcolor='#eee'),
            yaxis2=dict(title='Rate', side='right', tickformat='.0%', gridcolor='#f5f5f5', overlaying='y'),
            showlegend=True,
        )
        self.dataset_fig = go.FigureWidget(dataset_base_fig)

        header = self._s_header('\u2756', 'Dataset Information — 数据集信息')
        controls = VBox([self.time_controls, self.label_controls])
        return VBox([HTML(f'<div class="jp-RuleStrategyApp-section">{header}'), controls,
                    self.dataset_table_html, self.dataset_fig,
                    HTML('</div>')])

    def _on_split_field_change(self, change):
        field = change['new']
        if field == 'apply_time':
            self.time_controls.layout.display = ''
            self.label_controls.layout.display = 'none'
            self._init_time_dropdowns()
        else:
            self.time_controls.layout.display = 'none'
            self.label_controls.layout.display = ''
            self._init_label_dropdowns(field)

    def _init_time_dropdowns(self):
        if self.date_col and self.date_col in self.df.columns:
            dates = pd.to_datetime(self.df[self.date_col])
            min_date = dates.min()
            max_date = dates.max()
            year_months = pd.date_range(start=min_date, end=max_date, freq='MS')
            options = [(d.strftime('%Y-%m'), d) for d in year_months]
            if len(options) > 2:
                self.dev_start_dropdown.options = options[:-2]
                self.dev_start_dropdown.value = options[0][1]
                self.dev_end_dropdown.options = options[1:-1]
                self.dev_end_dropdown.value = options[len(options)//3][1] if len(options) > 3 else options[1][1]
                self.val_end_dropdown.options = options[2:]
                self.val_end_dropdown.value = options[len(options)*2//3][1] if len(options) > 6 else options[-1][1]

    def _init_label_dropdowns(self, field):
        if field in self.df.columns:
            unique_vals = sorted(self.df[field].dropna().unique())
            options = [(str(v), v) for v in unique_vals]
            self.dev_label_dropdown.options = options
            self.val_label_dropdown.options = options
            self.oot_label_dropdown.options = options
            if len(options) >= 1:
                self.dev_label_dropdown.value = unique_vals[0]
            if len(options) >= 2:
                self.val_label_dropdown.value = unique_vals[1]
            if len(options) >= 3:
                self.oot_label_dropdown.value = unique_vals[-1]

    def _perform_split(self, _):
        df = self.df.copy()
        if self.time_controls.layout.display != 'none':
            split_field = self.split_field_time.value
        else:
            split_field = self.split_field_label.value

        if split_field == 'apply_time':
            df['date'] = pd.to_datetime(df[split_field])
            df = df.sort_values('date')
            dev_start = self.dev_start_dropdown.value
            dev_end = self.dev_end_dropdown.value
            val_end = self.val_end_dropdown.value
            df_dev = df[(df['date'] >= dev_start) & (df['date'] < dev_end)].copy()
            df_val = df[(df['date'] >= dev_end) & (df['date'] < val_end)].copy()
            df_oot = df[df['date'] >= val_end].copy()
            self.samples = {'Dev': df_dev, 'Val': df_val, 'OOT': df_oot}
        else:
            dev_label = self.dev_label_dropdown.value
            val_label = self.val_label_dropdown.value
            oot_label = self.oot_label_dropdown.value
            df_dev = df[df[split_field] == dev_label].copy()
            df_val = df[df[split_field] == val_label].copy()
            df_oot = df[df[split_field] == oot_label].copy()
            self.samples = {'Dev': df_dev, 'Val': df_val, 'OOT': df_oot}

        self.current_sample_key = 'Dev'
        self.sample_dropdown.options = list(self.samples.keys())
        self.sample_dropdown.value = 'Dev'
        self._update_sample_data()
        self._update_dataset_info()

    def _update_sample_data(self):
        current_df = self.samples.get(self.current_sample_key, self.df)
        self.current_df = current_df
        self.current_target_vals = current_df[self.target].values

    def _update_dataset_info(self):
        data = []
        for name, sample_df in self.samples.items():
            total = len(sample_df)
            if total == 0:
                continue
            pass_count = total
            if 'pass_label' in sample_df.columns:
                pass_count = (sample_df['pass_label'] == 1).sum()
            reject_count = total - pass_count
            bad_count = sample_df[self.target].sum()
            bad_rate = bad_count / total if total > 0 else 0
            manual_review_count = 0
            if 'manual_review' in sample_df.columns:
                manual_review_count = (sample_df['manual_review'].notna()).sum()
            manual_review_rate = manual_review_count / total if total > 0 else 0

            data.append({
                'Dataset': name,
                'Total': total,
                'Pass': pass_count,
                'Reject': reject_count,
                'Pass Rate': f"{pass_count/total:.1%}" if total > 0 else '-',
                'Bad Rate': f"{bad_rate:.2%}",
                'Manual Rate': f"{manual_review_rate:.1%}",
            })

        if not data:
            self.dataset_table_html.value = ''
            with self.dataset_fig.batch_update():
                self.dataset_fig.data = []
            return

        info_df = pd.DataFrame(data)
        info_df_display = info_df.copy()
        for col in ['Total', 'Pass', 'Reject']:
            info_df_display[col] = info_df_display[col].apply(lambda x: f"{x:,}")

        rows_html = '\n'.join(
            '<tr style="border-bottom:1px solid #eee;">'
            + ''.join(f'<td style="padding:6px 12px;text-align:center;">{v}</td>' for v in row)
            + '</tr>'
            for _, row in info_df_display.iterrows()
        )

        th_html = ''.join(
            f'<th style="padding:8px 12px;border-bottom:2px solid {STYLE["accent_light"]};text-align:center;">{c}</th>'
            for c in info_df_display.columns
        )

        self.dataset_table_html.value = f"""<div style="margin:8px 0;">
        <table style="width:100%;border-collapse:collapse;font-family:Georgia,serif;font-size:12px;">
        <thead><tr style="background:#f1f5f9;color:{STYLE['accent']};font-weight:700;">
        {th_html}
        </tr></thead><tbody>
        {rows_html}
        </tbody></table></div>"""

        chart_data = []
        for name, sample_df in self.samples.items():
            if len(sample_df) == 0:
                continue
            if 'apply_time' in sample_df.columns:
                sd = sample_df.copy()
                sd['month'] = pd.to_datetime(sd['apply_time']).dt.to_period('M').astype(str)
                grouped = sd.groupby('month').agg(
                    total=('target', 'count'),
                    pass_count=('pass_label', lambda x: (x == 1).sum()) if 'pass_label' in sd.columns else ('target', 'count'),
                    bad_count=('target', 'sum'),
                ).reset_index()
                grouped['pass_rate'] = grouped['pass_count'] / grouped['total']
                grouped['bad_rate'] = grouped['bad_count'] / grouped['total']
                grouped['Dataset'] = name
                chart_data.append(grouped)

        with self.dataset_fig.batch_update():
            self.dataset_fig.data = []
            if not chart_data:
                return

            all_chart = pd.concat(chart_data, ignore_index=True)
            colors = ['#1f77b4', '#ff7f0e', '#2ca02c']

            for di, dataset in enumerate(all_chart['Dataset'].unique()):
                ds_data = all_chart[all_chart['Dataset'] == dataset]
                c = colors[di % len(colors)]
                r, g, b = _h2r(c)
                self.dataset_fig.add_bar(
                    x=ds_data['month'], y=ds_data['total'],
                    name=f'{dataset} Vol.',
                    marker_color=f'rgba({r},{g},{b},0.35)',
                    yaxis='y',
                )
                self.dataset_fig.add_scatter(
                    x=ds_data['month'], y=ds_data['pass_rate'],
                    name=f'{dataset} Pass Rate', line=dict(color=c, width=2),
                    yaxis='y2',
                )
                self.dataset_fig.add_scatter(
                    x=ds_data['month'], y=ds_data['bad_rate'],
                    name=f'{dataset} Bad Rate', line=dict(color=c, width=1.5, dash='dash'),
                    yaxis='y2',
                )

            self.dataset_fig.update_layout(
                barmode='group',
                xaxis=dict(title='Month', tickfont=dict(size=9), domain=[0, 1]),
                yaxis=dict(title='Volume', side='left', gridcolor='#eee'),
                yaxis2=dict(title='Rate', side='right', tickformat='.0%', gridcolor='#f5f5f5', overlaying='y'),
                hovermode='x unified',
                legend=dict(orientation='h', yanchor='bottom', y=1.05, xanchor='center', x=0.5, font=dict(size=9)),
            )

    def _build_rule_search_tab(self):
        self.min_leaf_ratio = w.FloatSlider(
            value=0.02, min=0.001, max=0.1, step=0.005,
            description='Min Leaf Ratio:',
            style={'description_width': 'initial'},
            layout=Layout(width='180px')
        )
        self.max_rules = w.IntSlider(
            value=4, min=2, max=8, step=1,
            description='Rule Depth:',
            style={'description_width': 'initial'},
            layout=Layout(width='170px')
        )
        self.n_trees = w.IntSlider(
            value=100, min=10, max=500, step=10,
            description='N Trees:',
            style={'description_width': 'initial'},
            layout=Layout(width='170px')
        )
        self.max_features_ratio = w.FloatSlider(
            value=0.5, min=0.1, max=1.0, step=0.1,
            description='Feature Ratio:',
            style={'description_width': 'initial'},
            layout=Layout(width='180px')
        )
        self.progress_bar = w.IntProgress(
            value=0, min=0, max=100,
            description='',
            style={'bar_color': STYLE['accent_light']},
            layout=Layout(width='100%')
        )
        self.progress_label = HTML(value='')

        self.decision_tree_btn = Button(
            icon='tree', description=' Decision Tree',
            layout=Layout(width='155px', margin='0 4px'),
            style={'button_color': '#f0f4ff', 'text_color': STYLE['accent'],
                   'font_weight': 'bold', 'border': f'1px solid {STYLE["accent_light"]}', 'border_radius': '6px'}
        )
        self.decision_tree_btn.on_click(self._run_decision_tree)

        self.random_forest_btn = Button(
            icon='tree', description=' Random Forest',
            layout=Layout(width='155px', margin='0 4px'),
            style={'button_color': '#f0f4ff', 'text_color': STYLE['accent'],
                   'font_weight': 'bold', 'border': f'1px solid {STYLE["accent_light"]}', 'border_radius': '6px'}
        )
        self.random_forest_btn.on_click(self._run_random_forest)

        self.intrees_btn = Button(
            icon='sitemap', description=' inTrees',
            layout=Layout(width='155px', margin='0 4px'),
            style={'button_color': '#f0f4ff', 'text_color': STYLE['accent'],
                   'font_weight': 'bold', 'border': f'1px solid {STYLE["accent_light"]}', 'border_radius': '6px'}
        )
        self.intrees_btn.on_click(self._run_intrees)

        self.high_th_slider = w.FloatSlider(value=2.5, min=1.0, max=5.0, step=0.1,
            description='High Lift≥', style={'description_width': 'initial', 'handle_color': STYLE['red']},
            layout=Layout(width='280px'))
        self.mid_th_slider = w.FloatSlider(value=1.8, min=1.0, max=3.0, step=0.1,
            description='Mid Lift≥', style={'description_width': 'initial', 'handle_color': STYLE['gold']},
            layout=Layout(width='280px'))
        self.low_th_slider = w.FloatSlider(value=1.2, min=1.0, max=2.0, step=0.1,
            description='Low Lift≥', style={'description_width': 'initial', 'handle_color': STYLE['green']},
            layout=Layout(width='280px'))

        self.high_th_slider.observe(self._on_threshold_change, names='value')
        self.mid_th_slider.observe(self._on_threshold_change, names='value')
        self.low_th_slider.observe(self._on_threshold_change, names='value')

        self.high_dropdown = w.SelectMultiple(
            options=[], description='',
            layout=Layout(width='32%', height='230px'),
            style={'font_size': '10px', 'description_width': '0px'}
        )
        self.mid_dropdown = w.SelectMultiple(
            options=[], description='',
            layout=Layout(width='32%', height='230px'),
            style={'font_size': '10px', 'description_width': '0px'}
        )
        self.low_dropdown = w.SelectMultiple(
            options=[], description='',
            layout=Layout(width='32%', height='230px'),
            style={'font_size': '10px', 'description_width': '0px'}
        )

        self.high_label = HTML(f"<b style='color:{STYLE['red']};font-size:12px;'>&#9670; High Lift</b>")
        self.mid_label = HTML(f"<b style='color:{STYLE['gold']};font-size:12px;'>&#9670; Mid Lift</b>")
        self.low_label = HTML(f"<b style='color:{STYLE['green']};font-size:12px;'>&#9670; Low Lift</b>")

        self.add_to_strategy_btn = Button(
            description='+ Add to Strategy',
            layout=Layout(width='150px'),
            style={'button_color': STYLE['green'], 'text_color': 'white',
                   'font_weight': 'bold', 'border': 'none', 'border_radius': '6px', 'font_size': '11px'}
        )
        self.add_to_strategy_btn.on_click(self._add_selected_to_strategy)

        params_row = HBox([self.min_leaf_ratio, self.max_rules, self.n_trees, self.max_features_ratio],
                         layout=Layout(margin='4px 0', flex_wrap='wrap'))
        btn_row = HBox([self.decision_tree_btn, self.random_forest_btn, self.intrees_btn],
                      layout=Layout(margin='8px 0'))
        progress_row = VBox([self.progress_label, self.progress_bar],
                           layout=Layout(margin='3px 0'))
        controls = HBox([self.high_th_slider, self.mid_th_slider, self.low_th_slider],
                       layout=Layout(margin='6px 0'))
        label_row = HBox([self.high_label, self.mid_label, self.low_label],
                        layout=Layout(justify_content='space-between'))
        rule_dropdowns = HBox([self.high_dropdown, self.mid_dropdown, self.low_dropdown],
                             layout=Layout(justify_content='space-between'))
        add_btn_row = HBox([self.add_to_strategy_btn],
                          layout=Layout(margin='6px 0', justify_content='center'))

        header = self._s_header('\u2659', 'Rule Mining — 规则发现')
        return VBox([
            HTML(f'<div class="jp-RuleStrategyApp-section">{header}'),
            params_row, btn_row, progress_row,
            controls, label_row, rule_dropdowns, add_btn_row,
            HTML('</div>')
        ])

    def _run_decision_tree(self, _):
        self.progress_bar.value = 0
        self.progress_label.value = f'<span style="color:{STYLE["accent_light"]};">Training Decision Tree...</span>'
        min_samples_leaf = max(5, int(len(self.current_df) * self.min_leaf_ratio.value))
        max_depth = self.max_rules.value
        self.rule_df = generate_decision_tree_rules(
            self.current_df, self.feature_cols, self.target,
            max_depth=max_depth, min_samples_leaf=min_samples_leaf,
            n_rules=40, progress_callback=self._update_progress
        )
        self.progress_bar.value = 100
        self.progress_label.value = f'<span style="color:{STYLE["green"]};">Decision Tree complete. {len(self.rule_df)} rules generated.</span>'
        self._refresh_rules()

    def _run_random_forest(self, _):
        self.progress_bar.value = 0
        self.progress_label.value = f'<span style="color:{STYLE["accent_light"]};">Training Random Forest...</span>'
        min_samples_leaf = max(5, int(len(self.current_df) * self.min_leaf_ratio.value))
        n_trees = self.n_trees.value
        max_depth = self.max_rules.value
        max_features_ratio = self.max_features_ratio.value
        self.rule_df = generate_random_forest_rules(
            self.current_df, self.feature_cols, self.target,
            n_trees=n_trees, max_depth=max_depth,
            min_samples_leaf=min_samples_leaf,
            max_features='sqrt' if max_features_ratio >= 0.9 else max_features_ratio,
            n_rules=25, progress_callback=self._update_progress
        )
        self.progress_bar.value = 100
        self.progress_label.value = f'<span style="color:{STYLE["green"]};">Random Forest complete. {len(self.rule_df)} rules generated.</span>'
        self._refresh_rules()

    def _run_intrees(self, _):
        self.progress_bar.value = 0
        self.progress_label.value = f'<span style="color:{STYLE["accent_light"]};">Training inTrees...</span>'
        min_samples_leaf = max(5, int(len(self.current_df) * self.min_leaf_ratio.value))
        n_trees = self.n_trees.value
        max_depth = self.max_rules.value
        max_features_ratio = self.max_features_ratio.value
        self.rule_df = generate_intrees_rules(
            self.current_df, self.feature_cols, self.target,
            n_trees=n_trees, max_depth=max_depth,
            min_samples_leaf=min_samples_leaf,
            max_features='sqrt' if max_features_ratio >= 0.9 else max_features_ratio,
            n_rules=50, progress_callback=self._update_progress
        )
        self.progress_bar.value = 100
        self.progress_label.value = f'<span style="color:{STYLE["green"]};">inTrees complete. {len(self.rule_df)} rules generated.</span>'
        self._refresh_rules()

    def _update_progress(self, progress, message=''):
        self.progress_bar.value = int(progress)
        if message:
            self.progress_label.value = f'<span style="color:{STYLE["accent_light"]};">{message}</span>'

    def _refresh_rules(self):
        self.all_masks = np.array([row['mask'] for _, row in self.rule_df.iterrows()])
        self.high_rules, self.mid_rules, self.low_rules = classify_rules(
            self.rule_df,
            high_th=self.high_th_slider.value,
            mid_th=self.mid_th_slider.value,
            low_th=self.low_th_slider.value
        )
        self._update_rule_search_tab()

    def _on_threshold_change(self, change):
        try:
            if change is None or change.get('new') is None:
                return
            self.high_rules, self.mid_rules, self.low_rules = classify_rules(
                self.rule_df,
                high_th=self.high_th_slider.value,
                mid_th=self.mid_th_slider.value,
                low_th=self.low_th_slider.value
            )
            self._update_rule_search_tab()
        except Exception:
            pass

    def _update_rule_search_tab(self):
        def format_rules(df):
            if df.empty:
                return [('(No rules match)', -1)]
            options = []
            for i, (_, row) in enumerate(df.iterrows()):
                vc = row.get('var_count', 1)
                label = f"[H:{row['hit_rate']:.1%}] [L:{row['lift']:.2f}] [V{vc}] {row['rule']}"
                options.append((label, i))
            return options

        self.high_dropdown.options = format_rules(self.high_rules)
        self.mid_dropdown.options = format_rules(self.mid_rules)
        self.low_dropdown.options = format_rules(self.low_rules)

    def _build_strategy_tab(self):
        self.hit_ratio_cap = w.FloatSlider(
            value=0.35, min=0.05, max=0.80, step=0.05,
            description='Hit Ratio Cap:',
            style={'description_width': 'initial'},
            layout=Layout(width='220px')
        )
        self.random_n_comb = w.IntSlider(
            value=50, min=10, max=200, step=10,
            description='N Combinations:',
            style={'description_width': 'initial'},
            layout=Layout(width='220px')
        )
        self.random_hit_th = w.FloatSlider(
            value=0.35, min=0.05, max=0.80, step=0.05,
            description='Random Hit Cap:',
            style={'description_width': 'initial'},
            layout=Layout(width='220px')
        )

        self.greedy_btn = Button(
            description='Greedy (Lift Priority)',
            layout=Layout(width='200px', margin='0 4px'),
            style={'button_color': STYLE['accent'], 'text_color': 'white',
                   'font_weight': 'bold', 'border': 'none', 'border_radius': '6px'}
        )
        self.greedy_btn.on_click(self._run_greedy)

        self.random_path_btn = Button(
            description='Random Path',
            layout=Layout(width='180px', margin='0 4px'),
            style={'button_color': '#553c9a', 'text_color': 'white',
                   'font_weight': 'bold', 'border': 'none', 'border_radius': '6px'}
        )
        self.random_path_btn.on_click(self._run_random_path)

        self.strategy_display = w.SelectMultiple(
            options=[], description='',
            layout=Layout(width='100%', height='130px'),
            style={'font_size': '10px', 'description_width': '0px'}
        )
        self.stats_display = Output()

        self.remove_from_strategy_btn = Button(
            description='\u2212 Remove Selected',
            layout=Layout(width='150px'),
            style={'button_color': '#fff5f5', 'text_color': STYLE['red'],
                   'font_weight': 'bold', 'border': f'1px solid {STYLE["red"]}', 'border_radius': '6px', 'font_size': '11px'}
        )
        self.remove_from_strategy_btn.on_click(self._remove_selected_from_strategy)

        greedy_params = HBox([self.hit_ratio_cap],
                           layout=Layout(margin='4px 0'))
        random_params = HBox([self.random_n_comb, self.random_hit_th],
                           layout=Layout(margin='4px 0'))
        btn_row = HBox([self.greedy_btn, self.random_path_btn],
                      layout=Layout(margin='8px 0'))

        strategy_box = VBox([
            HTML(f"<b style='color:{STYLE['accent']};'>Strategy Rule Set</b>"),
            self.strategy_display,
            self.remove_from_strategy_btn
        ], layout=Layout(width='48%', height='auto', overflow='auto',
                        border=f'1px solid {STYLE["border"]}', padding='10px', background='white', border_radius='8px'))

        stats_box = VBox([
            HTML(f"<b style='color:{STYLE['accent']};'>Strategy Metrics</b>"),
            self.stats_display
        ], layout=Layout(width='48%', height='auto', overflow='visible',
                        border=f'1px solid {STYLE["border"]}', padding='10px', background='white', border_radius='8px'))

        bottom_row = HBox([strategy_box, stats_box], layout=Layout(margin='8px 0'))

        header = self._s_header('\u2656', 'Strategy Integration — 策略集成')
        return VBox([
            HTML(f'<div class="jp-RuleStrategyApp-section">{header}'),
            greedy_params, random_params, btn_row, bottom_row,
            HTML('</div>')
        ])

    def _run_greedy(self, _):
        idx, mask = greedy_lift_select(
            self.all_masks, self.current_target_vals,
            max_rules=10, lift_threshold=0.05,
            hit_increment_threshold=0.002,
            hit_ratio_cap=self.hit_ratio_cap.value
        )
        self._update_strategy(idx, mask)

    def _run_random_path(self, _):
        idx, mask = random_path_search(
            self.all_masks, self.current_target_vals,
            n_combinations=self.random_n_comb.value,
            max_rules_per_set=6,
            hit_ratio_threshold=self.random_hit_th.value,
            min_hit_rate=0.02,
            top_k=5
        )
        self._update_strategy(idx, mask)

    def _add_selected_to_strategy(self, _):
        selected_indices = []
        selected_indices.extend(self.high_dropdown.value)
        selected_indices.extend(self.mid_dropdown.value)
        selected_indices.extend(self.low_dropdown.value)
        if not selected_indices or -1 in selected_indices:
            return
        current_set = set(self.strategy_indices) if self.strategy_indices else set()
        for idx in selected_indices:
            if idx >= 0:
                current_set.add(idx)
        self.strategy_indices = sorted(list(current_set))
        self._recompute_strategy_mask()

    def _remove_selected_from_strategy(self, _):
        selected_in_strategy = self.strategy_display.value
        if not selected_in_strategy:
            return
        to_remove = set(selected_in_strategy)
        self.strategy_indices = [i for i in self.strategy_indices if i not in to_remove]
        self._recompute_strategy_mask()

    def _recompute_strategy_mask(self):
        if not self.strategy_indices:
            self.strategy_mask = None
            self._update_strategy_display()
            return
        n_target = len(self.current_target_vals)
        combined_mask = np.zeros(n_target, dtype=bool)
        for idx in self.strategy_indices:
            if idx >= len(self.rule_df):
                continue
            mask = self.rule_df.iloc[idx]['mask']
            if len(mask) != n_target:
                continue
            combined_mask |= mask.astype(bool) if not isinstance(mask, np.ndarray) else mask
        self.strategy_mask = combined_mask
        self._update_strategy_display()

    def _update_strategy(self, indices, mask):
        self.strategy_indices = list(indices) if hasattr(indices, '__iter__') else [indices]
        self.strategy_mask = mask
        self._update_strategy_display()

    def _update_strategy_display(self, method=''):
        self.stats_display.clear_output(wait=True)
        strategy_options = []
        for idx in self.strategy_indices:
            if idx < len(self.rule_df):
                row = self.rule_df.iloc[idx]
                strategy_options.append((f"L:{row['lift']:.2f} H:{row['hit_rate']:.1%} | {row['rule']}", idx))
        self.strategy_display.options = strategy_options

        with self.stats_display:
            if self.strategy_mask is not None and self.strategy_mask.sum() > 0:
                try:
                    n_mask = len(self.strategy_mask)
                    n_target = len(self.current_target_vals)
                    if n_mask <= n_target:
                        vals = self.current_target_vals[:n_mask]
                    else:
                        vals = self.current_target_vals
                        self.strategy_mask = self.strategy_mask[:n_target]
                    stats = compute_strategy_stats(self.strategy_mask, vals)
                    rows = [
                        ('Lift', f"{stats['Lift']:.2f}", STYLE['accent_light']),
                        ('Pass Rate', f"{stats['通过率']:.1%}", STYLE['text']),
                        ('Reject Rate', f"{stats['拒绝率']:.1%}", STYLE['text']),
                        ('Hit Ratio', f"{stats['命中占比']:.1%}", STYLE['text']),
                        ('Hit Bad Rate', f"{stats['命中坏账率']:.2%}", STYLE['red']),
                        ('Overall Bad Rate', f"{stats['整体坏账率']:.2%}", STYLE['text_muted']),
                    ]
                    rows_html = ''.join(
                        f'<tr><td style="padding:4px 10px;border-bottom:1px solid #f0f0f0;font-weight:600;">{n}</td>'
                        f'<td style="padding:4px 10px;border-bottom:1px solid #f0f0f0;text-align:right;color:{c};font-weight:500;">{v}</td></tr>'
                        for n, v, c in rows
                    )
                    ipydisplay(HTML(f"""
                    <table style="width:100%;font-size:12px;font-family:Georgia,serif;border-collapse:collapse;">
                    {rows_html}
                    </table>
                    """))
                except Exception:
                    ipydisplay(HTML('<div style="font-size:12px;color:#c53030;">Error computing metrics. Please re-run strategy building.</div>'))
            else:
                ipydisplay(HTML('<div style="font-size:12px;color:#999;">No strategy configured</div>'))

    def _build_monitor_tab(self):
        self.freq_dropdown = w.Dropdown(
            options=[('Day', 'D'), ('Week', 'W'), ('Month', 'M')],
            value='W', description='Frequency:',
            style={'description_width': 'initial'}, layout=Layout(width='160px')
        )
        self.sample_dropdown = w.Dropdown(
            options=list(self.samples.keys()),
            value=self.current_sample_key,
            description='Sample:',
            style={'description_width': 'initial'}, layout=Layout(width='200px')
        )
        self.sample_dropdown.observe(self._on_sample_change, names='value')

        self.update_monitor_btn = Button(
            description='Show Monitor',
            layout=Layout(width='130px'),
            style={'button_color': STYLE['accent_light'], 'text_color': 'white',
                   'font_weight': 'bold', 'border': 'none', 'border_radius': '6px'}
        )
        self.update_monitor_btn.on_click(lambda _: self._update_monitor_plots())

        self.monitor_output = Output()
        with self.monitor_output:
            ipydisplay(HTML(f'<div style="text-align:center;padding:60px;color:{STYLE["text_muted"]};'
                          f'font-family:Georgia,serif;font-size:13px;">Click "Show Monitor" to generate plots</div>'))

        controls = HBox([self.sample_dropdown, self.freq_dropdown, self.update_monitor_btn],
                        layout=Layout(margin='6px 0', align_items='center'))
        header = self._s_header('\u25C9', 'Strategy Monitor — 策略监控')
        return VBox([
            HTML(f'<div class="jp-RuleStrategyApp-section">{header}'),
            controls, self.monitor_output,
            HTML('</div>')
        ])

    def _on_sample_change(self, change):
        if change is None or change.get('new') is None:
            return
        self.current_sample_key = self.sample_dropdown.value
        self._update_sample_data()

    def _update_monitor_plots(self, change=None):
        if self.strategy_mask is None or self.strategy_mask.sum() == 0:
            with self.monitor_output:
                clear_output(wait=True)
                ipydisplay(HTML(f'<div style="text-align:center;padding:60px;color:{STYLE["text_muted"]};'
                              f'font-family:Georgia,serif;font-size:13px;">Please generate a strategy first</div>'))
            return

        sample_name = self.sample_dropdown.value
        df_sample = self.samples.get(sample_name, self.current_df)
        if len(df_sample) < 10:
            with self.monitor_output:
                clear_output(wait=True)
                ipydisplay(HTML(f'<div style="text-align:center;padding:60px;color:{STYLE["red"]};'
                              f'font-family:Georgia,serif;font-size:13px;">Sample "{sample_name}" has insufficient data</div>'))
            return

        freq = self.freq_dropdown.value
        date_col = self.date_col or 'apply_time'
        if date_col not in df_sample.columns:
            with self.monitor_output:
                clear_output(wait=True)
                ipydisplay(HTML(f'<div style="text-align:center;padding:60px;color:{STYLE["red"]};'
                              f'font-family:Georgia,serif;font-size:13px;">Date column "{date_col}" not found</div>'))
            return

        np.random.seed(42)
        base_mask = np.random.choice([True, False], len(df_sample), p=[0.5, 0.5])
        new_mask_src = self.strategy_mask
        if len(new_mask_src) < len(df_sample):
            new_mask_src = np.pad(new_mask_src, (0, len(df_sample) - len(new_mask_src)), constant_values=False)
        new_mask = new_mask_src[:len(df_sample)]

        target_vals = df_sample[self.target].values

        try:
            base_stats = aggregate_stats(df_sample, date_col, base_mask, self.target, freq)
            new_stats = aggregate_stats(df_sample, date_col, new_mask, self.target, freq)
            swap_s = swap_analysis(base_mask, new_mask, target_vals)
            fig = plot_strategy_monitor(base_stats, new_stats, freq, swap_s)

            with self.monitor_output:
                clear_output(wait=True)
                ipydisplay(fig)
        except Exception as e:
            import traceback
            with self.monitor_output:
                clear_output(wait=True)
                ipydisplay(HTML(f'<div style="text-align:center;padding:60px;color:{STYLE["red"]};'
                              f'font-family:Georgia,serif;font-size:13px;">Error: {str(e)}</div>'))
            traceback.print_exc()

    def _update_ui(self):
        self._update_rule_search_tab()
        self._update_dataset_info()

    def display(self):
        return self.ui


def _h2r(hex_color):
    hex_color = hex_color.lstrip('#')
    return tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))
