"""
M05_Statistics.py — Generic statistical analysis utilities.

Design principles:
- All stat functions accept column names as parameters (no hardcoded column names).
- Plot functions are unified (one boxplot, one lineplot, one topomap).
- Reuses get_sorted_eeg_channels from M01 for channel alias handling.
- Dataset-specific column name constants live at the top for easy override.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import mne
from scipy import stats

from utils_store.M01_DataLoader import get_sorted_eeg_channels

# ── Dataset-specific column names (import these in page files) ──────────────
BENZO_COL  = 'BENZO_24HBEFORE_EEG (0=NO/1=YES))'
TIMING_COL = 'TIMEINTERVAL_STROKEONSET_EEG'
DELRAT_COL = 'DELRAT_DEF (0= no delirium, 1=delirium)'
PT_ID_COL  = 'PT_ID'


# ── Data Loading & Merging ──────────────────────────────────────────────────

def load_fooof_csv(uploaded_file):
    """Load a FOOOF feature CSV exported by M03. Extracts PT_ID from filename column."""
    df = pd.read_csv(uploaded_file)
    df[PT_ID_COL] = df['Unnamed: 0'].str.extract(r'P(\d+)').astype(int)
    return df.drop(columns=['Unnamed: 0'])


def load_clinical_xlsx(uploaded_file):
    return pd.read_excel(uploaded_file)


def merge_features_clinical(df_deli, df_nodeli, df_clinical):
    """Concatenate Deli + NoDeli feature tables then inner-join with clinical data."""
    return pd.merge(
        pd.concat([df_deli, df_nodeli], ignore_index=True),
        df_clinical, on=PT_ID_COL, how='inner'
    )


# ── Feature Column Helpers ──────────────────────────────────────────────────

def get_feature_columns(df, prefix):
    """Return all columns whose name starts with 'prefix_' (e.g. 'OS', 'EXP', 'CF')."""
    return [c for c in df.columns if c.startswith(prefix + '_')]


def get_aperiodic_columns(df):
    """Shortcut: return (os_cols, exp_cols) for the FOOOF aperiodic features."""
    return get_feature_columns(df, 'OS'), get_feature_columns(df, 'EXP')


# ── Statistical Tests ───────────────────────────────────────────────────────

def compute_ttest(df, feature_cols, group_col, group_vals=(0, 1), group_names=None):
    """
    Welch independent t-test for each feature between two groups.

    Parameters
    ----------
    group_col   : column that defines the two groups (any binary column)
    group_vals  : (val_g0, val_g1) — the two values to compare
    group_names : (name_g0, name_g1) — display labels; defaults to str(group_vals)
    """
    v0, v1 = group_vals
    n0 = str(group_names[0]) if group_names else str(v0)
    n1 = str(group_names[1]) if group_names else str(v1)

    g0 = df[df[group_col] == v0]
    g1 = df[df[group_col] == v1]

    rows = []
    for col in feature_cols:
        a = g0[col].dropna().values
        b = g1[col].dropna().values
        t, p = stats.ttest_ind(a, b, equal_var=False)
        rows.append({
            'Feature'                   : col,
            'Channel'                   : col.split('_', 1)[1],
            f'Mean {n0} (n={len(a)})'   : round(float(a.mean()), 4),
            f'Mean {n1} (n={len(b)})'   : round(float(b.mean()), 4),
            't-statistic'               : round(float(t), 4),
            'p-value'                   : round(float(p), 4),
            'Significant (p<0.05)'      : bool(p < 0.05),
        })
    return pd.DataFrame(rows)


def assign_groups(df, value_col, bins=None, labels=None):
    """
    Assign group labels to a numeric column and return a new DataFrame.

    If bins/labels are given  → pd.cut (categorical binning).
    Otherwise                 → use original values as group labels.

    Returns
    -------
    df_g       : DataFrame with new column '_group'
    group_col  : name of the new column (always '_group')
    group_order: ordered list of group labels
    """
    df = df.copy()
    if bins is not None and labels is not None:
        df['_group'] = pd.cut(df[value_col], bins=bins,
                              labels=labels).astype(str)
        order = list(labels)
    else:
        df['_group'] = df[value_col].astype(str)
        order = sorted(df['_group'].dropna().unique(),
                       key=lambda x: float(x) if x.lstrip('-').replace('.', '', 1).isdigit() else x)
    return df, '_group', order


def compute_anova(df_g, feature_cols, group_col, group_order):
    """
    One-way ANOVA for each feature across pre-defined groups.

    Parameters
    ----------
    df_g        : DataFrame that already contains the group column
    group_col   : name of the column that defines the groups
    group_order : list of group labels in display order

    Returns
    -------
    results_df   : per-feature F-statistic and p-value
    excluded     : groups skipped because n < 2
    """
    excluded, rows = [], []
    for col in feature_cols:
        groups = []
        for g in group_order:
            vals = df_g[df_g[group_col] == g][col].dropna().values
            if len(vals) >= 2:
                groups.append(vals)
            elif g not in excluded:
                excluded.append(g)
        if len(groups) < 2:
            continue
        f, p = stats.f_oneway(*groups)
        rows.append({
            'Feature'              : col,
            'Channel'              : col.split('_', 1)[1],
            'F-statistic'          : round(float(f), 4),
            'p-value'              : round(float(p), 4),
            'Significant (p<0.05)' : bool(p < 0.05),
        })
    return pd.DataFrame(rows), excluded


def compute_spearman(df, feature_cols, x_col):
    """
    Spearman correlation between a continuous column (x_col) and each feature.
    Works for timing, age, NIHSS, or any numeric variable.
    """
    rows = []
    for col in feature_cols:
        valid = df[[col, x_col]].dropna()
        rho, p = stats.spearmanr(valid[x_col], valid[col])
        rows.append({
            'Feature'              : col,
            'Channel'              : col.split('_', 1)[1],
            'Spearman ρ'           : round(float(rho), 4),
            'p-value'              : round(float(p), 4),
            'Significant (p<0.05)' : bool(p < 0.05),
        })
    return pd.DataFrame(rows)


# ── Internal helpers ────────────────────────────────────────────────────────

def _sig_label(p):
    if p < 0.001: return '***'
    if p < 0.01:  return '**'
    if p < 0.05:  return '*'
    return 'ns'


def _make_eeg_info(channels):
    """
    Create an MNE Info object for a list of channel names.
    Applies the standard T3/T4/T5/T6 → T7/T8/P7/P8 aliases automatically
    (reusing get_sorted_eeg_channels from M01 for the alias map).
    Returns (info, renamed_channels) or (None, None) on failure.
    """
    # get_sorted_eeg_channels already handles aliases internally;
    # we replicate the alias map here just for the rename step.
    _ALIASES = {'T3': 'T7', 'T4': 'T8', 'T5': 'P7', 'T6': 'P8'}
    renamed = [_ALIASES.get(ch, ch) for ch in channels]
    try:
        info = mne.create_info(ch_names=renamed, ch_types='eeg', sfreq=250)
        info.set_montage(mne.channels.make_standard_montage('standard_1020'),
                         on_missing='ignore')
        return info, renamed
    except Exception:
        return None, None


# ── Unified plot functions ──────────────────────────────────────────────────

def plot_boxplot_groups(df, value_col, group_col, group_order,
                        title=None, xlabel='', ylabel='Value',
                        value_map=None, palette=None, p_value=None,
                        test_label='p'):
    """
    Generic boxplot + strip plot for any grouping.

    Parameters
    ----------
    value_col  : column to plot on Y axis (numeric)
    group_col  : column that defines the groups (string or numeric)
    group_order: ordered list of group values for the X axis
    value_map  : optional dict to rename group values for display
                 e.g. {0: 'No Benzo', 1: 'Benzo'}
    p_value    : if provided, adds a significance bracket on top
    test_label : label prefix for the p-value annotation
    """
    df = df.copy()
    display_col = '_display_group'
    df[display_col] = df[group_col].map(value_map) if value_map else df[group_col]
    display_order = [value_map[g] if value_map else g for g in group_order]
    display_order = [g for g in display_order if g in df[display_col].values]

    n_groups = len(display_order)
    fig, ax = plt.subplots(figsize=(max(5, n_groups * 1.8), 5))

    sns.boxplot(data=df, x=display_col, y=value_col,
                palette=palette or 'Set2', width=0.5,
                order=display_order, ax=ax)
    sns.stripplot(data=df, x=display_col, y=value_col,
                  color='black', alpha=0.4, size=4, jitter=True,
                  order=display_order, ax=ax)

    if p_value is not None and n_groups == 2:
        ymin = df[value_col].min()
        ymax = df[value_col].max()
        rng  = ymax - ymin if ymax != ymin else 1.0
        top  = ymax + rng * 0.10
        ax.plot([0, 0, 1, 1], [top, top + rng*0.03, top + rng*0.03, top],
                lw=1.2, color='black')
        ax.text(0.5, top + rng*0.04, _sig_label(p_value),
                ha='center', va='bottom', fontsize=13)
        ax.set_ylim(top=top + rng * 0.22)

    t = title or value_col
    p_txt = f'  ({test_label} = {p_value:.4f})' if p_value is not None else ''
    ax.set_title(t + p_txt, fontweight='bold')
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(True, axis='y', linestyle='--', alpha=0.5)
    plt.xticks(rotation=15)
    plt.tight_layout()
    return fig


def plot_lineplot_groups(df, value_col, group_col, group_order,
                         title=None, xlabel='', ylabel='Value'):
    """
    Line plot of group means with shaded ±SD band (no error bar lines).
    Works for any grouping: timing, benzo, delirium, etc.
    """
    valid_order = [g for g in group_order if g in df[group_col].values]
    means, stds, ns = [], [], []
    for g in valid_order:
        vals = df[df[group_col] == g][value_col].dropna()
        means.append(vals.mean())
        stds.append(vals.std())
        ns.append(len(vals))

    x = list(range(len(valid_order)))
    fig, ax = plt.subplots(figsize=(max(6, len(valid_order) * 1.6), 5))
    ax.plot(x, means, 'o-', linewidth=2, markersize=8,
            color='#2196F3', label='Mean ± SD')
    ax.fill_between(x,
                    [m - s for m, s in zip(means, stds)],
                    [m + s for m, s in zip(means, stds)],
                    alpha=0.15, color='#2196F3')
    for xi, (m, n) in enumerate(zip(means, ns)):
        ax.annotate(f'n={n}', (xi, m), textcoords='offset points',
                    xytext=(0, 14), ha='center', fontsize=9, color='#555')
    ax.set_xticks(x)
    ax.set_xticklabels(valid_order, rotation=15)
    ax.set_title(title or value_col, fontweight='bold')
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(True, axis='y', linestyle='--', alpha=0.5)
    ax.legend(fontsize=9)
    plt.tight_layout()
    return fig


def plot_pvalue_bar(results_df, feature_prefix, title):
    """-log10(p) bar chart across channels. Red = significant, blue = not."""
    sub = results_df[results_df['Feature'].str.startswith(feature_prefix + '_')].copy()
    if sub.empty:
        return None
    sub['-log10(p)'] = -np.log10(sub['p-value'].clip(lower=1e-10))
    threshold = -np.log10(0.05)

    # Sort channels using M01's standard order
    sorted_channels = get_sorted_eeg_channels(sub['Channel'].tolist())
    sub['Channel'] = pd.Categorical(sub['Channel'], categories=sorted_channels, ordered=True)
    sub = sub.sort_values('Channel')

    fig, ax = plt.subplots(figsize=(12, 2.8))
    colors = ['#d62728' if v >= threshold else '#aec7e8' for v in sub['-log10(p)']]
    ax.bar(sub['Channel'], sub['-log10(p)'], color=colors, edgecolor='white')
    ax.axhline(threshold, color='red', linestyle='--', linewidth=1.2,
               label=f'p = 0.05  (−log10 = {threshold:.2f})')
    ax.set_title(title, fontweight='bold')
    ax.set_ylabel('-log10(p-value)')
    ax.set_xlabel('Channel')
    ax.legend(fontsize=9)
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    return fig


def plot_topomap_pvalues(results_df, feature_prefix, title):
    """-log10(p) topomap. Reuses _make_eeg_info for MNE setup."""
    sub = results_df[results_df['Feature'].str.startswith(feature_prefix + '_')].copy()
    if sub.empty:
        return None

    channels  = sub['Channel'].tolist()
    neg_log_p = -np.log10(sub['p-value'].clip(lower=1e-10).values)
    info, _   = _make_eeg_info(channels)
    if info is None:
        return None

    fig, ax = plt.subplots(figsize=(4.5, 4.5))
    vmax = max(float(neg_log_p.max()), -np.log10(0.05) + 0.5)
    im, _ = mne.viz.plot_topomap(neg_log_p, info, axes=ax,
                                  cmap='Reds', vlim=(0, vmax), show=False)
    cbar = plt.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label('-log10(p)')
    cbar.ax.axhline(-np.log10(0.05), color='black', linestyle='--', linewidth=1)
    ax.set_title(title, fontweight='bold', pad=10)
    plt.tight_layout()
    return fig


def get_channels_from_feature_df(df):
    """Extract sorted unique channel names from a FOOOF-style feature DataFrame.
    Works with any DataFrame whose columns follow the '{PREFIX}_{CHANNEL}' pattern."""
    known = {'CF', 'BW', 'PW', 'OS', 'EXP'}
    channels = {c.split('_', 1)[1] for c in df.columns
                if '_' in c and c.split('_', 1)[0] in known}
    return get_sorted_eeg_channels(list(channels))


def highlight_sig(row):
    """Pandas Styler row function: highlight rows where Significant==True."""
    return ['background-color: #ffe0e0' if row.get('Significant (p<0.05)', False) else ''
            for _ in row]
