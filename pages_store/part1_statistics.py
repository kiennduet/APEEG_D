import streamlit as st
import pandas as pd

# ── Reuse from existing modules ──────────────────────────────────────────────
from utils_store.M03_FeatureExtraction import (
    select_features_from_df, select_channels_from_df,
    plot_combined_topomaps, plot_feature_line,
)
from utils_store.M04_Classification import load_features_subjects

from utils_store.M05_Statistics import (
    # data
    load_fooof_csv, load_clinical_xlsx, merge_features_clinical,
    get_aperiodic_columns, get_feature_columns, get_channels_from_feature_df,
    # stats
    compute_ttest, assign_groups, compute_anova, compute_spearman,
    # plots
    plot_boxplot_groups, plot_lineplot_groups,
    plot_pvalue_bar, plot_topomap_pvalues,
    # helpers
    highlight_sig, BENZO_COL, TIMING_COL,
)

_BENZO_MAP   = {0: 'No Benzo', 1: 'Benzo'}
_BENZO_ORDER = [0, 1]
_BENZO_PAL   = {'No Benzo': '#4C72B0', 'Benzo': '#DD8452'}

_TIMING_PRESETS = {
    "3 groups: ≤1 / 2-3 / ≥4 days":        dict(bins=[-1, 1, 3, 999],
                                                  labels=['≤1 day', '2-3 days', '≥4 days']),
    "2 groups: Early (≤2d) / Late (≥3d)":   dict(bins=[-1, 2, 999],
                                                  labels=['Early ≤2d', 'Late ≥3d']),
    "Original values (all distinct days)":  dict(bins=None, labels=None),
}


# ════════════════════════════════════════════════════════════════════════════
#  TAB 1 — GROUP COMPARISON
#  Reuses M03 (topo + line) and M04 (CSV loader).
#  Works with any two FOOOF-style feature CSV files.
# ════════════════════════════════════════════════════════════════════════════

def _show_group_comparison():
    st.markdown(
        "Compare **any two groups** using topographic maps, line plots, "
        "and per-channel t-tests. Upload the feature CSV files produced by "
        "the Feature Extraction step."
    )

    # ── Load two CSV files ───────────────────────────────────────────────────
    st.subheader("Step 1: Upload Feature Tables", divider="blue")
    c1, c2 = st.columns(2)
    with c1:
        f1 = st.file_uploader("📂 Group 1 CSV", type="csv", key="gc_g1")
        n1 = st.text_input("Group 1 name", value="Group 1", key="gc_n1")
    with c2:
        f2 = st.file_uploader("📂 Group 2 CSV", type="csv", key="gc_g2")
        n2 = st.text_input("Group 2 name", value="Group 2", key="gc_n2")

    if not (f1 and f2):
        st.info("Upload both CSV files to continue.")
        return

    df_g1_raw = load_features_subjects(f1)
    df_g2_raw = load_features_subjects(f2)
    st.success(f"Loaded — {n1}: {len(df_g1_raw)} rows | {n2}: {len(df_g2_raw)} rows")

    # ── Feature & channel selection ──────────────────────────────────────────
    st.subheader("Step 2: Select Features & Channels", divider="blue")
    selected_features = st.multiselect(
        "Feature types:",
        options=['CF', 'BW', 'PW', 'OS', 'EXP'],
        default=['OS', 'EXP'],
        key="gc_feat",
    )
    if not selected_features:
        st.warning("Select at least one feature type.")
        return

    all_channels = get_channels_from_feature_df(df_g1_raw)
    selected_channels = st.multiselect(
        "Channels:", all_channels, default=all_channels, key="gc_ch"
    )
    if not selected_channels:
        st.warning("Select at least one channel.")
        return

    # Filter both DataFrames to chosen features + channels
    df_g1 = select_channels_from_df(
        select_features_from_df(df_g1_raw, selected_features), selected_channels)
    df_g2 = select_channels_from_df(
        select_features_from_df(df_g2_raw, selected_features), selected_channels)

    st.markdown("---")

    # ── Analysis ─────────────────────────────────────────────────────────────
    st.subheader("Step 3: Run Analysis", divider="blue")

    if st.button("▶ Run Group Comparison", use_container_width=True, key="gc_run"):
        st.session_state['gc_done']  = True
        st.session_state['gc_g1']    = df_g1
        st.session_state['gc_g2']    = df_g2
        st.session_state['gc_n1']    = n1
        st.session_state['gc_n2']    = n2
        st.session_state['gc_feats'] = selected_features

    if not st.session_state.get('gc_done'):
        return

    df_g1   = st.session_state['gc_g1']
    df_g2   = st.session_state['gc_g2']
    n1      = st.session_state['gc_n1']
    n2      = st.session_state['gc_n2']
    sel_f   = st.session_state['gc_feats']

    # Topographic comparison (reuses M03)
    st.subheader("Topographic Map Comparison", divider="gray")
    try:
        fig_topo = plot_combined_topomaps(df_g1, df_g2, sel_f, n1, n2,
                                          orientation='v')
        st.pyplot(fig_topo)
    except Exception as e:
        st.warning(f"Topomap could not be rendered: {e}")

    # Line plot per feature (reuses M03)
    st.subheader("Line Plot per Feature", divider="gray")
    for feat in sel_f:
        fig_line = plot_feature_line(df_g1, df_g2, feature=feat,
                                     name_g1=n1, name_g2=n2)
        if fig_line:
            st.pyplot(fig_line)

    # T-test per channel (reuses M05)
    st.subheader("T-test per Channel", divider="gray")
    for feat in sel_f:
        feat_cols = [c for c in df_g1.columns if c.startswith(feat + '_')]
        df_combined = pd.concat([
            df_g1[feat_cols].assign(_group=0),
            df_g2[feat_cols].assign(_group=1),
        ], ignore_index=True)
        ttest_res = compute_ttest(df_combined, feat_cols, '_group',
                                  group_vals=(0, 1),
                                  group_names=[n1, n2])
        n_sig = int(ttest_res['Significant (p<0.05)'].sum())
        st.markdown(f"**{feat}** — significant channels: {n_sig} / {len(ttest_res)}")
        st.dataframe(ttest_res.style.apply(highlight_sig, axis=1),
                     use_container_width=True)

        c_bar, c_topo = st.columns([2, 1])
        with c_bar:
            fig = plot_pvalue_bar(ttest_res, feat,
                                  f"{feat} — −log10(p): {n1} vs {n2}")
            if fig: st.pyplot(fig)
        with c_topo:
            fig2 = plot_topomap_pvalues(ttest_res, feat,
                                         f"{feat} p-value map")
            if fig2: st.pyplot(fig2)

        st.download_button(f"⬇ Download {feat} t-test (.csv)",
                           ttest_res.to_csv(index=False),
                           f"ttest_{feat}_{n1}_vs_{n2}.csv", "text/csv",
                           key=f"dl_gc_{feat}")


# ════════════════════════════════════════════════════════════════════════════
#  TAB 2 — CLINICAL ASSOCIATION  (Part 1)
# ════════════════════════════════════════════════════════════════════════════

def _show_clinical_association():
    st.markdown(
        "Study the relation between **aperiodic EEG components** (Offset & Exponent) "
        "and **benzodiazepine use** / **timing of EEG** recording."
    )

    # ── Load Data ─────────────────────────────────────────────────────────────
    st.subheader("Step 1: Load Data", divider="blue")
    c1, c2, c3 = st.columns(3)
    with c1:
        deli_file     = st.file_uploader("📂 Deli.csv", type="csv", key="stat_deli")
    with c2:
        nodeli_file   = st.file_uploader("📂 NoDeli.csv", type="csv", key="stat_nodeli")
    with c3:
        clinical_file = st.file_uploader("📂 Clinical .xlsx",
                                         type=["xlsx", "xls"], key="stat_clin")

    if deli_file and nodeli_file and clinical_file:
        if st.button("🔗 Merge & Load Dataset", use_container_width=True):
            with st.spinner("Merging data…"):
                df_merged = merge_features_clinical(
                    load_fooof_csv(deli_file),
                    load_fooof_csv(nodeli_file),
                    load_clinical_xlsx(clinical_file),
                )
                st.session_state['df_stat'] = df_merged
                for k in ['ttest_res', 'anova_res', 'df_g', 'grp_col', 'grp_order', 'corr_res']:
                    st.session_state.pop(k, None)
            st.success(f"✅ Loaded {len(df_merged)} patients — {df_merged.shape[1]} columns")

    if 'df_stat' not in st.session_state:
        st.info("Upload all 3 files and click **Merge & Load Dataset** to continue.")
        return

    df = st.session_state['df_stat']
    os_cols, exp_cols = get_aperiodic_columns(df)

    with st.expander("📋 Dataset preview"):
        st.dataframe(df.head(10), use_container_width=True)
        bc = df[BENZO_COL].map(_BENZO_MAP).value_counts()
        st.write(f"**Benzodiazepine:** No = {bc.get('No Benzo', 0)} | Yes = {bc.get('Benzo', 0)}")
        st.write("**Timing distribution (days after stroke):**")
        st.bar_chart(df[TIMING_COL].value_counts().sort_index())

    st.markdown("---")

    # ── Feature Selection ─────────────────────────────────────────────────────
    st.subheader("Step 2: Select Aperiodic Feature Type", divider="blue")
    feat_choice = st.radio("Feature:", ["Offset (OS)", "Exponent (EXP)", "Both"],
                           horizontal=True)
    if feat_choice == "Offset (OS)":
        selected_cols, prefixes = os_cols, ['OS']
    elif feat_choice == "Exponent (EXP)":
        selected_cols, prefixes = exp_cols, ['EXP']
    else:
        selected_cols, prefixes = os_cols + exp_cols, ['OS', 'EXP']

    st.markdown("---")

    # ── T-test: Benzodiazepine ────────────────────────────────────────────────
    st.subheader("🧪 T-test: Benzodiazepine Use", divider="orange")
    st.markdown("Compare features between patients **with** and **without** "
                "benzodiazepine in the 24 h before EEG.")

    if st.button("▶ Run T-test", use_container_width=True, key="btn_ttest"):
        with st.spinner("Running t-tests…"):
            st.session_state['ttest_res'] = compute_ttest(
                df, selected_cols, BENZO_COL,
                group_vals=_BENZO_ORDER,
                group_names=['No Benzo', 'Benzo'],
            )

    if 'ttest_res' in st.session_state:
        ttest_res = st.session_state['ttest_res']
        n_sig = int(ttest_res['Significant (p<0.05)'].sum())
        st.info(f"Significant (p < 0.05): **{n_sig} / {len(ttest_res)}**")
        st.dataframe(ttest_res.style.apply(highlight_sig, axis=1), use_container_width=True)
        st.download_button("⬇ Download (.csv)", ttest_res.to_csv(index=False),
                           "ttest_benzo.csv", "text/csv")

        st.subheader("p-value Overview", divider="gray")
        for pfx in prefixes:
            cb, ct = st.columns([2, 1])
            with cb:
                fig = plot_pvalue_bar(ttest_res, pfx,
                                      f"{pfx} — −log10(p)  vs  Benzodiazepine")
                if fig: st.pyplot(fig)
            with ct:
                fig2 = plot_topomap_pvalues(ttest_res, pfx, f"{pfx} topomap")
                if fig2: st.pyplot(fig2)

        st.subheader("Boxplot & Line Plot", divider="gray")
        benzo_opts = {}
        for pfx in prefixes:
            benzo_opts[f"All channels — mean {pfx}"] = ('mean', pfx)
        for col in selected_cols:
            benzo_opts[col] = ('single', col)

        sel_b = st.selectbox("Choose channel / feature:", list(benzo_opts.keys()),
                             key="ttest_box_sel")
        mode_b, payload_b = benzo_opts[sel_b]
        cb_box, cb_line = st.columns(2)

        if mode_b == 'mean':
            pfx = payload_b
            mean_col = f'_mean_{pfx}'
            df[mean_col] = df[get_feature_columns(df, pfx)].mean(axis=1)
            p_val = float(compute_ttest(df, [mean_col], BENZO_COL,
                                        _BENZO_ORDER, ['No Benzo', 'Benzo'])['p-value'].iloc[0])
            df_bl = df.copy(); df_bl['_bl'] = df_bl[BENZO_COL].map(_BENZO_MAP)
            with cb_box:
                st.pyplot(plot_boxplot_groups(df, mean_col, BENZO_COL, _BENZO_ORDER,
                    title=f'Mean {pfx}', value_map=_BENZO_MAP,
                    palette=_BENZO_PAL, p_value=p_val))
            with cb_line:
                st.pyplot(plot_lineplot_groups(df_bl, mean_col, '_bl',
                    ['No Benzo', 'Benzo'],
                    title=f'Mean {pfx} — group means ± SD', ylabel=f'Mean {pfx}'))
        else:
            feat = payload_b
            p_row = ttest_res.loc[ttest_res['Feature'] == feat, 'p-value']
            p_val = float(p_row.values[0]) if len(p_row) else None
            df_bl = df.copy(); df_bl['_bl'] = df_bl[BENZO_COL].map(_BENZO_MAP)
            with cb_box:
                st.pyplot(plot_boxplot_groups(df, feat, BENZO_COL, _BENZO_ORDER,
                    title=feat, value_map=_BENZO_MAP,
                    palette=_BENZO_PAL, p_value=p_val))
            with cb_line:
                st.pyplot(plot_lineplot_groups(df_bl, feat, '_bl',
                    ['No Benzo', 'Benzo'],
                    title=f'{feat} — group means ± SD', ylabel='Value'))

    st.markdown("---")

    # ── ANOVA: Timing ─────────────────────────────────────────────────────────
    st.subheader("📈 ANOVA: Timing of EEG", divider="orange")
    st.markdown("Compare features across groups defined by "
                "**days from stroke onset to EEG recording**.")

    grouping     = st.radio("Grouping method:", list(_TIMING_PRESETS.keys()),
                            key="anova_grouping")
    add_spearman = st.checkbox("Also compute Spearman correlation (timing as continuous)",
                               key="anova_spearman")

    if st.button("▶ Run ANOVA", use_container_width=True, key="btn_anova"):
        with st.spinner("Running ANOVA…"):
            df_g, grp_col, grp_order = assign_groups(
                df, TIMING_COL, **_TIMING_PRESETS[grouping])
            res, excluded = compute_anova(df_g, selected_cols, grp_col, grp_order)
            st.session_state.update({'anova_res': res, 'df_g': df_g,
                                     'grp_col': grp_col, 'grp_order': grp_order})
            if excluded:
                st.warning(f"Groups excluded (n < 2): **{', '.join(excluded)}**")
            if add_spearman:
                st.session_state['corr_res'] = compute_spearman(df, selected_cols, TIMING_COL)
            else:
                st.session_state.pop('corr_res', None)

    if 'anova_res' in st.session_state:
        anova_res = st.session_state['anova_res']
        df_g      = st.session_state['df_g']
        grp_col   = st.session_state['grp_col']
        grp_order = st.session_state['grp_order']

        n_sig = int(anova_res['Significant (p<0.05)'].sum())
        st.info(f"Significant (p < 0.05): **{n_sig} / {len(anova_res)}**")
        st.dataframe(anova_res.style.apply(highlight_sig, axis=1), use_container_width=True)
        st.download_button("⬇ Download ANOVA (.csv)", anova_res.to_csv(index=False),
                           "anova_timing.csv", "text/csv")

        if 'corr_res' in st.session_state:
            st.subheader("Spearman Correlation (continuous timing)", divider="gray")
            corr_res = st.session_state['corr_res']
            st.info(f"Significant: **{int(corr_res['Significant (p<0.05)'].sum())} / {len(corr_res)}**")
            st.dataframe(corr_res.style.apply(highlight_sig, axis=1), use_container_width=True)
            st.download_button("⬇ Download Spearman (.csv)", corr_res.to_csv(index=False),
                               "spearman_timing.csv", "text/csv")

        st.subheader("p-value Overview", divider="gray")
        for pfx in prefixes:
            cb, ct = st.columns([2, 1])
            with cb:
                fig = plot_pvalue_bar(anova_res, pfx,
                                      f"{pfx} — −log10(p)  vs  Timing of EEG")
                if fig: st.pyplot(fig)
            with ct:
                fig2 = plot_topomap_pvalues(anova_res, pfx, f"{pfx} topomap")
                if fig2: st.pyplot(fig2)

        st.subheader("Boxplot & Line Plot", divider="gray")
        anova_opts = {}
        for pfx in prefixes:
            anova_opts[f"All channels — mean {pfx}"] = ('mean', pfx)
        for col in selected_cols:
            anova_opts[col] = ('single', col)

        sel_a = st.selectbox("Choose channel / feature:", list(anova_opts.keys()),
                             key="anova_box_sel")
        mode_a, payload_a = anova_opts[sel_a]
        ca_box, ca_line = st.columns(2)

        if mode_a == 'mean':
            pfx = payload_a
            mean_col = f'_mean_{pfx}'
            df_g[mean_col] = df_g[get_feature_columns(df_g, pfx)].mean(axis=1)
            anova_mean, _ = compute_anova(df_g, [mean_col], grp_col, grp_order)
            p_val = float(anova_mean['p-value'].iloc[0]) if not anova_mean.empty else None
            with ca_box:
                st.pyplot(plot_boxplot_groups(df_g, mean_col, grp_col, grp_order,
                    title=f'Mean {pfx} across all channels',
                    xlabel='Days from stroke onset to EEG',
                    p_value=p_val, test_label='ANOVA p'))
            with ca_line:
                st.pyplot(plot_lineplot_groups(df_g, mean_col, grp_col, grp_order,
                    title=f'Mean {pfx} — group means ± SD',
                    xlabel='Days from stroke onset to EEG', ylabel=f'Mean {pfx}'))
        else:
            feat = payload_a
            p_row = anova_res.loc[anova_res['Feature'] == feat, 'p-value']
            p_val = float(p_row.values[0]) if len(p_row) else None
            with ca_box:
                st.pyplot(plot_boxplot_groups(df_g, feat, grp_col, grp_order,
                    title=feat, xlabel='Days from stroke onset to EEG',
                    p_value=p_val, test_label='ANOVA p'))
            with ca_line:
                st.pyplot(plot_lineplot_groups(df_g, feat, grp_col, grp_order,
                    title=f'{feat} — group means ± SD',
                    xlabel='Days from stroke onset to EEG', ylabel='Value'))


# ════════════════════════════════════════════════════════════════════════════
#  ENTRY POINT
# ════════════════════════════════════════════════════════════════════════════

def show_statistics():
    st.header("📊 Statistical Analysis")
    tab1, tab2 = st.tabs([
        "🔬 Group Comparison",
        "📋 Clinical Association",
    ])
    with tab1:
        _show_group_comparison()
    with tab2:
        _show_clinical_association()
