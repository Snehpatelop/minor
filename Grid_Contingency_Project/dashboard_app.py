import os
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px

# ───────────── Page Config ─────────────
st.set_page_config(
    page_title="Grid Contingency Benchmark Dashboard",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ───────────── Paths ─────────────
base_dir = os.path.dirname(os.path.abspath(__file__))
output_dir = os.path.join(base_dir, "outputs")

sa_csv       = os.path.join(base_dir, "benchmark_simulated_annealing.csv")
ga_csv       = os.path.join(base_dir, "benchmark_genetic.csv")
stats_csv    = os.path.join(base_dir, "statistical_comparison.csv")
multi_csv    = os.path.join(base_dir, "multi_system_benchmark.csv")
log_path     = os.path.join(output_dir, "run.log")
image_path   = os.path.join(output_dir, "powergrid_pruned.png")
conv_path    = os.path.join(output_dir, "convergence_comparison.png")
html_path    = os.path.join(output_dir, "top10_n1_contingencies.html")

# ───────────── Custom CSS ─────────────
st.markdown("""
<style>
    .block-container {padding-top: 1rem;}
    .metric-card {
        background: linear-gradient(135deg, #1e3a5f 0%, #2d5986 100%);
        padding: 1.2rem; border-radius: 12px; text-align: center;
        color: white; margin-bottom: 0.5rem;
        box-shadow: 0 4px 6px rgba(0,0,0,0.3);
    }
    .metric-card h3 {margin: 0; font-size: 0.85rem; opacity: 0.85;}
    .metric-card h1 {margin: 0.2rem 0 0 0; font-size: 1.6rem;}
    .section-divider {border-top: 2px solid #444; margin: 1.5rem 0;}
</style>
""", unsafe_allow_html=True)

# ───────────── Header ─────────────
st.markdown("# ⚡ Grid Contingency Benchmark Dashboard")
st.caption("Classical N-1 / N-2 Contingency Analysis  |  IEEE Bus Systems  |  SA vs GA Comparison")

# ───────────── Sidebar ─────────────
with st.sidebar:
    st.header("🔧 Controls")
    show_raw = st.checkbox("Show raw data tables", value=False)
    show_log = st.checkbox("Show run log", value=False)
    st.markdown("---")
    st.markdown("**Generated Files**")
    for fp, label in [(sa_csv, "SA CSV"), (ga_csv, "GA CSV"), (stats_csv, "Stats CSV"),
                       (multi_csv, "Multi-System CSV"), (conv_path, "Convergence PNG"),
                       (image_path, "Grid PNG"), (html_path, "N-1 HTML Chart")]:
        exists = "✅" if os.path.exists(fp) else "❌"
        st.text(f"{exists} {label}")

# ╔══════════════════════════════════════════════╗
# ║  SECTION 1 — KEY METRICS                    ║
# ╚══════════════════════════════════════════════╝
st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
st.subheader("📊 Key Metrics at a Glance")

df_sa = pd.read_csv(sa_csv) if os.path.exists(sa_csv) else None
df_ga = pd.read_csv(ga_csv) if os.path.exists(ga_csv) else None

if df_sa is not None:
    n1_rows = df_sa[df_sa["Type"] == "N-1"]
    n2_sa_row = df_sa[df_sa["Type"].str.contains("N-2", na=False)]
    n2_ga_row = df_ga[df_ga["Type"].str.contains("N-2", na=False)] if df_ga is not None else None

    top_n1_score = n1_rows["Score"].max() if not n1_rows.empty else 0
    sa_n2_score = n2_sa_row["Score"].values[0] if n2_sa_row is not None and not n2_sa_row.empty else 0
    ga_n2_score = n2_ga_row["Score"].values[0] if n2_ga_row is not None and not n2_ga_row.empty else 0
    total_lines = len(n1_rows)

    m1, m2, m3, m4 = st.columns(4)
    with m1:
        st.markdown(f'<div class="metric-card"><h3>Total Lines Scanned</h3><h1>{total_lines}</h1></div>', unsafe_allow_html=True)
    with m2:
        st.markdown(f'<div class="metric-card"><h3>Top N-1 Score</h3><h1>{top_n1_score:.4f}</h1></div>', unsafe_allow_html=True)
    with m3:
        st.markdown(f'<div class="metric-card"><h3>SA N-2 Score</h3><h1>{sa_n2_score:.4f}</h1></div>', unsafe_allow_html=True)
    with m4:
        st.markdown(f'<div class="metric-card"><h3>GA N-2 Score</h3><h1>{ga_n2_score:.4f}</h1></div>', unsafe_allow_html=True)
else:
    st.warning("⚠️ No benchmark data found. Run `setup_and_run.py` first.")

# ╔══════════════════════════════════════════════╗
# ║  SECTION 2 — N-1 CONTINGENCY BAR CHART      ║
# ╚══════════════════════════════════════════════╝
st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
st.subheader("🔴 Top N-1 Contingencies (Single-Line Failures)")

if df_sa is not None:
    n1 = df_sa[df_sa["Type"] == "N-1"].head(15)
    fig_n1 = go.Figure()
    fig_n1.add_trace(go.Bar(
        x=[str(x) for x in n1["Contingency"]],
        y=n1["Score"],
        marker=dict(
            color=n1["Score"],
            colorscale="Reds",
            showscale=True,
            colorbar=dict(title="Score"),
        ),
        text=[f"{s:.4f}" for s in n1["Score"]],
        textposition="outside",
        hovertemplate="Line: %{x}<br>Score: %{y:.4f}<extra></extra>",
    ))
    fig_n1.update_layout(
        xaxis_title="Line Index", yaxis_title="Severity Score",
        template="plotly_dark", height=420,
        title="Top 15 N-1 Contingency Severity Scores",
        margin=dict(t=50, b=40),
    )
    st.plotly_chart(fig_n1, use_container_width=True)

# ╔══════════════════════════════════════════════╗
# ║  SECTION 3 — SA vs GA COMPARISON             ║
# ╚══════════════════════════════════════════════╝
st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
st.subheader("🆚 Simulated Annealing vs Genetic Algorithm")

col_sa, col_ga = st.columns(2)

with col_sa:
    st.markdown("#### Simulated Annealing (N-2)")
    if df_sa is not None and not n2_sa_row.empty:
        st.metric("Best N-2 Score", f"{sa_n2_score:.4f}")
        st.metric("Contingency Lines", n2_sa_row["Contingency"].values[0])
    if show_raw and df_sa is not None:
        st.dataframe(df_sa, use_container_width=True)

with col_ga:
    st.markdown("#### Genetic Algorithm (N-2)")
    if df_ga is not None and n2_ga_row is not None and not n2_ga_row.empty:
        st.metric("Best N-2 Score", f"{ga_n2_score:.4f}")
        st.metric("Contingency Lines", n2_ga_row["Contingency"].values[0])
    if show_raw and df_ga is not None:
        st.dataframe(df_ga, use_container_width=True)

# ╔══════════════════════════════════════════════╗
# ║  SECTION 4 — CONVERGENCE COMPARISON          ║
# ╚══════════════════════════════════════════════╝
st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
st.subheader("📈 Convergence Comparison")

if os.path.exists(conv_path):
    st.image(conv_path, caption="SA vs GA Convergence Over Iterations", use_container_width=True)
else:
    st.info("Convergence plot not found. Run setup_and_run.py.")

# ╔══════════════════════════════════════════════╗
# ║  SECTION 5 — STATISTICAL ANALYSIS            ║
# ╚══════════════════════════════════════════════╝
st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
st.subheader("📐 Statistical Multi-Seed Analysis (10 Runs)")

if os.path.exists(stats_csv):
    df_stats = pd.read_csv(stats_csv)

    col_tbl, col_chart = st.columns([1, 2])
    with col_tbl:
        st.dataframe(df_stats, use_container_width=True, hide_index=True)

    with col_chart:
        fig_stats = go.Figure()
        for i, row in df_stats.iterrows():
            fig_stats.add_trace(go.Bar(
                name=row["Algorithm"],
                x=[row["Algorithm"]],
                y=[row["Mean Score"]],
                error_y=dict(type="data", array=[row["Std Dev"]], visible=True),
                text=f'{row["Mean Score"]:.4f} ± {row["Std Dev"]:.4f}',
                textposition="outside",
            ))
        fig_stats.update_layout(
            yaxis_title="Mean Severity Score",
            template="plotly_dark", height=350, showlegend=False,
            title="Mean Score with Std Dev (10 Seeds)",
            margin=dict(t=50, b=30),
        )
        st.plotly_chart(fig_stats, use_container_width=True)
else:
    st.info("Statistical comparison CSV not found. Run setup_and_run.py.")

# ╔══════════════════════════════════════════════╗
# ║  SECTION 6 — MULTI-SYSTEM BENCHMARK          ║
# ╚══════════════════════════════════════════════╝
st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
st.subheader("🌐 Multi-System Benchmark (IEEE 14 / 30 / 57 / 118)")

if os.path.exists(multi_csv):
    df_multi = pd.read_csv(multi_csv)

    col_tbl2, col_chart2 = st.columns([1, 2])
    with col_tbl2:
        st.dataframe(df_multi, use_container_width=True, hide_index=True)

    with col_chart2:
        fig_multi = go.Figure()
        fig_multi.add_trace(go.Bar(
            name="Top N-1", x=df_multi["System"], y=df_multi["Top N-1 Score"],
            marker_color="#ef553b",
        ))
        fig_multi.add_trace(go.Bar(
            name="SA N-2", x=df_multi["System"], y=df_multi["SA N-2 Score"],
            marker_color="#636efa",
        ))
        fig_multi.add_trace(go.Bar(
            name="GA N-2", x=df_multi["System"], y=df_multi["GA N-2 Score"],
            marker_color="#00cc96",
        ))
        fig_multi.update_layout(
            barmode="group", template="plotly_dark", height=400,
            yaxis_title="Severity Score",
            title="Contingency Scores Across IEEE Bus Systems",
            margin=dict(t=50, b=30),
        )
        st.plotly_chart(fig_multi, use_container_width=True)

    # Scalability: Lines vs Score scatter
    fig_scale = px.scatter(
        df_multi, x="Lines", y="Top N-1 Score", text="System",
        size="Lines", color="System",
        title="System Size vs Top N-1 Score (Scalability)",
        template="plotly_dark",
    )
    fig_scale.update_traces(textposition="top center")
    fig_scale.update_layout(height=350, margin=dict(t=50, b=30))
    st.plotly_chart(fig_scale, use_container_width=True)
else:
    st.info("Multi-system benchmark CSV not found. Run setup_and_run.py.")

# ╔══════════════════════════════════════════════╗
# ║  SECTION 7 — GRID VISUALIZATION              ║
# ╚══════════════════════════════════════════════╝
st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
st.subheader("🗺️ Power Grid Topology (Failed Lines Highlighted)")

if os.path.exists(image_path):
    st.image(image_path, caption="IEEE 118-Bus — Pruned (Failed) Lines in Red", use_container_width=True)
else:
    st.info("Power grid image not found. Run setup_and_run.py.")

# ╔══════════════════════════════════════════════╗
# ║  SECTION 8 — RUN LOG                         ║
# ╚══════════════════════════════════════════════╝
if show_log:
    st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
    st.subheader("📜 Run Log (Last 200 Lines)")
    if os.path.exists(log_path):
        with open(log_path, "r", encoding="utf-8") as f:
            log_lines = f.readlines()[-200:]
        st.code("".join(log_lines), language="log")
    else:
        st.info("Log file not found.")

# ───────────── Footer ─────────────
st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
st.caption("Grid Contingency Benchmark Dashboard • N-1/N-2 Analysis • SA vs GA • Multi-System IEEE Benchmark")
