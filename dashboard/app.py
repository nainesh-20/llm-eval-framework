"""
Streamlit Leaderboard — LLM Eval Framework Dashboard

Reads eval results from SQLite (live runs) with fallback to benchmark JSON (demo mode).
Deploy to Streamlit Community Cloud for a public URL to link in your README.

Run locally:
    streamlit run dashboard/app.py
"""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# ── Page config ───────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="LLM Eval Leaderboard",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded",
)

METRICS = ["faithfulness", "hallucination", "pii", "toxicity", "latency"]
METRIC_DESCRIPTIONS = {
    "faithfulness": "Answer grounded in context (RAGAS)",
    "hallucination": "No fabricated facts (G-Eval / LLM-judge)",
    "pii": "No PII leakage (regex + LLM-judge)",
    "toxicity": "Safe, non-toxic response (LLM-judge)",
    "latency": "Response speed score (penalizes >1s)",
}
METRIC_COLORS = {
    "faithfulness": "#2196F3",
    "hallucination": "#F44336",
    "pii": "#9C27B0",
    "toxicity": "#FF5722",
    "latency": "#4CAF50",
}
RED_TEAM_CATEGORIES = [
    "prompt_injection",
    "jailbreak",
    "pii_exfiltration",
    "prompt_leakage",
    "toxicity_induction",
]
MODEL_COLORS = ["#1565C0", "#B71C1C", "#1B5E20", "#E65100", "#4A148C"]


# ── Data loading ──────────────────────────────────────────────────────────────

@st.cache_data(ttl=60)
def load_from_sqlite(db_path: str) -> list[dict] | None:
    """Load runs from SQLite if available."""
    path = Path(db_path)
    if not path.exists():
        return None
    try:
        conn = sqlite3.connect(str(path))
        conn.row_factory = sqlite3.Row

        run_rows = conn.execute(
            "SELECT id, model_name, created_at FROM eval_runs ORDER BY created_at DESC"
        ).fetchall()

        if not run_rows:
            conn.close()
            return None

        runs = []
        for run_row in run_rows:
            run_id = run_row["id"]
            metrics_rows = conn.execute(
                """
                SELECT metric, AVG(score) as avg_score, COUNT(*) as total,
                       SUM(passed) as passed_count
                FROM eval_results WHERE run_id = ?
                GROUP BY metric
                """,
                (run_id,),
            ).fetchall()

            red_team_rows = conn.execute(
                """
                SELECT category, COUNT(*) as total, SUM(passed) as passed_count,
                       AVG(score) as avg_score
                FROM red_team_results WHERE run_id = ?
                GROUP BY category
                """,
                (run_id,),
            ).fetchall()

            metrics = {}
            for m in metrics_rows:
                metrics[m["metric"]] = {
                    "avg_score": m["avg_score"],
                    "pass_rate": m["passed_count"] / m["total"] if m["total"] > 0 else 0,
                    "total": m["total"],
                }

            red_team = {}
            for r in red_team_rows:
                red_team[r["category"]] = {
                    "pass_rate": r["passed_count"] / r["total"] if r["total"] > 0 else 0,
                    "avg_score": r["avg_score"],
                    "total": r["total"],
                }

            runs.append({
                "run_id": run_id,
                "model_name": run_row["model_name"],
                "display_name": run_row["model_name"],
                "created_at": run_row["created_at"],
                "metrics": metrics,
                "red_team": red_team,
            })

        conn.close()
        return runs
    except Exception as e:
        st.warning(f"Could not load from SQLite: {e}")
        return None


@st.cache_data(ttl=3600)
def load_demo_data() -> list[dict]:
    json_path = Path("results/benchmark_2026_03.json")
    if not json_path.exists():
        # Try relative to script location
        json_path = Path(__file__).parent.parent / "results" / "benchmark_2026_03.json"
    with open(json_path, encoding="utf-8") as f:
        data = json.load(f)
    return data["runs"]


def get_runs() -> tuple[list[dict], bool]:
    """Returns (runs, is_live_data). Falls back to demo JSON if SQLite unavailable."""
    db_path = Path("results/evals.db")
    if not db_path.exists():
        db_path = Path(__file__).parent.parent / "results" / "evals.db"

    live = load_from_sqlite(str(db_path))
    if live:
        return live, True
    return load_demo_data(), False


# ── Build DataFrames ───────────────────────────────────────────────────────────

def build_leaderboard_df(runs: list[dict]) -> pd.DataFrame:
    rows = []
    for run in runs:
        row = {"Model": run.get("display_name", run["model_name"])}
        total_score = 0.0
        count = 0
        for metric in METRICS:
            m = run["metrics"].get(metric, {})
            score = m.get("avg_score", 0.0)
            row[metric.capitalize()] = round(score, 2)
            total_score += score
            count += 1
        row["Overall"] = round(total_score / count if count > 0 else 0, 2)

        # Red-team overall pass rate
        rt = run.get("red_team", {})
        if rt:
            rt_pass_rates = [v["pass_rate"] for v in rt.values()]
            row["Red-Team"] = f"{sum(rt_pass_rates) / len(rt_pass_rates) * 100:.0f}%"
        else:
            row["Red-Team"] = "N/A"

        rows.append(row)

    df = pd.DataFrame(rows)
    df = df.sort_values("Overall", ascending=False).reset_index(drop=True)
    df.index += 1  # 1-based rank
    df.index.name = "Rank"
    return df


def build_metric_df(runs: list[dict]) -> pd.DataFrame:
    rows = []
    for run in runs:
        model = run.get("display_name", run["model_name"])
        for metric in METRICS:
            m = run["metrics"].get(metric, {})
            rows.append({
                "model": model,
                "metric": metric,
                "score": m.get("avg_score", 0.0),
                "pass_rate": m.get("pass_rate", 0.0) * 100,
            })
    return pd.DataFrame(rows)


def build_red_team_df(runs: list[dict]) -> pd.DataFrame:
    rows = []
    for run in runs:
        model = run.get("display_name", run["model_name"])
        for cat in RED_TEAM_CATEGORIES:
            rt = run.get("red_team", {}).get(cat, {})
            rows.append({
                "model": model,
                "category": cat.replace("_", " ").title(),
                "pass_rate": rt.get("pass_rate", 0.0) * 100,
                "avg_score": rt.get("avg_score", 0.0),
            })
    return pd.DataFrame(rows)


# ── Main app ──────────────────────────────────────────────────────────────────

def main() -> None:
    # ── Header ────────────────────────────────────────────────────────────────
    st.markdown(
        """
        <h1 style='margin-bottom:0'>🔬 LLM Eval Framework</h1>
        <p style='color:#aaa; font-size:1.1rem; margin-top:0.25rem'>
        Benchmarking LLMs across faithfulness, hallucination, PII safety, toxicity & latency
        </p>
        """,
        unsafe_allow_html=True,
    )
    st.markdown("---")

    runs, is_live = get_runs()

    # Surface the benchmark date in a subtle way
    if not is_live and runs:
        st.caption("📊 Showing pre-computed benchmark results from March 2026. Run `llm-eval run` locally to generate live data.")

    # ── Sidebar ───────────────────────────────────────────────────────────────
    with st.sidebar:
        st.markdown("### Filters")
        all_models = [r.get("display_name", r["model_name"]) for r in runs]
        selected_models = st.multiselect(
            "Models", options=all_models, default=all_models
        )

        st.markdown("---")
        st.markdown("### Data Source")
        if is_live:
            st.success("📡 Live SQLite data")
        else:
            st.info("📊 Demo data (benchmark_2026_03.json)")
            st.markdown("*Run `llm-eval run` to populate live data.*")

        st.markdown("---")
        st.markdown("### Resources")
        st.markdown("- [GitHub Repository](https://github.com/nainesh-20/llm-eval-framework)")
        st.markdown("- [Quick Start Docs](https://github.com/nainesh-20/llm-eval-framework#quick-start)")
        st.markdown("- [Architecture](https://github.com/nainesh-20/llm-eval-framework#architecture)")

    # Filter runs to selected models
    filtered_runs = [r for r in runs if r.get("display_name", r["model_name"]) in selected_models]
    if not filtered_runs:
        st.warning("No models selected. Choose at least one model in the sidebar.")
        return

    # ── Leaderboard Table ─────────────────────────────────────────────────────
    st.markdown("## 🏆 Leaderboard")
    lb_df = build_leaderboard_df(filtered_runs)

    # Highlight top model
    def color_score(val):
        if isinstance(val, float):
            if val >= 8.0:
                return "background-color: #1b5e2022; color: #1b5e20; font-weight: bold"
            elif val >= 6.0:
                return "background-color: #f57f1722; color: #e65100"
            else:
                return "background-color: #b71c1c22; color: #b71c1c"
        return ""

    styled = lb_df.style.map(
        color_score,
        subset=[c.capitalize() for c in METRICS] + ["Overall"],
    ).format(
        {c.capitalize(): "{:.2f}" for c in METRICS} | {"Overall": "{:.2f}"}
    )
    st.dataframe(styled, use_container_width=True, height=200)

    # ── Metric Comparison Charts ───────────────────────────────────────────────
    st.markdown("## 📊 Metric Comparison")
    metric_df = build_metric_df(filtered_runs)

    tab1, tab2 = st.tabs(["Score (0–10)", "Pass Rate (%)"])

    with tab1:
        fig = px.bar(
            metric_df,
            x="metric",
            y="score",
            color="model",
            barmode="group",
            title="Average Score per Evaluator (higher = better)",
            labels={"score": "Avg Score (0–10)", "metric": "Evaluator", "model": "Model"},
            color_discrete_sequence=MODEL_COLORS,
        )
        fig.add_hline(y=7, line_dash="dash", line_color="gray", annotation_text="Pass threshold")
        fig.update_layout(
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)",
            yaxis_range=[0, 10.5],
            legend=dict(orientation="h", yanchor="bottom", y=1.02),
        )
        st.plotly_chart(fig, use_container_width=True)

    with tab2:
        fig2 = px.bar(
            metric_df,
            x="metric",
            y="pass_rate",
            color="model",
            barmode="group",
            title="Pass Rate per Evaluator (% of samples that passed threshold)",
            labels={"pass_rate": "Pass Rate (%)", "metric": "Evaluator", "model": "Model"},
            color_discrete_sequence=MODEL_COLORS,
        )
        fig2.add_hline(y=80, line_dash="dash", line_color="gray", annotation_text="80% target")
        fig2.update_layout(
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)",
            yaxis_range=[0, 110],
            legend=dict(orientation="h", yanchor="bottom", y=1.02),
        )
        st.plotly_chart(fig2, use_container_width=True)

    # ── Radar Chart ───────────────────────────────────────────────────────────
    st.markdown("## 🕸 Model Capability Radar")
    fig_radar = go.Figure()
    categories = [m.capitalize() for m in METRICS]

    for i, run in enumerate(filtered_runs):
        scores = []
        for m in METRICS:
            metric_data = run["metrics"].get(m, {})
            scores.append(metric_data.get("avg_score", 0.0))

        model_name = run.get("display_name", run["model_name"])
        fig_radar.add_trace(go.Scatterpolar(
            r=scores + [scores[0]],
            theta=categories + [categories[0]],
            fill="toself",
            name=model_name,
            line_color=MODEL_COLORS[i % len(MODEL_COLORS)],
            opacity=0.6,
        ))

    fig_radar.update_layout(
        polar=dict(
            radialaxis=dict(visible=True, range=[0, 10], tickvals=[2, 4, 6, 8, 10]),
        ),
        showlegend=True,
        title="Model Capability Radar (all metrics 0–10)",
        height=450,
    )
    st.plotly_chart(fig_radar, use_container_width=True)

    # ── Red-Team Results ───────────────────────────────────────────────────────
    st.markdown("## ⚔️ Red-Team Robustness")
    rt_df = build_red_team_df(filtered_runs)

    if rt_df.empty or rt_df["pass_rate"].isna().all():
        st.info("No red-team results available. Run `llm-eval red-team` to generate.")
    else:
        fig_rt = px.bar(
            rt_df,
            x="category",
            y="pass_rate",
            color="model",
            barmode="group",
            title="Red-Team Pass Rate by Category (higher = more robust)",
            labels={"pass_rate": "Pass Rate (%)", "category": "Attack Category", "model": "Model"},
            color_discrete_sequence=MODEL_COLORS,
        )
        fig_rt.add_hline(y=80, line_dash="dash", line_color="gray", annotation_text="Robust threshold")
        fig_rt.update_layout(
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)",
            yaxis_range=[0, 110],
            xaxis_tickangle=-15,
            legend=dict(orientation="h", yanchor="bottom", y=1.02),
        )
        st.plotly_chart(fig_rt, use_container_width=True)

    # ── Latency Details ────────────────────────────────────────────────────────
    st.markdown("## ⚡ Latency Details")
    cols = st.columns(len(filtered_runs))
    for col, run in zip(cols, filtered_runs):
        lat = run["metrics"].get("latency", {})
        with col:
            model_name = run.get("display_name", run["model_name"])
            st.markdown(f"**{model_name}**")
            if "p50_ms" in lat:
                st.metric("p50", f"{lat['p50_ms']:.0f}ms")
                st.metric("p95", f"{lat['p95_ms']:.0f}ms")
                st.metric("p99", f"{lat.get('p99_ms', 0):.0f}ms")
            else:
                st.metric("Avg Score", f"{lat.get('avg_score', 0):.2f}/10")

    # ── Evaluator Descriptions ─────────────────────────────────────────────────
    with st.expander("📖 About the Evaluators"):
        for metric, desc in METRIC_DESCRIPTIONS.items():
            st.markdown(f"**{metric.capitalize()}** — {desc}")

    st.markdown("---")
    st.caption(
        "Built with [llm-eval-framework](https://github.com/nainesh-20/llm-eval-framework) • "
        "Powered by RAGAS, DeepEval, OpenAI, Anthropic"
    )


if __name__ == "__main__":
    main()
