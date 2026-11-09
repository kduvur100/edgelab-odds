"""
edgelab_odds — Streamlit Dashboard
===================================
Run with:
    streamlit run app.py

Requires the pipeline to have been run at least once:
    python -m edgelab_odds.ingest.loader
    python -m edgelab_odds.features.build
    python -m edgelab_odds.model.train
    python -m edgelab_odds.model.predict --all
    python -m edgelab_odds.backtest.simulate
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="EdgeLab Odds",
    page_icon="🥊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(level="INFO")
log = logging.getLogger(__name__)

# ── Lazy imports (avoid import errors if DB not seeded yet) ───────────────────
@st.cache_resource
def _get_db_conn():
    from edgelab_odds.db import get_conn, init_db
    conn = get_conn(read_only=True)
    init_db(conn)
    return conn


def _safe_query(sql: str, params: list | None = None) -> pd.DataFrame:
    try:
        conn = _get_db_conn()
        return conn.execute(sql, params or []).df()
    except Exception as exc:
        st.error(f"Query failed: {exc}")
        return pd.DataFrame()


# ── Sidebar ───────────────────────────────────────────────────────────────────
st.sidebar.title("🥊 EdgeLab Odds")
st.sidebar.markdown("UFC fight prediction engine")

section = st.sidebar.radio(
    "Navigate",
    ["📋 Upcoming Card", "📊 Model Performance", "💰 Backtest ROI", "📈 CLV Analysis", "🗄️ Fight History"],
)

# ── Helper renderers ──────────────────────────────────────────────────────────

def _prob_bar(prob: float) -> str:
    filled = round(prob * 20)
    return "█" * filled + "░" * (20 - filled)


def _format_prob(p: float) -> str:
    return f"{p*100:.1f}%"


def _american_odds(no_vig_prob: float) -> str:
    if pd.isna(no_vig_prob) or no_vig_prob <= 0:
        return "—"
    if no_vig_prob >= 0.5:
        return f"-{round(no_vig_prob / (1 - no_vig_prob) * 100)}"
    return f"+{round((1 - no_vig_prob) / no_vig_prob * 100)}"


# ══════════════════════════════════════════════════════════════════════════════
# 1. UPCOMING CARD
# ══════════════════════════════════════════════════════════════════════════════
if section == "📋 Upcoming Card":
    st.title("📋 Upcoming Card Predictions")

    preds = _safe_query("""
        SELECT
            p.fight_id,
            f.red_fighter,
            f.blue_fighter,
            f.weight_class,
            f.title_bout,
            f.event_date,
            p.prob_red_wins,
            p.prob_blue_wins,
            p.predicted_label,
            feat.red_imp_prob_novigml  AS market_red,
            feat.blue_imp_prob_novigml AS market_blue,
            p.clv_red
        FROM predictions p
        JOIN fights  f    USING (fight_id)
        JOIN features feat USING (fight_id)
        WHERE f.label = -1 OR f.label IS NULL
        ORDER BY f.event_date ASC, f.title_bout DESC
    """)

    if preds.empty:
        st.info(
            "No upcoming fight predictions found. Run the refresh pipeline first:\n\n"
            "```bash\n"
            "python -m edgelab_odds.ingest.refresh\n"
            "python -m edgelab_odds.model.predict\n"
            "```"
        )
    else:
        event_dates = sorted(preds["event_date"].dropna().unique())
        selected_date = st.selectbox("Event date", event_dates) if len(event_dates) > 1 else event_dates[0]
        card = preds[preds["event_date"] == selected_date] if selected_date else preds

        st.markdown(f"### {selected_date}  —  {len(card)} bouts")

        for _, row in card.iterrows():
            title_badge = "🏆 " if row.get("title_bout") else ""
            with st.expander(
                f"{title_badge}{row['red_fighter']} vs {row['blue_fighter']}  —  {row['weight_class']}",
                expanded=True,
            ):
                col1, col2, col3 = st.columns([3, 1, 3])
                with col1:
                    st.markdown(f"**🔴 {row['red_fighter']}**")
                    prob_r = float(row["prob_red_wins"])
                    st.markdown(f"`{_prob_bar(prob_r)}` **{_format_prob(prob_r)}**")
                    mkt_r = row.get("market_red")
                    if not pd.isna(mkt_r):
                        st.caption(f"Market: {_format_prob(mkt_r)}  ({_american_odds(mkt_r)})")
                with col2:
                    winner = row["red_fighter"] if row["predicted_label"] == 1 else row["blue_fighter"]
                    st.markdown("<div style='text-align:center; padding-top:20px;'>⚔️<br><small>Pick:</small><br>"
                                f"<b>{winner}</b></div>", unsafe_allow_html=True)
                with col3:
                    st.markdown(f"**🔵 {row['blue_fighter']}**")
                    prob_b = float(row["prob_blue_wins"])
                    st.markdown(f"`{_prob_bar(prob_b)}` **{_format_prob(prob_b)}**")
                    mkt_b = row.get("market_blue")
                    if not pd.isna(mkt_b):
                        st.caption(f"Market: {_format_prob(mkt_b)}  ({_american_odds(mkt_b)})")

                if not pd.isna(row.get("clv_red")):
                    clv = float(row["clv_red"])
                    color = "green" if clv > 0 else "red"
                    st.caption(f"CLV (Red): :{color}[{clv:+.3f}]")


# ══════════════════════════════════════════════════════════════════════════════
# 2. MODEL PERFORMANCE
# ══════════════════════════════════════════════════════════════════════════════
elif section == "📊 Model Performance":
    st.title("📊 Model Performance")

    runs = _safe_query("""
        SELECT run_id, model_type, trained_at, n_samples, n_features,
               cv_accuracy, cv_roc_auc, cv_log_loss
        FROM model_runs
        ORDER BY trained_at DESC
        LIMIT 20
    """)

    if runs.empty:
        st.info("No trained models yet. Run `python -m edgelab_odds.model.train` first.")
    else:
        st.subheader("Training runs")
        st.dataframe(
            runs.style.format({
                "cv_accuracy": "{:.4f}", "cv_roc_auc": "{:.4f}", "cv_log_loss": "{:.4f}"
            }),
            use_container_width=True,
        )

        # Accuracy over time
        preds_hist = _safe_query("""
            SELECT
                date_trunc('month', f.event_date) AS month,
                avg((p.predicted_label = p.actual_label)::int) AS accuracy,
                count(*) AS n
            FROM predictions p
            JOIN fights f USING (fight_id)
            WHERE p.actual_label IS NOT NULL AND p.actual_label != -1
            GROUP BY 1 ORDER BY 1
        """)
        if not preds_hist.empty:
            st.subheader("Monthly prediction accuracy")
            st.line_chart(preds_hist.set_index("month")["accuracy"], height=300)

        # Calibration: predicted prob vs actual win rate
        cal = _safe_query("""
            SELECT
                round(prob_red_wins::double / 0.1) * 0.1 AS bucket,
                avg((actual_label = 1)::int) AS actual_win_rate,
                count(*) AS n
            FROM predictions
            WHERE actual_label IS NOT NULL AND actual_label != -1
            GROUP BY 1
            HAVING count(*) >= 10
            ORDER BY 1
        """)
        if not cal.empty:
            st.subheader("Calibration (predicted prob vs actual win rate)")
            cal_chart = cal.set_index("bucket")[["actual_win_rate"]]
            st.line_chart(cal_chart, height=300)
            st.caption("A well-calibrated model follows the diagonal (predicted ≈ actual).")


# ══════════════════════════════════════════════════════════════════════════════
# 3. BACKTEST ROI
# ══════════════════════════════════════════════════════════════════════════════
elif section == "💰 Backtest ROI":
    st.title("💰 Backtest ROI")

    bt = _safe_query("""
        SELECT bt_id, run_id, strategy, computed_at,
               n_bets, n_wins, win_rate, total_staked, total_return,
               roi, avg_clv, max_drawdown
        FROM backtest_results
        ORDER BY computed_at DESC
        LIMIT 30
    """)

    if bt.empty:
        st.info(
            "No backtest results yet. Run:\n\n"
            "```bash\n"
            "python -m edgelab_odds.model.predict --all\n"
            "python -m edgelab_odds.backtest.simulate\n"
            "```"
        )
    else:
        latest_run = bt["run_id"].iloc[0]
        latest = bt[bt["run_id"] == latest_run]

        col1, col2, col3 = st.columns(3)
        for col, (_, row) in zip([col1, col2, col3], latest.iterrows()):
            roi_pct = row["roi"] * 100
            color = "normal" if roi_pct >= 0 else "inverse"
            col.metric(
                label=row["strategy"].replace("_", " ").title(),
                value=f"{roi_pct:+.2f}% ROI",
                delta=f"Win rate: {row['win_rate']*100:.1f}%",
            )

        st.subheader("All backtest runs")
        st.dataframe(
            bt.style.format({
                "win_rate": "{:.3f}", "roi": "{:+.4f}",
                "max_drawdown": "{:.4f}", "avg_clv": "{:+.5f}",
            }),
            use_container_width=True,
        )


# ══════════════════════════════════════════════════════════════════════════════
# 4. CLV ANALYSIS
# ══════════════════════════════════════════════════════════════════════════════
elif section == "📈 CLV Analysis":
    st.title("📈 Closing-Line Value Analysis")

    clv_df = _safe_query("""
        SELECT
            p.clv_red,
            p.prob_red_wins,
            feat.red_imp_prob_novigml AS market_prob,
            f.weight_class,
            f.event_date,
            (p.predicted_label = p.actual_label)::int AS correct
        FROM predictions p
        JOIN features feat USING (fight_id)
        JOIN fights   f    USING (fight_id)
        WHERE p.actual_label IS NOT NULL AND p.actual_label != -1
              AND p.clv_red IS NOT NULL
        ORDER BY f.event_date ASC
    """)

    if clv_df.empty:
        st.info("No CLV data yet — run `python -m edgelab_odds.model.predict --all` first.")
    else:
        from scipy import stats as scipy_stats
        clv = clv_df["clv_red"].astype(float).dropna()
        t, p = scipy_stats.ttest_1samp(clv, 0.0)

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Mean CLV", f"{clv.mean():+.4f}")
        c2.metric("% CLV > 0", f"{(clv > 0).mean()*100:.1f}%")
        c3.metric("t-stat", f"{t:.2f}")
        c4.metric("p-value", f"{p:.4f}", delta="✓ sig" if p < 0.05 else "✗ not sig")

        st.subheader("CLV distribution")
        st.bar_chart(
            clv.value_counts(bins=30, sort=False).sort_index(),
            height=300,
        )

        st.subheader("Cumulative CLV over time")
        cum_clv = clv_df.set_index("event_date")["clv_red"].astype(float).cumsum()
        st.line_chart(cum_clv, height=300)

        st.subheader("Accuracy by CLV bucket")
        clv_df["clv_bucket"] = pd.cut(
            clv_df["clv_red"],
            bins=[-1, -0.05, 0.0, 0.05, 1.0],
            labels=["<-5%", "-5–0%", "0–5%", ">5%"],
        )
        bucket_acc = (
            clv_df.groupby("clv_bucket", observed=True)["correct"]
            .agg(["mean", "count"])
            .rename(columns={"mean": "accuracy", "count": "n"})
        )
        st.dataframe(bucket_acc.style.format({"accuracy": "{:.3f}"}))


# ══════════════════════════════════════════════════════════════════════════════
# 5. FIGHT HISTORY
# ══════════════════════════════════════════════════════════════════════════════
elif section == "🗄️ Fight History":
    st.title("🗄️ Fight History")

    col_a, col_b, col_c = st.columns([2, 2, 1])
    fighter_query = col_a.text_input("Fighter name contains…", "")
    wc_options = ["All"] + list(
        _safe_query("SELECT DISTINCT weight_class FROM fights ORDER BY 1")
        .get("weight_class", pd.Series()).tolist()
    )
    weight_class = col_b.selectbox("Weight class", wc_options)
    limit = col_c.number_input("Max rows", 50, 500, 100)

    where_clauses = ["label IN (0, 1)"]
    params: list = []
    if fighter_query:
        where_clauses.append(
            "(LOWER(red_fighter) LIKE ? OR LOWER(blue_fighter) LIKE ?)"
        )
        params += [f"%{fighter_query.lower()}%", f"%{fighter_query.lower()}%"]
    if weight_class != "All":
        where_clauses.append("weight_class = ?")
        params.append(weight_class)

    where_sql = " AND ".join(where_clauses)
    fights = _safe_query(
        f"""
        SELECT event_date, red_fighter, blue_fighter, weight_class,
               winner, finish, title_bout, red_odds, blue_odds
        FROM fights
        WHERE {where_sql}
        ORDER BY event_date DESC
        LIMIT {int(limit)}
        """,
        params,
    )

    st.markdown(f"**{len(fights)} fights**")
    st.dataframe(fights, use_container_width=True)
