-- edgelab-odds database schema
-- Engine: DuckDB  (CREATE TABLE IF NOT EXISTS syntax)
-- Run via: edgelab_odds.db.init_db()

-- ── Raw fights ────────────────────────────────────────────────────────────────
-- One row per historical UFC fight, loaded from ufc-master.csv.
-- Column names mirror the CSV so the ingest step is a direct bulk-insert.
CREATE TABLE IF NOT EXISTS fights (
    -- Primary key
    fight_id        VARCHAR PRIMARY KEY,   -- generated: "{Date}_{RedFighter}_{BlueFighter}"

    -- Event metadata
    event_date      DATE        NOT NULL,
    location        VARCHAR,
    country         VARCHAR,
    weight_class    VARCHAR     NOT NULL,
    gender          VARCHAR     NOT NULL   DEFAULT 'MALE',
    number_of_rounds INTEGER    NOT NULL,
    title_bout      BOOLEAN     NOT NULL   DEFAULT FALSE,
    empty_arena     BOOLEAN     NOT NULL   DEFAULT FALSE,

    -- Fighters
    red_fighter     VARCHAR     NOT NULL,
    blue_fighter    VARCHAR     NOT NULL,
    winner          VARCHAR     NOT NULL,  -- 'Red' | 'Blue' | 'No Contest'
    label           TINYINT     NOT NULL,  -- 1 = Red wins, 0 = Blue wins

    -- Opening / closing moneyline odds (American format)
    red_odds        DOUBLE,
    blue_odds       DOUBLE,

    -- Method-specific closing odds
    red_dec_odds    DOUBLE,
    blue_dec_odds   DOUBLE,
    r_sub_odds      DOUBLE,
    b_sub_odds      DOUBLE,
    r_ko_odds       DOUBLE,
    b_ko_odds       DOUBLE,

    -- Implied probabilities (derived from odds, vig-adjusted)
    red_imp_prob    DOUBLE,
    blue_imp_prob   DOUBLE,

    -- Outcome
    finish          VARCHAR,       -- 'KO/TKO' | 'SUB' | 'U-DEC' | 'S-DEC' | 'M-DEC' | 'NC'
    finish_details  VARCHAR,
    finish_round    SMALLINT,
    finish_round_time VARCHAR,
    total_fight_time_secs INTEGER,

    -- Fighter stats — Red corner
    red_current_lose_streak  SMALLINT,
    red_current_win_streak   SMALLINT,
    red_draws                SMALLINT,
    red_avg_sig_str_landed   DOUBLE,
    red_avg_sig_str_pct      DOUBLE,
    red_avg_sub_att          DOUBLE,
    red_avg_td_landed        DOUBLE,
    red_avg_td_pct           DOUBLE,
    red_longest_win_streak   SMALLINT,
    red_losses               SMALLINT,
    red_total_rounds_fought  SMALLINT,
    red_total_title_bouts    SMALLINT,
    red_wins_dec_majority    SMALLINT,
    red_wins_dec_split       SMALLINT,
    red_wins_dec_unanimous   SMALLINT,
    red_wins_ko              SMALLINT,
    red_wins_sub             SMALLINT,
    red_wins_tko_doc         SMALLINT,
    red_wins                 SMALLINT,
    red_stance               VARCHAR,
    red_height_cms           DOUBLE,
    red_reach_cms            DOUBLE,
    red_weight_lbs           DOUBLE,
    red_age                  DOUBLE,

    -- Fighter stats — Blue corner
    blue_current_lose_streak SMALLINT,
    blue_current_win_streak  SMALLINT,
    blue_draws               SMALLINT,
    blue_avg_sig_str_landed  DOUBLE,
    blue_avg_sig_str_pct     DOUBLE,
    blue_avg_sub_att         DOUBLE,
    blue_avg_td_landed       DOUBLE,
    blue_avg_td_pct          DOUBLE,
    blue_longest_win_streak  SMALLINT,
    blue_losses              SMALLINT,
    blue_total_rounds_fought SMALLINT,
    blue_total_title_bouts   SMALLINT,
    blue_wins_dec_majority   SMALLINT,
    blue_wins_dec_split      SMALLINT,
    blue_wins_dec_unanimous  SMALLINT,
    blue_wins_ko             SMALLINT,
    blue_wins_sub            SMALLINT,
    blue_wins_tko_doc        SMALLINT,
    blue_wins                SMALLINT,
    blue_stance              VARCHAR,
    blue_height_cms          DOUBLE,
    blue_reach_cms           DOUBLE,
    blue_weight_lbs          DOUBLE,
    blue_age                 DOUBLE,

    -- Pre-computed differentials (Red minus Blue)
    lose_streak_dif          SMALLINT,
    win_streak_dif           SMALLINT,
    longest_win_streak_dif   SMALLINT,
    win_dif                  SMALLINT,
    loss_dif                 SMALLINT,
    total_round_dif          SMALLINT,
    total_title_bout_dif     SMALLINT,
    ko_dif                   SMALLINT,
    sub_dif                  SMALLINT,
    height_dif               DOUBLE,
    reach_dif                DOUBLE,
    age_dif                  DOUBLE,
    sig_str_dif              DOUBLE,
    avg_sub_att_dif          DOUBLE,
    avg_td_dif               DOUBLE,

    -- Rankings
    r_match_wc_rank          SMALLINT,
    b_match_wc_rank          SMALLINT,
    better_rank              VARCHAR,

    -- Audit
    ingested_at              TIMESTAMPTZ NOT NULL DEFAULT now()
);

-- ── Engineered features ───────────────────────────────────────────────────────
-- One row per fight; built by features/build.py from the fights table.
-- The model layer reads exclusively from this table.
CREATE TABLE IF NOT EXISTS features (
    fight_id        VARCHAR PRIMARY KEY REFERENCES fights(fight_id),

    -- Odds-derived features
    red_imp_prob_novigml    DOUBLE,  -- no-vig implied prob from moneyline
    blue_imp_prob_novigml   DOUBLE,
    odds_edge               DOUBLE,  -- red_imp_prob - blue_imp_prob (market lean)
    log_odds_ratio          DOUBLE,  -- log(red_imp_prob / blue_imp_prob)

    -- Physical differentials (normalised to cm/years)
    height_dif_cm           DOUBLE,
    reach_dif_cm            DOUBLE,
    age_dif_yrs             DOUBLE,

    -- Record differentials
    win_rate_dif            DOUBLE,  -- (red_wins/red_total) - (blue_wins/blue_total)
    experience_dif          INTEGER, -- total fights red - total fights blue
    win_streak_dif          SMALLINT,
    lose_streak_dif         SMALLINT,

    -- Striking differentials
    sig_str_landed_dif      DOUBLE,
    sig_str_pct_dif         DOUBLE,
    sig_str_output_score_dif DOUBLE,  -- (red_landed×red_pct) − (blue_landed×blue_pct); effective strike output differential

    -- Grappling differentials
    td_landed_dif           DOUBLE,
    td_pct_dif              DOUBLE,
    sub_att_dif             DOUBLE,

    -- Win-method rates (red)
    red_ko_rate             DOUBLE,
    red_sub_rate            DOUBLE,
    red_dec_rate            DOUBLE,

    -- Win-method rates (blue)
    blue_ko_rate            DOUBLE,
    blue_sub_rate           DOUBLE,
    blue_dec_rate           DOUBLE,

    -- Stance encoding  (0=Orthodox 1=Southpaw 2=Switch)
    red_stance_enc          SMALLINT,
    blue_stance_enc         SMALLINT,
    ortho_vs_southpaw       BOOLEAN,

    -- Ranking features
    red_ranked              BOOLEAN,
    blue_ranked             BOOLEAN,
    rank_dif                SMALLINT, -- lower is better; red_rank - blue_rank (NULL if either unranked)

    -- Context flags
    title_bout              BOOLEAN,
    is_5_round              BOOLEAN,
    empty_arena             BOOLEAN,

    -- Target
    label                   TINYINT NOT NULL,  -- 1 = Red wins

    built_at                TIMESTAMPTZ NOT NULL DEFAULT now()
);

-- ── Model runs ────────────────────────────────────────────────────────────────
-- Tracks each training run: hyperparams, CV scores, artifact path.
CREATE TABLE IF NOT EXISTS model_runs (
    run_id          VARCHAR PRIMARY KEY,   -- uuid4
    model_type      VARCHAR NOT NULL,      -- 'lr' | 'rf' | 'xgb' | 'ensemble'
    trained_at      TIMESTAMPTZ NOT NULL DEFAULT now(),
    n_samples       INTEGER,
    n_features      INTEGER,
    cv_accuracy     DOUBLE,
    cv_roc_auc      DOUBLE,
    cv_log_loss     DOUBLE,
    artifact_path   VARCHAR,               -- path to .pkl file
    params          JSON                   -- hyperparameter dict
);

-- ── Predictions ───────────────────────────────────────────────────────────────
-- Stores out-of-sample predictions for backtesting and CLV analysis.
CREATE TABLE IF NOT EXISTS predictions (
    pred_id         VARCHAR PRIMARY KEY,   -- uuid4
    run_id          VARCHAR REFERENCES model_runs(run_id),
    fight_id        VARCHAR REFERENCES fights(fight_id),
    predicted_at    TIMESTAMPTZ NOT NULL DEFAULT now(),

    prob_red_wins   DOUBLE NOT NULL,
    prob_blue_wins  DOUBLE NOT NULL,
    predicted_label TINYINT NOT NULL,      -- 1 = Red, 0 = Blue
    actual_label    TINYINT,               -- filled in post-fight

    -- Closing-line value: our prob vs market implied prob
    clv_red         DOUBLE,  -- prob_red_wins - red_imp_prob_novigml
    correct         BOOLEAN  -- predicted_label = actual_label
);

-- ── Backtest results ──────────────────────────────────────────────────────────
-- Aggregated ROI and CLV metrics per model run / betting strategy.
CREATE TABLE IF NOT EXISTS backtest_results (
    bt_id           VARCHAR PRIMARY KEY,
    run_id          VARCHAR REFERENCES model_runs(run_id),
    strategy        VARCHAR NOT NULL,   -- 'flat' | 'kelly' | 'fractional_kelly'
    computed_at     TIMESTAMPTZ NOT NULL DEFAULT now(),

    n_bets          INTEGER,
    n_wins          INTEGER,
    win_rate        DOUBLE,
    total_staked    DOUBLE,
    total_return    DOUBLE,
    roi             DOUBLE,             -- (total_return - total_staked) / total_staked
    avg_clv         DOUBLE,
    max_drawdown    DOUBLE
);

-- ── Indexes ───────────────────────────────────────────────────────────────────
CREATE INDEX IF NOT EXISTS idx_fights_date        ON fights (event_date);
CREATE INDEX IF NOT EXISTS idx_fights_weight      ON fights (weight_class);
CREATE INDEX IF NOT EXISTS idx_fights_red         ON fights (red_fighter);
CREATE INDEX IF NOT EXISTS idx_fights_blue        ON fights (blue_fighter);
CREATE INDEX IF NOT EXISTS idx_predictions_run    ON predictions (run_id);
CREATE INDEX IF NOT EXISTS idx_backtest_run       ON backtest_results (run_id);
