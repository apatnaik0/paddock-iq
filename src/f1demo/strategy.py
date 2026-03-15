from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import fastf1
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

from .analysis import clean_session_laps
from .data import init_fastf1_cache, laps_dataframe, load_sessions


@dataclass
class StrategyOverview:
    strategy_rows: list[dict]
    stint_pace_rows: list[dict]
    team_outlook_rows: list[dict]
    undercut_rows: list[dict]
    stint_extension_rows: list[dict]
    race_mode_rows: list[dict]
    pit_window_reference: list[dict]
    hybrid_meta: dict[str, Any]
    total_laps: int


def _format_laptime(seconds: float | int | None) -> str:
    if seconds is None or (isinstance(seconds, float) and pd.isna(seconds)):
        return ""
    total = float(seconds)
    mins = int(total // 60)
    secs = total - mins * 60
    return f"{mins}:{secs:06.3f}"


def _normalize_team_name(team: str) -> str:
    t = (team or "").strip()
    key = t.lower()
    alias = {
        "kick sauber": "Audi",
        "sauber": "Audi",
        "alfa romeo": "Audi",
        "alfa romeo racing": "Audi",
        "rb": "Racing Bulls",
        "alphatauri": "Racing Bulls",
        "scuderia alphatauri": "Racing Bulls",
        "toro rosso": "Racing Bulls",
        "renault": "Alpine",
        "racing point": "Aston Martin",
        "force india": "Aston Martin",
    }
    return alias.get(key, t)


def _add_tyre_age(df: pd.DataFrame) -> pd.DataFrame:
    out = df.sort_values(["session", "Driver", "Stint", "LapNumber"]).copy()
    out["tyre_age_lap"] = out.groupby(["session", "Driver", "Stint"]).cumcount() + 1
    return out


def _scheduled_race_laps(bundle: object) -> int | None:
    event_name = str(getattr(bundle, "event_name", "") or "").strip().lower()
    if not event_name:
        return None
    lap_map = {
        "australian grand prix": 58,
        "chinese grand prix": 56,
    }
    return lap_map.get(event_name)


def _race_lap_count(bundle: object) -> int:
    race = bundle.sessions.get("Race")
    fallback = _scheduled_race_laps(bundle) or 58
    if race is None:
        return fallback
    race_laps = laps_dataframe(race, "Race")
    if race_laps.empty or "LapNumber" not in race_laps.columns:
        return fallback
    return int(pd.to_numeric(race_laps["LapNumber"], errors="coerce").max() or fallback)


def _practice_pool(bundle: object) -> pd.DataFrame:
    parts: list[pd.DataFrame] = []
    for s in ["FP2", "FP3", "FP1"]:
        sess = bundle.sessions.get(s)
        if sess is None:
            continue
        laps = laps_dataframe(sess, s)
        if laps.empty:
            continue
        parts.append(clean_session_laps(laps, s))
    if not parts:
        return pd.DataFrame()
    pool = pd.concat(parts, ignore_index=True)
    pool["Compound"] = pool["Compound"].fillna("UNKNOWN").astype(str).str.upper()
    return pool


def _race_sim_pool(pool: pd.DataFrame) -> pd.DataFrame:
    if pool.empty:
        return pd.DataFrame()
    work = _add_tyre_age(pool)
    work = work[work["Compound"].isin(["SOFT", "MEDIUM", "HARD"])].copy()
    stint_lens = (
        work.groupby(["session", "Driver", "Stint"], as_index=False)
        .agg(stint_laps=("lap_seconds", "count"))
    )
    long_stints = stint_lens[stint_lens["stint_laps"] >= 3]
    sim = work.merge(long_stints[["session", "Driver", "Stint"]], on=["session", "Driver", "Stint"], how="inner")
    # Drop first lap in each stint as warmup/outlap artifact.
    sim = sim[sim["tyre_age_lap"] >= 2]
    # Prefer FP2/FP3 race-sim stints; fallback to all practice sessions if sparse.
    pref = sim[sim["session"].isin(["FP2", "FP3"])]
    return pref if len(pref) >= 80 else sim


def _team_compound_slopes(pool: pd.DataFrame) -> pd.DataFrame:
    if pool.empty:
        return pd.DataFrame(columns=["Team", "Compound", "deg_s_per_lap"])
    work = _add_tyre_age(pool)
    rows: list[dict] = []
    for (team, drv, comp, sess, stint), g in work.groupby(["Team", "Driver", "Compound", "session", "Stint"]):
        g = g.dropna(subset=["lap_seconds", "tyre_age_lap"])
        if len(g) < 5 or g["tyre_age_lap"].nunique() < 4:
            continue
        x = g["tyre_age_lap"].astype(float).values
        y = g["lap_seconds"].astype(float).values
        slope, _ = np.polyfit(x, y, 1)
        rows.append({"Team": team, "Compound": comp, "deg_s_per_lap": float(slope)})
    if not rows:
        return pd.DataFrame(columns=["Team", "Compound", "deg_s_per_lap"])
    out = pd.DataFrame(rows)
    out["deg_s_per_lap"] = out["deg_s_per_lap"].clip(lower=0.02, upper=0.16)
    return out.groupby(["Team", "Compound"], as_index=False)["deg_s_per_lap"].median()


def _team_features_from_practice(bundle: object) -> pd.DataFrame:
    pool = _practice_pool(bundle)
    if pool.empty:
        return pd.DataFrame()
    race_sim = _race_sim_pool(pool)
    pace_src = race_sim if not race_sim.empty else pool
    pace = (
        pace_src.groupby(["Team", "Compound"], as_index=False)
        .agg(median_pace_s=("lap_seconds", "median"), laps=("lap_seconds", "count"))
    )
    pace = pace[pace["Compound"].isin(["SOFT", "MEDIUM", "HARD"])]
    if pace.empty:
        return pd.DataFrame()
    slopes = _team_compound_slopes(race_sim if not race_sim.empty else pool)
    default_slopes = {"SOFT": 0.090, "MEDIUM": 0.070, "HARD": 0.055}
    global_comp_pace = (
        pace.groupby("Compound", as_index=False)["median_pace_s"]
        .median()
        .set_index("Compound")["median_pace_s"]
        .to_dict()
    )
    total_laps = _race_lap_count(bundle)
    rows: list[dict] = []
    for team in sorted(pace["Team"].dropna().astype(str).unique().tolist()):
        tpace = pace[pace["Team"] == team].copy()
        if tpace.empty:
            continue
        comp_pace = {r["Compound"]: float(r["median_pace_s"]) for _, r in tpace.iterrows()}
        for c in ["SOFT", "MEDIUM", "HARD"]:
            comp_pace.setdefault(c, float(global_comp_pace.get(c, tpace["median_pace_s"].median())))
        tsl = slopes[slopes["Team"] == team]
        slope_map = {r["Compound"]: float(r["deg_s_per_lap"]) for _, r in tsl.iterrows()}
        deg = float(np.nanmedian([slope_map.get(c, default_slopes[c]) for c in ["SOFT", "MEDIUM", "HARD"]]))
        base = float(min(comp_pace.values()))
        rows.append(
            {
                "Team": team,
                "feat_deg": round(deg, 5),
                "feat_total_laps": int(total_laps),
                "feat_best_pace": round(base, 4),
                "feat_soft_gap": round(comp_pace["SOFT"] - base, 4),
                "feat_medium_gap": round(comp_pace["MEDIUM"] - base, 4),
                "feat_hard_gap": round(comp_pace["HARD"] - base, 4),
            }
        )
    return pd.DataFrame(rows)


def _team_labels_from_race(bundle: object) -> pd.DataFrame:
    race = bundle.sessions.get("Race")
    if race is None:
        return pd.DataFrame()
    laps = laps_dataframe(race, "Race")
    if laps.empty or "Stint" not in laps.columns:
        return pd.DataFrame()
    df = laps.copy()
    df["Stint"] = pd.to_numeric(df["Stint"], errors="coerce")
    df["LapNumber"] = pd.to_numeric(df["LapNumber"], errors="coerce")
    df = df.dropna(subset=["Driver", "Team", "LapNumber", "Stint"])
    if df.empty:
        return pd.DataFrame()
    per_driver = (
        df.groupby(["Team", "Driver"], as_index=False)
        .agg(stints=("Stint", "nunique"))
    )
    per_driver["target_stops"] = (per_driver["stints"] - 1).clip(lower=0, upper=4)
    pit1 = (
        df[df["Stint"] >= 2]
        .groupby(["Team", "Driver"], as_index=False)["LapNumber"]
        .min()
        .rename(columns={"LapNumber": "target_pit1_lap"})
    )
    pit2 = (
        df[df["Stint"] >= 3]
        .groupby(["Team", "Driver"], as_index=False)["LapNumber"]
        .min()
        .rename(columns={"LapNumber": "target_pit2_lap"})
    )
    out = per_driver.merge(pit1, on=["Team", "Driver"], how="left").merge(pit2, on=["Team", "Driver"], how="left")
    # Proxy target for undercut edge: outgoing stint late-lap median vs early settled lap median after stop.
    edge_rows: list[dict] = []
    for (team, drv), g in df.groupby(["Team", "Driver"]):
        first_pit = out[(out["Team"] == team) & (out["Driver"] == drv)]["target_pit1_lap"]
        if first_pit.empty or pd.isna(first_pit.iloc[0]):
            continue
        pit_lap = int(first_pit.iloc[0])
        pre = g[(g["Stint"] == 1) & (g["LapNumber"] >= pit_lap - 3) & (g["LapNumber"] <= pit_lap - 1)]["lap_seconds"]
        post = g[(g["Stint"] == 2) & (g["LapNumber"] >= pit_lap + 2) & (g["LapNumber"] <= pit_lap + 4)]["lap_seconds"]
        if len(pre) < 2 or len(post) < 2:
            continue
        edge = float(np.clip(pre.median() - post.median(), -0.8, 1.2))
        edge_rows.append({"Team": team, "Driver": drv, "target_edge_s": edge})
    edge_df = pd.DataFrame(edge_rows) if edge_rows else pd.DataFrame(columns=["Team", "Driver", "target_edge_s"])
    out = out.merge(edge_df, on=["Team", "Driver"], how="left")
    out = (
        out.groupby("Team", as_index=False)
        .agg(
            target_stops=("target_stops", "median"),
            target_pit1_lap=("target_pit1_lap", "median"),
            target_pit2_lap=("target_pit2_lap", "median"),
            target_edge_s=("target_edge_s", "median"),
        )
    )
    return out


def _strategy_training_frame(
    train_season: int,
    round_end: int,
    rounds: tuple[int, ...] | None = None,
    events: tuple[tuple[int, int], ...] | None = None,
) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    if events:
        event_iter = list(events)
    else:
        round_iter = list(rounds) if rounds else list(range(1, round_end + 1))
        event_iter = [(train_season, rnd) for rnd in round_iter]
    for yr, rnd in event_iter:
        try:
            bundle = load_sessions(yr, rnd)
            feats = _team_features_from_practice(bundle)
            labels = _team_labels_from_race(bundle)
            if feats.empty or labels.empty:
                continue
            merged = feats.merge(labels, on="Team", how="inner")
            merged["season"] = yr
            merged["round"] = rnd
            frames.append(merged)
        except Exception:
            continue
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def _historical_compound_prior(
    events: tuple[tuple[int, int], ...] | None,
) -> pd.DataFrame:
    """
    Team-level compound prior from same-GP historical races.
    Uses only race laps from the provided events (typically last 5 editions).
    """
    cols = [
        "Team",
        "hist_soft_share",
        "hist_medium_share",
        "hist_hard_share",
        "hist_soft_start_share",
        "hist_medium_start_share",
        "hist_hard_start_share",
        "hist_soft_stint1_laps",
        "hist_medium_stint1_laps",
        "hist_hard_stint1_laps",
        "hist_events_used",
    ]
    if not events:
        return pd.DataFrame(columns=cols)

    init_fastf1_cache()
    usage_rows: list[dict[str, Any]] = []
    start_rows: list[dict[str, Any]] = []
    stint_rows: list[dict[str, Any]] = []
    used_events = 0
    newest_year = max((int(y) for y, _ in events), default=0)
    for yr, rnd in events:
        try:
            race = fastf1.get_session(int(yr), int(rnd), "Race")
            race.load(laps=True, telemetry=False, weather=False, messages=False)
        except Exception:
            continue
        laps = laps_dataframe(race, "Race")
        if laps.empty:
            continue
        work = laps.copy()
        work["Compound"] = work["Compound"].fillna("UNKNOWN").astype(str).str.upper()
        work = work[work["Compound"].isin(["SOFT", "MEDIUM", "HARD"])].copy()
        if work.empty:
            continue
        used_events += 1
        year_w = 1.0 + 0.18 * max(0, newest_year - int(yr))
        recency_w = 1.0 / year_w

        # Usage share by race laps.
        team_comp = (
            work.groupby(["Team", "Compound"], as_index=False)
            .agg(laps=("lap_seconds", "count"))
        )
        team_comp["Team"] = team_comp["Team"].astype(str).map(_normalize_team_name)
        team_comp = team_comp.groupby(["Team", "Compound"], as_index=False)["laps"].sum()
        team_tot = team_comp.groupby("Team", as_index=False).agg(total_laps=("laps", "sum"))
        team_comp = team_comp.merge(team_tot, on="Team", how="left")
        team_comp["share"] = team_comp["laps"] / team_comp["total_laps"].replace(0, np.nan)
        for _, r in team_comp.iterrows():
            usage_rows.append(
                {
                    "Team": str(r["Team"]),
                    "Compound": str(r["Compound"]),
                    "share": float(r["share"]) if pd.notna(r["share"]) else np.nan,
                    "w": float(recency_w),
                }
            )

        # Start compound share from first stint per driver.
        st = work.copy()
        st["Stint"] = pd.to_numeric(st["Stint"], errors="coerce")
        st["LapNumber"] = pd.to_numeric(st["LapNumber"], errors="coerce")
        st = st.dropna(subset=["Team", "Driver", "Stint", "LapNumber"])
        if not st.empty:
            first = (
                st[st["Stint"] == 1]
                .sort_values(["Team", "Driver", "LapNumber"])
                .groupby(["Team", "Driver"], as_index=False)
                .first()[["Team", "Driver", "Compound"]]
            )
            first["Team"] = first["Team"].astype(str).map(_normalize_team_name)
            if not first.empty:
                start = (
                    first.groupby(["Team", "Compound"], as_index=False)
                    .agg(n=("Driver", "count"))
                )
                start_tot = start.groupby("Team", as_index=False).agg(total=("n", "sum"))
                start = start.merge(start_tot, on="Team", how="left")
                start["share"] = start["n"] / start["total"].replace(0, np.nan)
                for _, r in start.iterrows():
                    start_rows.append(
                        {
                            "Team": str(r["Team"]),
                            "Compound": str(r["Compound"]),
                            "share": float(r["share"]) if pd.notna(r["share"]) else np.nan,
                            "w": float(recency_w),
                        }
                    )
            # First-stint length by starting compound (helps identify durable starts).
            first_stint = (
                st[st["Stint"] == 1]
                .groupby(["Team", "Driver", "Compound"], as_index=False)
                .agg(stint1_laps=("LapNumber", "max"))
            )
            first_stint["Team"] = first_stint["Team"].astype(str).map(_normalize_team_name)
            for _, r in first_stint.iterrows():
                stint_rows.append(
                    {
                        "Team": str(r["Team"]),
                        "Compound": str(r["Compound"]),
                        "stint1_laps": float(r["stint1_laps"]),
                        "w": float(recency_w),
                    }
                )

    if used_events == 0 or not usage_rows:
        return pd.DataFrame(columns=cols)

    usage = pd.DataFrame(usage_rows).dropna(subset=["share", "w"])
    if usage.empty:
        return pd.DataFrame(columns=cols)
    usage["share_w"] = usage["share"] * usage["w"]
    usage = (
        usage.groupby(["Team", "Compound"], as_index=False)
        .agg(share_w=("share_w", "sum"), w_sum=("w", "sum"))
    )
    usage["share"] = usage["share_w"] / usage["w_sum"].replace(0, np.nan)
    usage = usage.pivot(index="Team", columns="Compound", values="share").fillna(0.0)
    for c in ["SOFT", "MEDIUM", "HARD"]:
        if c not in usage.columns:
            usage[c] = 0.0
    usage = usage[["SOFT", "MEDIUM", "HARD"]].rename(
        columns={
            "SOFT": "hist_soft_share",
            "MEDIUM": "hist_medium_share",
            "HARD": "hist_hard_share",
        }
    )

    if start_rows:
        start = pd.DataFrame(start_rows).dropna(subset=["share", "w"])
        start["share_w"] = start["share"] * start["w"]
        start = (
            start.groupby(["Team", "Compound"], as_index=False)
            .agg(share_w=("share_w", "sum"), w_sum=("w", "sum"))
        )
        start["share"] = start["share_w"] / start["w_sum"].replace(0, np.nan)
        start = start.pivot(index="Team", columns="Compound", values="share").fillna(0.0)
        for c in ["SOFT", "MEDIUM", "HARD"]:
            if c not in start.columns:
                start[c] = 0.0
        start = start[["SOFT", "MEDIUM", "HARD"]].rename(
            columns={
                "SOFT": "hist_soft_start_share",
                "MEDIUM": "hist_medium_start_share",
                "HARD": "hist_hard_start_share",
            }
        )
    else:
        start = pd.DataFrame(
            columns=["hist_soft_start_share", "hist_medium_start_share", "hist_hard_start_share"]
        )

    if stint_rows:
        st1 = pd.DataFrame(stint_rows).dropna(subset=["stint1_laps", "w"])
        st1["laps_w"] = st1["stint1_laps"] * st1["w"]
        st1 = (
            st1.groupby(["Team", "Compound"], as_index=False)
            .agg(laps_w=("laps_w", "sum"), w_sum=("w", "sum"))
        )
        st1["stint1_laps"] = st1["laps_w"] / st1["w_sum"].replace(0, np.nan)
        st1 = st1.pivot(index="Team", columns="Compound", values="stint1_laps").fillna(0.0)
        for c in ["SOFT", "MEDIUM", "HARD"]:
            if c not in st1.columns:
                st1[c] = 0.0
        st1 = st1[["SOFT", "MEDIUM", "HARD"]].rename(
            columns={
                "SOFT": "hist_soft_stint1_laps",
                "MEDIUM": "hist_medium_stint1_laps",
                "HARD": "hist_hard_stint1_laps",
            }
        )
    else:
        st1 = pd.DataFrame(columns=["hist_soft_stint1_laps", "hist_medium_stint1_laps", "hist_hard_stint1_laps"])

    out = usage.merge(start, left_index=True, right_index=True, how="left").merge(st1, left_index=True, right_index=True, how="left").reset_index()
    out["hist_events_used"] = int(used_events)
    for c in ["hist_soft_start_share", "hist_medium_start_share", "hist_hard_start_share"]:
        if c not in out.columns:
            out[c] = 0.0
        out[c] = out[c].fillna(0.0)
    for c in ["hist_soft_stint1_laps", "hist_medium_stint1_laps", "hist_hard_stint1_laps"]:
        if c not in out.columns:
            out[c] = 0.0
        out[c] = out[c].fillna(0.0)
    return out[cols]


def _make_regressor(seed: int) -> Pipeline:
    return Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            (
                "rf",
                RandomForestRegressor(
                    n_estimators=260,
                    random_state=seed,
                    min_samples_leaf=2,
                ),
            ),
        ]
    )


def _ml_strategy_predictions(
    current_features: pd.DataFrame,
    train_season: int,
    train_round_end: int,
    train_rounds: tuple[int, ...] | None = None,
    train_events: tuple[tuple[int, int], ...] | None = None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    if current_features.empty:
        return pd.DataFrame(), {"enabled": False, "reason": "no_current_features"}
    train_df = _strategy_training_frame(train_season, train_round_end, rounds=train_rounds, events=train_events)
    feature_cols = [
        "feat_deg",
        "feat_total_laps",
        "feat_best_pace",
        "feat_soft_gap",
        "feat_medium_gap",
        "feat_hard_gap",
    ]
    if len(train_df) < 6:
        return pd.DataFrame(), {"enabled": False, "reason": "insufficient_training_rows", "rows": int(len(train_df))}

    pred = current_features[["Team"] + feature_cols].copy()
    meta: dict[str, Any] = {
        "enabled": True,
        "rows": int(len(train_df)),
        "train_season": train_season,
        "train_round_end": train_round_end,
        "train_rounds": list(train_rounds) if train_rounds else None,
        "train_events": [f"{y}-R{r}" for (y, r) in train_events] if train_events else None,
    }

    stop_train = train_df.dropna(subset=["target_stops"]).copy()
    if len(stop_train) >= 6:
        m_stop = _make_regressor(31)
        m_stop.fit(stop_train[feature_cols], stop_train["target_stops"])
        pred["ml_expected_stops"] = np.clip(m_stop.predict(pred[feature_cols]), 1.0, 3.2)
    else:
        pred["ml_expected_stops"] = float(np.clip(stop_train["target_stops"].median(), 1.0, 3.2)) if len(stop_train) else np.nan
    pit1_train = train_df.dropna(subset=["target_pit1_lap"]).copy()
    if len(pit1_train) >= 6:
        m_p1 = _make_regressor(37)
        m_p1.fit(pit1_train[feature_cols], pit1_train["target_pit1_lap"])
        pred["ml_pit1_lap"] = m_p1.predict(pred[feature_cols])
    else:
        pred["ml_pit1_lap"] = float(pit1_train["target_pit1_lap"].median()) if len(pit1_train) else np.nan
    pit2_train = train_df.dropna(subset=["target_pit2_lap"]).copy()
    if len(pit2_train) >= 6:
        m_p2 = _make_regressor(41)
        m_p2.fit(pit2_train[feature_cols], pit2_train["target_pit2_lap"])
        pred["ml_pit2_lap"] = m_p2.predict(pred[feature_cols])
    else:
        pred["ml_pit2_lap"] = float(pit2_train["target_pit2_lap"].median()) if len(pit2_train) else np.nan
    edge_train = train_df.dropna(subset=["target_edge_s"]).copy()
    if len(edge_train) >= 6:
        m_edge = _make_regressor(53)
        m_edge.fit(edge_train[feature_cols], edge_train["target_edge_s"])
        pred["ml_edge_s"] = np.clip(m_edge.predict(pred[feature_cols]), -0.8, 1.2)
    else:
        pred["ml_edge_s"] = float(np.clip(edge_train["target_edge_s"].median(), -0.8, 1.2)) if len(edge_train) else np.nan
    # Confidence is intentionally conservative and tied to available historical team-round rows.
    meta["confidence"] = round(float(np.clip(0.40 + (len(train_df) / 200.0), 0.40, 0.82)), 2)
    return pred, meta


def build_strategy_overview(
    bundle: object,
    race_predictions: pd.DataFrame | None = None,
    strategy_train_season: int | None = None,
    strategy_train_round_end: int = 8,
    strategy_train_rounds: tuple[int, ...] | None = None,
    strategy_train_events: tuple[tuple[int, int], ...] | None = None,
) -> StrategyOverview:
    pool = _practice_pool(bundle)
    if pool.empty:
        return StrategyOverview([], [], [], [], [], [], [], {"enabled": False, "reason": "no_practice_pool"}, 0)

    race_sim = _race_sim_pool(pool)
    pace_src = race_sim if not race_sim.empty else pool
    pace = (
        pace_src.groupby(["Team", "Compound"], as_index=False)
        .agg(median_pace_s=("lap_seconds", "median"), laps=("lap_seconds", "count"))
    )
    pace = pace[pace["Compound"].isin(["SOFT", "MEDIUM", "HARD"])]
    slopes = _team_compound_slopes(race_sim if not race_sim.empty else pool)

    total_laps = _race_lap_count(bundle)
    default_slopes = {"SOFT": 0.090, "MEDIUM": 0.070, "HARD": 0.055}
    global_comp_pace = (
        pace.groupby("Compound", as_index=False)["median_pace_s"]
        .median()
        .set_index("Compound")["median_pace_s"]
        .to_dict()
    )
    ordered_teams = (
        pace.groupby("Team", as_index=False)["median_pace_s"]
        .min()
        .sort_values("median_pace_s")["Team"]
        .astype(str)
        .tolist()
    )

    strategy_rows: list[dict] = []
    stint_rows: list[dict] = []
    team_rows: list[dict] = []
    undercut_rows: list[dict] = []
    extension_rows: list[dict] = []
    race_mode_rows: list[dict] = []
    current_feats = _team_features_from_practice(bundle)
    train_season = strategy_train_season if strategy_train_season is not None else max(2018, int(bundle.season) - 1)
    ml_pred_df, ml_meta = _ml_strategy_predictions(
        current_feats,
        train_season=train_season,
        train_round_end=strategy_train_round_end,
        train_rounds=strategy_train_rounds,
        train_events=strategy_train_events,
    )
    ml_by_team: dict[str, dict[str, Any]] = {}
    if not ml_pred_df.empty:
        ml_by_team = ml_pred_df.set_index("Team").to_dict(orient="index")
    ml_weight = 0.0
    if bool(ml_meta.get("enabled", False)):
        # Keep current-weekend signals dominant and use ML as a secondary stabilizer.
        ml_weight = float(np.clip(0.10 + 0.22 * float(ml_meta.get("confidence", 0.55)), 0.18, 0.30))
    event_name = str(getattr(bundle, "event_name", "") or "").strip().lower()
    prefer_one_stop = bool(int(getattr(bundle, "season", 0)) == 2026 and int(getattr(bundle, "round_number", 0)) == 1 and "australian" in event_name)
    hist_prior_df = _historical_compound_prior(strategy_train_events)
    hist_by_team: dict[str, dict[str, Any]] = {}
    if not hist_prior_df.empty:
        hist_by_team = hist_prior_df.set_index("Team").to_dict(orient="index")

    for team in ordered_teams:
        tpace = pace[pace["Team"] == team].copy()
        if tpace.empty:
            continue
        tpace = tpace.sort_values("median_pace_s")
        comp_pace = {r["Compound"]: float(r["median_pace_s"]) for _, r in tpace.iterrows()}
        for c in ["SOFT", "MEDIUM", "HARD"]:
            comp_pace.setdefault(c, float(global_comp_pace.get(c, tpace["median_pace_s"].median())))

        # Hybrid compound selection:
        # - current weekend pace/degradation (heuristic primary)
        # - same-GP historical stint usage from last editions (prior secondary)
        ordered_comp = sorted(comp_pace.items(), key=lambda x: x[1])
        hist_row = hist_by_team.get(_normalize_team_name(team), {})
        hist_share = {
            "SOFT": float(hist_row.get("hist_soft_share", 1.0 / 3.0)),
            "MEDIUM": float(hist_row.get("hist_medium_share", 1.0 / 3.0)),
            "HARD": float(hist_row.get("hist_hard_share", 1.0 / 3.0)),
        }
        hist_start = {
            "SOFT": float(hist_row.get("hist_soft_start_share", 1.0 / 3.0)),
            "MEDIUM": float(hist_row.get("hist_medium_start_share", 1.0 / 3.0)),
            "HARD": float(hist_row.get("hist_hard_start_share", 1.0 / 3.0)),
        }
        hist_stint1 = {
            "SOFT": float(hist_row.get("hist_soft_stint1_laps", 0.0)),
            "MEDIUM": float(hist_row.get("hist_medium_stint1_laps", 0.0)),
            "HARD": float(hist_row.get("hist_hard_stint1_laps", 0.0)),
        }
        tsl = slopes[slopes["Team"] == team]
        slope_map = {r["Compound"]: float(r["deg_s_per_lap"]) for _, r in tsl.iterrows()}

        # Soft start gate:
        # requires long-run viability + meaningful historical same-GP support.
        medium_p = comp_pace.get("MEDIUM", np.inf)
        hard_p = comp_pace.get("HARD", np.inf)
        soft_p = comp_pace.get("SOFT", np.inf)
        base_comp = float(min(comp_pace.values()))
        comp_delta = {c: float(comp_pace[c] - base_comp) for c in ["SOFT", "MEDIUM", "HARD"]}
        soft_deg = float(slope_map.get("SOFT", default_slopes["SOFT"]))
        med_deg = float(slope_map.get("MEDIUM", default_slopes["MEDIUM"]))
        hard_deg = float(slope_map.get("HARD", default_slopes["HARD"]))
        soft_long = soft_p + soft_deg * 9.0
        med_long = medium_p + med_deg * 9.0
        hard_long = hard_p + hard_deg * 9.0

        soft_hist_support = 0.55 * hist_start["SOFT"] + 0.45 * hist_share["SOFT"]
        soft_durable_hist = hist_stint1["SOFT"] >= max(8.0, 0.20 * total_laps)
        soft_long_run_ok = soft_long <= min(med_long, hard_long) + 0.06
        soft_deg_ok = soft_deg <= (med_deg + 0.015)
        use_soft_start = bool(soft_long_run_ok and soft_deg_ok and (soft_hist_support >= 0.28 or soft_durable_hist))

        comp_score = {
            c: (
                comp_delta[c]
                + 0.28 * (1.0 - hist_share[c])   # penalize compounds rarely used in recent same-GP races
                + 0.20 * (1.0 - hist_start[c])   # penalize unlikely opening compounds
            )
            for c in ["SOFT", "MEDIUM", "HARD"]
        }
        # Add long-run/race-pace penalty so short-run soft headline pace doesn't dominate.
        comp_score["SOFT"] += max(0.0, soft_long - min(med_long, hard_long))
        comp_score["MEDIUM"] += max(0.0, med_long - min(soft_long, hard_long)) * 0.55
        comp_score["HARD"] += max(0.0, hard_long - min(soft_long, med_long)) * 0.55
        if not use_soft_start:
            comp_score["SOFT"] += 2.00

        ordered_comp = sorted(comp_score.items(), key=lambda x: x[1])
        # User preference: strategy suggestions should only use Medium/Hard.
        candidate = [c for c, _ in ordered_comp if c in {"MEDIUM", "HARD"}]
        if len(candidate) < 2:
            candidate = ["MEDIUM", "HARD"]
        if len(candidate) == 2:
            candidate.append(candidate[1])
        best_comp, second_comp, third_comp = candidate[0], candidate[1], candidate[2]

        deg = float(np.nanmedian([slope_map.get(c, default_slopes[c]) for c in ["SOFT", "MEDIUM", "HARD"]]))
        deg = float(np.clip(deg, 0.03, 0.14))

        sig = 1.0 / (1.0 + np.exp(-((deg - 0.065) / 0.012)))
        p1_h = float(np.clip(0.72 - 0.50 * sig, 0.05, 0.90))
        p2_h = float(np.clip(0.22 + 0.42 * sig, 0.05, 0.90))
        p3_h = float(max(0.01, 1.0 - p1_h - p2_h))
        norm = p1_h + p2_h + p3_h
        p1_h, p2_h, p3_h = p1_h / norm, p2_h / norm, p3_h / norm
        exp_stops_h = 1.0 * p1_h + 2.0 * p2_h + 3.0 * p3_h
        stop_mode_h = int(np.clip(round(exp_stops_h), 1, 3))

        if stop_mode_h == 1:
            c1h = int(0.48 * total_laps - np.clip((deg - 0.06) * 120, -6, 6))
            c1h = int(np.clip(c1h, 6, total_laps - 6))
            h_w1s, h_w1e = max(8, c1h - 3), min(total_laps - 8, c1h + 3)
            h_w2s, h_w2e = None, None
        elif stop_mode_h == 2:
            c1h = int(0.30 * total_laps - np.clip((deg - 0.07) * 90, -5, 5))
            c2h = int(0.66 * total_laps - np.clip((deg - 0.07) * 70, -4, 4))
            h_w1s, h_w1e = max(6, c1h - 3), min(total_laps - 14, c1h + 3)
            h_w2s, h_w2e = max(18, c2h - 3), min(total_laps - 4, c2h + 3)
        else:
            c1h = int(0.22 * total_laps)
            c2h = int(0.48 * total_laps)
            h_w1s, h_w1e = max(5, c1h - 2), min(total_laps - 20, c1h + 2)
            h_w2s, h_w2e = max(14, c2h - 2), min(total_laps - 8, c2h + 2)

        ml_row = ml_by_team.get(team, {})
        ml_exp_raw = ml_row.get("ml_expected_stops")
        ml_exp = None if ml_exp_raw is None or (isinstance(ml_exp_raw, float) and pd.isna(ml_exp_raw)) else float(ml_exp_raw)
        exp_stops = float(exp_stops_h if ml_exp is None else (1.0 - ml_weight) * exp_stops_h + ml_weight * ml_exp)
        exp_stops = float(np.clip(exp_stops, 1.0, 3.0))
        # One-race override requested: bias Australia 2026 recommendations toward 1-stop.
        if prefer_one_stop:
            exp_stops = float(np.clip(exp_stops - 0.32, 1.0, 3.0))
        stop_mode = int(np.clip(round(exp_stops), 1, 3))

        ml_p1_raw = ml_row.get("ml_pit1_lap")
        ml_p2_raw = ml_row.get("ml_pit2_lap")
        ml_p1 = None if ml_p1_raw is None or (isinstance(ml_p1_raw, float) and pd.isna(ml_p1_raw)) else int(round(float(ml_p1_raw)))
        ml_p2 = None if ml_p2_raw is None or (isinstance(ml_p2_raw, float) and pd.isna(ml_p2_raw)) else int(round(float(ml_p2_raw)))
        if ml_p1 is not None:
            ml_p1 = int(np.clip(ml_p1, 6, total_laps - 6))
        if ml_p2 is not None:
            ml_p2 = int(np.clip(ml_p2, 14, total_laps - 4))

        if h_w1s is not None and h_w1e is not None and ml_p1 is not None:
            ml_w1s = max(6, ml_p1 - 3)
            ml_w1e = min(total_laps - 6, ml_p1 + 3)
            w1s = int(round((1.0 - ml_weight) * h_w1s + ml_weight * ml_w1s))
            w1e = int(round((1.0 - ml_weight) * h_w1e + ml_weight * ml_w1e))
        else:
            w1s, w1e = h_w1s, h_w1e

        if stop_mode >= 2:
            if h_w2s is not None and h_w2e is not None and ml_p2 is not None:
                ml_w2s = max(14, ml_p2 - 3)
                ml_w2e = min(total_laps - 4, ml_p2 + 3)
                w2s = int(round((1.0 - ml_weight) * h_w2s + ml_weight * ml_w2s))
                w2e = int(round((1.0 - ml_weight) * h_w2e + ml_weight * ml_w2e))
            else:
                w2s, w2e = h_w2s, h_w2e
            if w2s is None or w2e is None:
                pivot = int(0.66 * total_laps)
                w2s, w2e = max(18, pivot - 3), min(total_laps - 4, pivot + 3)
        else:
            w2s, w2e = None, None

        w1 = f"L{int(w1s)}-L{int(w1e)}" if w1s is not None and w1e is not None else ""
        w2 = f"L{int(w2s)}-L{int(w2e)}" if w2s is not None and w2e is not None else ""
        h_w1 = f"L{int(h_w1s)}-L{int(h_w1e)}" if h_w1s is not None and h_w1e is not None else ""
        h_w2 = f"L{int(h_w2s)}-L{int(h_w2e)}" if h_w2s is not None and h_w2e is not None else ""
        ml_w1 = f"L{max(6, ml_p1 - 3)}-L{min(total_laps - 6, ml_p1 + 3)}" if ml_p1 is not None else ""
        ml_w2 = f"L{max(14, ml_p2 - 3)}-L{min(total_laps - 4, ml_p2 + 3)}" if ml_p2 is not None else ""

        if exp_stops <= 2.0:
            p1 = float(np.clip(2.0 - exp_stops, 0.0, 1.0))
            p2 = float(np.clip(exp_stops - 1.0, 0.0, 1.0))
            p3 = 0.0
        else:
            p1 = 0.0
            p2 = float(np.clip(3.0 - exp_stops, 0.0, 1.0))
            p3 = float(np.clip(exp_stops - 2.0, 0.0, 1.0))

        if stop_mode == 1:
            strategy = f"{best_comp} -> {second_comp}"
            pit1_from, pit1_to = best_comp, second_comp
            pit2_from, pit2_to = None, None
        elif stop_mode == 2:
            strategy = f"{best_comp} -> {second_comp} -> {third_comp}"
            pit1_from, pit1_to = best_comp, second_comp
            pit2_from, pit2_to = second_comp, third_comp
        else:
            strategy = f"{best_comp} -> {second_comp} -> {best_comp} -> {third_comp}"
            pit1_from, pit1_to = best_comp, second_comp
            pit2_from, pit2_to = second_comp, best_comp

        base = float(min(comp_pace.values()))
        soft = comp_pace.get("SOFT", base + 0.20) + slope_map.get("SOFT", default_slopes["SOFT"]) * 7.0
        medium = comp_pace.get("MEDIUM", base + 0.35) + slope_map.get("MEDIUM", default_slopes["MEDIUM"]) * 7.0
        hard = comp_pace.get("HARD", base + 0.55) + slope_map.get("HARD", default_slopes["HARD"]) * 7.0

        strategy_rows.append(
            {
                "Team": team,
                "expected_stops": round(exp_stops, 2),
                "expected_stops_heuristic": round(exp_stops_h, 2),
                "expected_stops_ml": round(ml_exp, 2) if ml_exp is not None else None,
                "p_1_stop": round(p1 * 100.0, 1),
                "p_2_stop": round(p2 * 100.0, 1),
                "p_3_stop": round(p3 * 100.0, 1),
                "strategy": strategy,
                "pit_window_1": w1,
                "pit_window_2": w2,
                "pit_window_1_heuristic": h_w1,
                "pit_window_2_heuristic": h_w2,
                "pit_window_1_ml": ml_w1,
                "pit_window_2_ml": ml_w2,
                "pit1_transition": f"{pit1_from} -> {pit1_to}" if pit1_from and pit1_to else "",
                "pit2_transition": f"{pit2_from} -> {pit2_to}" if pit2_from and pit2_to else "",
                "start_compound": pit1_from if pit1_from else "",
                "post_pit1_compound": pit1_to if pit1_to else "",
                "post_pit2_compound": pit2_to if pit2_to else (pit1_to if pit1_to else ""),
                "pit1_to_compound": pit1_to if pit1_to else "",
                "pit2_to_compound": pit2_to if pit2_to else "",
                "pit1_start": w1s,
                "pit1_end": w1e,
                "pit2_start": w2s,
                "pit2_end": w2e,
                "strategy_source": "Hybrid (Heuristic + Historical Prior + ML)" if bool(ml_meta.get("enabled", False)) else "Heuristic + Historical Prior",
            }
        )
        # Hybrid undercut/overcut edge using heuristic baseline + ML edge estimate.
        out_deg = float(slope_map.get(best_comp, default_slopes.get(best_comp, 0.07)))
        in_deg = float(slope_map.get(second_comp, default_slopes.get(second_comp, 0.07)))
        out_base = float(comp_pace.get(best_comp, base))
        in_base = float(comp_pace.get(second_comp, base))
        undercut_h = float(np.clip((out_deg * 4.2) + (out_base - in_base) + 0.22, 0.03, 1.25))
        overcut_h = float(np.clip((in_base - out_base) + ((0.09 - out_deg) * 2.8) + 0.12, 0.03, 0.95))
        ml_edge_raw = ml_row.get("ml_edge_s")
        ml_edge = None if ml_edge_raw is None or (isinstance(ml_edge_raw, float) and pd.isna(ml_edge_raw)) else float(ml_edge_raw)
        if ml_edge is None:
            undercut_gain = undercut_h
            overcut_gain = overcut_h
        else:
            undercut_ml = float(np.clip(0.24 + max(0.0, ml_edge), 0.03, 1.25))
            overcut_ml = float(np.clip(0.24 + max(0.0, -ml_edge), 0.03, 0.95))
            undercut_gain = float(np.clip((1.0 - ml_weight) * undercut_h + ml_weight * undercut_ml, 0.03, 1.25))
            overcut_gain = float(np.clip((1.0 - ml_weight) * overcut_h + ml_weight * overcut_ml, 0.03, 0.95))

        edge_diff = float(undercut_gain - overcut_gain)
        total_edge = max(0.001, undercut_gain + overcut_gain)
        undercut_viability = int(round(np.clip((undercut_gain / total_edge) * 100.0, 5.0, 95.0)))
        overcut_viability = int(100 - undercut_viability)
        if undercut_gain > overcut_gain + 0.08:
            preference = "UNDERCUT"
        elif overcut_gain > undercut_gain + 0.08:
            preference = "OVERCUT"
        else:
            preference = "BALANCED"
        trigger_lap = int(round((w1s + w1e) / 2.0)) if w1s is not None and w1e is not None else None
        undercut_rows.append(
            {
                "Team": team,
                "preferred_call": preference,
                "pit_trigger_window": w1,
                "trigger_lap": trigger_lap,
                "undercut_gain_s": round(float(undercut_gain), 3),
                "overcut_gain_s": round(float(overcut_gain), 3),
                "edge_diff_s": round(edge_diff, 3),
                "undercut_viability_pct": undercut_viability,
                "overcut_viability_pct": overcut_viability,
            }
        )
        # Hybrid stint-extension recommendation for first stop decision.
        extend_laps = int(np.clip(round((0.095 - deg) * 70.0 + undercut_viability * 0.04), 0, 8))
        extend_penalty = float(np.clip(deg * max(1, extend_laps) * 0.9, 0.06, 0.75))
        extend_viability = int(np.clip(round(65.0 + extend_laps * 4.5 - extend_penalty * 55.0), 15, 92))
        if extend_laps >= 4 and extend_penalty <= 0.30:
            extend_call = "EXTEND"
        elif extend_penalty >= 0.45:
            extend_call = "BOX ON WINDOW"
        else:
            extend_call = "MARGINAL"
        extension_rows.append(
            {
                "Team": team,
                "stint_compound": best_comp,
                "pit_window": w1,
                "max_extend_laps": extend_laps,
                "expected_penalty_s": round(extend_penalty, 3),
                "extension_viability_pct": extend_viability,
                "extension_call": extend_call,
            }
        )

        # Hybrid race-mode recommendation (aggressive/defensive) for stint management.
        aggr_raw = 50.0 + edge_diff * 90.0 + (2.0 - exp_stops) * 7.0 - deg * 90.0 + undercut_viability * 0.25
        aggr_score = int(np.clip(round(aggr_raw), 5, 95))
        def_score = int(100 - aggr_score)
        if aggr_score >= 58:
            mode = "AGGRESSIVE"
            mode_hint = "Push undercut threat and attack around trigger lap."
        elif aggr_score <= 42:
            mode = "DEFENSIVE"
            mode_hint = "Protect tyre life and cover rivals in the pit window."
        else:
            mode = "BALANCED"
            mode_hint = "Flexible call: react to traffic and tyre state."
        race_mode_rows.append(
            {
                "Team": team,
                "mode": mode,
                "aggressive_score_pct": aggr_score,
                "defensive_score_pct": def_score,
                "primary_trigger": w1,
                "mode_hint": mode_hint,
            }
        )
        min_comp = min(soft, medium, hard)
        stint_rows.append(
            {
                "Team": team,
                "soft_pace_fmt": _format_laptime(soft),
                "medium_pace_fmt": _format_laptime(medium),
                "hard_pace_fmt": _format_laptime(hard),
                "soft_delta_ms": round((soft - min_comp) * 1000.0, 1),
                "medium_delta_ms": round((medium - min_comp) * 1000.0, 1),
                "hard_delta_ms": round((hard - min_comp) * 1000.0, 1),
            }
        )
        team_rows.append({"Team": team, "deg_s_per_lap": round(deg, 3), "expected_stops": round(exp_stops, 2)})

    team_df = pd.DataFrame(team_rows)
    if not team_df.empty:
        team_df = (
            team_df.groupby("Team", as_index=False)
            .agg(mean_deg_s_per_lap=("deg_s_per_lap", "mean"), avg_expected_stops=("expected_stops", "mean"))
            .sort_values("mean_deg_s_per_lap")
        )
        team_outlook_rows = team_df.to_dict(orient="records")
    else:
        team_outlook_rows = []

    deg_mean = float(np.nanmean([r["deg_s_per_lap"] for r in team_rows])) if team_rows else 0.07
    one_c = int(0.48 * total_laps - np.clip((deg_mean - 0.06) * 120, -5, 5))
    two_c1 = int(0.30 * total_laps - np.clip((deg_mean - 0.07) * 90, -4, 4))
    two_c2 = int(0.66 * total_laps - np.clip((deg_mean - 0.07) * 70, -3, 3))
    pit_window_reference = [
        {"plan": "One-stop reference", "window_1": f"L{max(8, one_c-3)}-L{min(total_laps-8, one_c+3)}", "window_2": ""},
        {"plan": "Two-stop reference", "window_1": f"L{max(6, two_c1-3)}-L{min(total_laps-14, two_c1+3)}", "window_2": f"L{max(18, two_c2-3)}-L{min(total_laps-4, two_c2+3)}"},
    ]

    return StrategyOverview(
        strategy_rows=strategy_rows,
        stint_pace_rows=stint_rows,
        team_outlook_rows=team_outlook_rows,
        undercut_rows=undercut_rows,
        stint_extension_rows=extension_rows,
        race_mode_rows=race_mode_rows,
        pit_window_reference=pit_window_reference,
        hybrid_meta={
            **ml_meta,
            "blend_weight_ml": round(ml_weight, 2),
            "blend_weight_heuristic": round(1.0 - ml_weight, 2),
        },
        total_laps=total_laps,
    )
