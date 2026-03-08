from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from typing import Any

import fastf1
import numpy as np
import pandas as pd

from .data import (
    init_fastf1_cache,
    laps_dataframe,
    load_sessions,
    qualifying_parts_dataframes,
    results_dataframe,
)


@dataclass
class ModelOutputs:
    quali_predictions: pd.DataFrame
    race_predictions: pd.DataFrame
    model_metrics: dict[str, Any]


def _rank_score(s: pd.Series) -> pd.Series:
    x = pd.to_numeric(s, errors="coerce")
    if x.notna().sum() == 0:
        return pd.Series(np.arange(1, len(s) + 1, dtype=float), index=s.index)
    r = x.rank(method="average")
    tail = float(r.max(skipna=True)) if r.notna().any() else float(len(s))
    return r.fillna(tail + 1.0)


def _event_round_numbers(season: int) -> list[int]:
    try:
        sched = fastf1.get_event_schedule(season, include_testing=False)
        if sched is None or sched.empty:
            return []
        if "RoundNumber" not in sched.columns:
            return []
        rounds = (
            pd.to_numeric(sched["RoundNumber"], errors="coerce")
            .dropna()
            .astype(int)
            .tolist()
        )
        return sorted(set(r for r in rounds if r > 0))
    except Exception:
        return []


@lru_cache(maxsize=8)
def _season_position_priors(prior_season: int) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    init_fastf1_cache()
    rounds = _event_round_numbers(prior_season)
    q_rows: list[dict[str, Any]] = []
    r_rows: list[dict[str, Any]] = []

    for rnd in rounds:
        try:
            q = fastf1.get_session(prior_season, rnd, "Qualifying")
            q.load(laps=False, telemetry=False, weather=False, messages=False)
            q_res = results_dataframe(q, "Qualifying")[["Driver", "Team", "Position"]].copy()
            q_res["Position"] = pd.to_numeric(q_res["Position"], errors="coerce")
            q_res = q_res.dropna(subset=["Driver", "Team", "Position"])
            q_rows.extend(
                q_res.rename(columns={"Position": "quali_position"}).to_dict(orient="records")
            )
        except Exception:
            pass

        try:
            r = fastf1.get_session(prior_season, rnd, "Race")
            r.load(laps=False, telemetry=False, weather=False, messages=False)
            r_res = results_dataframe(r, "Race")[["Driver", "Team", "Position"]].copy()
            r_res["Position"] = pd.to_numeric(r_res["Position"], errors="coerce")
            r_res = r_res.dropna(subset=["Driver", "Team", "Position"])
            r_rows.extend(
                r_res.rename(columns={"Position": "race_position"}).to_dict(orient="records")
            )
        except Exception:
            pass

    if q_rows:
        q_df = pd.DataFrame(q_rows)
        drv_q = q_df.groupby("Driver", as_index=False).agg(driver_prev_quali_avg=("quali_position", "mean"))
        team_q = q_df.groupby("Team", as_index=False).agg(team_prev_quali_avg=("quali_position", "mean"))
    else:
        drv_q = pd.DataFrame(columns=["Driver", "driver_prev_quali_avg"])
        team_q = pd.DataFrame(columns=["Team", "team_prev_quali_avg"])

    if r_rows:
        r_df = pd.DataFrame(r_rows)
        drv_r = r_df.groupby("Driver", as_index=False).agg(driver_prev_race_avg=("race_position", "mean"))
        team_r = r_df.groupby("Team", as_index=False).agg(team_prev_race_avg=("race_position", "mean"))
    else:
        drv_r = pd.DataFrame(columns=["Driver", "driver_prev_race_avg"])
        team_r = pd.DataFrame(columns=["Team", "team_prev_race_avg"])

    return drv_q, team_q, drv_r, team_r


def _event_round_number(season: int, event_name: str) -> int | None:
    try:
        sched = fastf1.get_event_schedule(season, include_testing=False)
        if sched is None or sched.empty:
            return None
        if "EventName" not in sched.columns or "RoundNumber" not in sched.columns:
            return None
        match = sched[sched["EventName"].astype(str).str.strip().str.lower() == event_name.strip().lower()]
        if match.empty:
            return None
        val = pd.to_numeric(match.iloc[0]["RoundNumber"], errors="coerce")
        if pd.isna(val):
            return None
        return int(val)
    except Exception:
        return None


def _overtake_difficulty(event_name: str | None) -> float:
    """Return [0,1] where higher means harder overtaking."""
    name = (event_name or "").strip().lower()
    # Lightweight circuit prior for race recovery potential.
    # Values are intentionally conservative and can be tuned later with data.
    if "monaco" in name:
        return 0.98
    if "hungarian" in name or "budapest" in name:
        return 0.86
    if "zandvoort" in name or "dutch" in name:
        return 0.80
    if "jeddah" in name or "saudi" in name:
        return 0.72
    if "australian" in name or "albert park" in name:
        return 0.66
    if "imola" in name or "emilia" in name:
        return 0.74
    if "singapore" in name:
        return 0.78
    if "las vegas" in name or "monza" in name or "baku" in name or "spa" in name:
        return 0.42
    return 0.60


@lru_cache(maxsize=32)
def _overtake_difficulty_last5_same_gp(season: int, event_name: str) -> tuple[float, int]:
    """
    Data-driven overtaking difficulty from last 5 editions of the same GP.
    Uses race results only (grid vs finish), no lap/telemetry pull.
    Returns (difficulty[0..1], races_used).
    """
    init_fastf1_cache()
    name = (event_name or "").strip()
    if not name:
        return _overtake_difficulty(name), 0

    difficulties: list[float] = []
    max_lookback_years = 15
    for yr in range(season - 1, season - max_lookback_years - 1, -1):
        if len(difficulties) >= 5:
            break
        try:
            rnd = _event_round_number(yr, name)
            if rnd is None:
                continue
            r = fastf1.get_session(yr, int(rnd), "Race")
            r.load(laps=False, telemetry=False, weather=False, messages=False)
            res = results_dataframe(r, "Race")
            if res.empty:
                continue
            work = res[["GridPosition", "Position"]].copy()
            work["GridPosition"] = pd.to_numeric(work["GridPosition"], errors="coerce")
            work["Position"] = pd.to_numeric(work["Position"], errors="coerce")
            work = work.dropna(subset=["GridPosition", "Position"])
            work = work[(work["GridPosition"] > 0) & (work["Position"] > 0)]
            if len(work) < 10:
                continue

            mean_abs_change = float((work["Position"] - work["GridPosition"]).abs().mean())
            rho = float(work["GridPosition"].corr(work["Position"], method="spearman"))
            if np.isnan(rho):
                continue

            # Fewer position changes + stronger grid/finish correlation => harder overtaking.
            hard_from_moves = float(np.clip(1.0 - (mean_abs_change - 1.5) / 5.0, 0.0, 1.0))
            hard_from_rho = float(np.clip((rho + 1.0) / 2.0, 0.0, 1.0))
            diff = 0.60 * hard_from_rho + 0.40 * hard_from_moves
            difficulties.append(float(np.clip(diff, 0.0, 1.0)))
        except Exception:
            continue

    if difficulties:
        return float(np.mean(difficulties)), len(difficulties)
    return _overtake_difficulty(name), 0


@lru_cache(maxsize=16)
def _event_position_priors(prior_season: int, event_name: str) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    init_fastf1_cache()
    rnd = _event_round_number(prior_season, event_name)
    if rnd is None:
        return (
            pd.DataFrame(columns=["Driver", "driver_prev_quali_circuit"]),
            pd.DataFrame(columns=["Team", "team_prev_quali_circuit"]),
            pd.DataFrame(columns=["Driver", "driver_prev_race_circuit"]),
            pd.DataFrame(columns=["Team", "team_prev_race_circuit"]),
        )

    q_rows: list[dict[str, Any]] = []
    r_rows: list[dict[str, Any]] = []
    try:
        q = fastf1.get_session(prior_season, rnd, "Qualifying")
        q.load(laps=False, telemetry=False, weather=False, messages=False)
        q_res = results_dataframe(q, "Qualifying")[["Driver", "Team", "Position"]].copy()
        q_res["Position"] = pd.to_numeric(q_res["Position"], errors="coerce")
        q_res = q_res.dropna(subset=["Driver", "Team", "Position"])
        q_rows.extend(q_res.rename(columns={"Position": "quali_position"}).to_dict(orient="records"))
    except Exception:
        pass
    try:
        r = fastf1.get_session(prior_season, rnd, "Race")
        r.load(laps=False, telemetry=False, weather=False, messages=False)
        r_res = results_dataframe(r, "Race")[["Driver", "Team", "Position"]].copy()
        r_res["Position"] = pd.to_numeric(r_res["Position"], errors="coerce")
        r_res = r_res.dropna(subset=["Driver", "Team", "Position"])
        r_rows.extend(r_res.rename(columns={"Position": "race_position"}).to_dict(orient="records"))
    except Exception:
        pass

    if q_rows:
        q_df = pd.DataFrame(q_rows)
        drv_q = q_df.groupby("Driver", as_index=False).agg(driver_prev_quali_circuit=("quali_position", "mean"))
        team_q = q_df.groupby("Team", as_index=False).agg(team_prev_quali_circuit=("quali_position", "mean"))
    else:
        drv_q = pd.DataFrame(columns=["Driver", "driver_prev_quali_circuit"])
        team_q = pd.DataFrame(columns=["Team", "team_prev_quali_circuit"])
    if r_rows:
        r_df = pd.DataFrame(r_rows)
        drv_r = r_df.groupby("Driver", as_index=False).agg(driver_prev_race_circuit=("race_position", "mean"))
        team_r = r_df.groupby("Team", as_index=False).agg(team_prev_race_circuit=("race_position", "mean"))
    else:
        drv_r = pd.DataFrame(columns=["Driver", "driver_prev_race_circuit"])
        team_r = pd.DataFrame(columns=["Team", "team_prev_race_circuit"])
    return drv_q, team_q, drv_r, team_r


def _attach_priors(df: pd.DataFrame, prior_season: int, event_name: str | None = None) -> pd.DataFrame:
    out = df.copy()
    drv_q, team_q, drv_r, team_r = _season_position_priors(prior_season)
    out = out.merge(drv_q, on="Driver", how="left")
    out = out.merge(team_q, on="Team", how="left")
    out = out.merge(drv_r, on="Driver", how="left")
    out = out.merge(team_r, on="Team", how="left")
    if event_name:
        c_drv_q, c_team_q, c_drv_r, c_team_r = _event_position_priors(prior_season, event_name)
        out = out.merge(c_drv_q, on="Driver", how="left")
        out = out.merge(c_team_q, on="Team", how="left")
        out = out.merge(c_drv_r, on="Driver", how="left")
        out = out.merge(c_team_r, on="Team", how="left")
    else:
        out["driver_prev_quali_circuit"] = np.nan
        out["team_prev_quali_circuit"] = np.nan
        out["driver_prev_race_circuit"] = np.nan
        out["team_prev_race_circuit"] = np.nan
    return out


def _practice_features(bundle: object) -> pd.DataFrame:
    parts: list[pd.DataFrame] = []
    for s in ("FP1", "FP2", "FP3"):
        sess = bundle.sessions.get(s)
        if sess is None:
            continue
        laps = laps_dataframe(sess, s)
        if laps.empty:
            continue
        part = laps[["Driver", "Team", "lap_seconds"]].copy()
        part["lap_seconds"] = pd.to_numeric(part["lap_seconds"], errors="coerce")
        part = part.dropna(subset=["Driver", "Team", "lap_seconds"])
        if not part.empty:
            parts.append(part)

    if not parts:
        return pd.DataFrame(columns=["Driver", "Team", "practice_median_s", "practice_best_s", "team_practice_median_s"])

    all_laps = pd.concat(parts, ignore_index=True)
    drv = all_laps.groupby(["Driver", "Team"], as_index=False).agg(
        practice_median_s=("lap_seconds", "median"),
        practice_best_s=("lap_seconds", "min"),
    )
    team = all_laps.groupby("Team", as_index=False).agg(team_practice_median_s=("lap_seconds", "median"))
    return drv.merge(team, on="Team", how="left")


def _fill_missing_quali_positions(q_session: object, q_res: pd.DataFrame) -> pd.DataFrame:
    out = q_res.copy()
    out["quali_position"] = pd.to_numeric(out["quali_position"], errors="coerce")
    missing = out["quali_position"].isna()
    if not missing.any():
        return out

    base_pos = int(out["quali_position"].dropna().max()) if out["quali_position"].notna().any() else 0
    next_pos = base_pos + 1

    # First, recover missing entries from Q1 best laps (covers no-Q2/no-Q3 exits).
    try:
        q_parts = qualifying_parts_dataframes(q_session)
        q1 = q_parts.get("Q1")
        if q1 is not None and not q1.empty:
            q1b = (
                q1.groupby("Driver", as_index=False)["lap_seconds"]
                .min()
                .sort_values("lap_seconds")
            )
            miss_drivers = set(out.loc[missing, "Driver"].astype(str).tolist())
            q1b = q1b[q1b["Driver"].astype(str).isin(miss_drivers)]
            for drv in q1b["Driver"].astype(str).tolist():
                idx = out.index[(out["Driver"].astype(str) == drv) & (out["quali_position"].isna())]
                if len(idx) == 0:
                    continue
                out.loc[idx[0], "quali_position"] = float(next_pos)
                next_pos += 1
    except Exception:
        pass

    # Any remaining no-time/no-lap entries: keep official results ordering and append.
    for idx in out.index[out["quali_position"].isna()].tolist():
        out.loc[idx, "quali_position"] = float(next_pos)
        next_pos += 1
    return out


def _round_feature_frame(
    season: int,
    round_number: int,
    prior_season: int | None = None,
    bundle: object | None = None,
) -> pd.DataFrame:
    b = bundle if bundle is not None else load_sessions(season, round_number)
    feats = _practice_features(b)
    if feats.empty:
        raise RuntimeError("No usable practice lap features found for this round.")

    q = b.sessions.get("Qualifying")
    r = b.sessions.get("Race")
    if q is None or r is None:
        raise RuntimeError("Need Qualifying and Race results for target round output.")

    q_res = results_dataframe(q, "Qualifying")[["Driver", "Position"]].rename(columns={"Position": "quali_position"})
    q_res = _fill_missing_quali_positions(q, q_res)
    r_res = results_dataframe(r, "Race")[["Driver", "Position"]].rename(columns={"Position": "race_position"})
    out = feats.merge(q_res, on="Driver", how="left").merge(r_res, on="Driver", how="left")
    out["quali_position"] = pd.to_numeric(out["quali_position"], errors="coerce")
    out["race_position"] = pd.to_numeric(out["race_position"], errors="coerce")

    ps = prior_season if prior_season is not None else max(2018, season - 1)
    out = _attach_priors(out, ps, event_name=str(getattr(b, "event_name", "")))
    out["season"] = season
    out["round"] = round_number
    out["event_name"] = str(getattr(b, "event_name", ""))
    return out


def build_training_dataset(
    train_season: int,
    round_start: int,
    round_end: int,
    prior_season: int | None = None,
    prior_round_end: int = 24,
    rounds: tuple[int, ...] | None = None,
    prior_rounds: tuple[int, ...] | None = None,
) -> pd.DataFrame:
    # Intentionally unused in the lightweight predictor path.
    return pd.DataFrame()


def train_and_predict(
    target_round_df: pd.DataFrame,
    train_df: pd.DataFrame,
) -> ModelOutputs:
    target = target_round_df.copy()
    metrics: dict[str, Any] = {}
    event_name = str(target.get("event_name", pd.Series([""])).iloc[0]) if not target.empty else ""
    season_val = int(pd.to_numeric(target.get("season", pd.Series([np.nan])).iloc[0], errors="coerce")) if not target.empty else np.nan
    if pd.isna(season_val):
        overtake_difficulty = _overtake_difficulty(event_name)
        od_races_used = 0
    else:
        overtake_difficulty, od_races_used = _overtake_difficulty_last5_same_gp(int(season_val), event_name)

    # Qualifying prediction:
    # current weekend pace + previous season qualifying priors (driver + team),
    # with extra weight for same-circuit prior
    pace_rank = _rank_score(target["practice_median_s"])
    drv_q_rank = _rank_score(target["driver_prev_quali_avg"])
    team_q_rank = _rank_score(target["team_prev_quali_avg"])
    drv_q_c_rank = _rank_score(target.get("driver_prev_quali_circuit", pd.Series(np.nan, index=target.index)))
    team_q_c_rank = _rank_score(target.get("team_prev_quali_circuit", pd.Series(np.nan, index=target.index)))
    target["pred_quali_score"] = (
        0.40 * pace_rank
        + 0.15 * drv_q_rank
        + 0.25 * team_q_rank
        + 0.08 * drv_q_c_rank
        + 0.12 * team_q_c_rank
    )
    target["pred_quali_position"] = target["pred_quali_score"].rank(method="first").astype(int)

    # Race prediction:
    # weekend pace + qualifying position + previous season race priors (driver + team),
    # with extra weight for same-circuit prior
    quali_input = pd.to_numeric(target["quali_position"], errors="coerce")
    if quali_input.isna().any():
        tail_base = int(quali_input.dropna().max()) if quali_input.notna().any() else int(len(target))
        miss_order = pd.to_numeric(target.loc[quali_input.isna(), "pred_quali_position"], errors="coerce").rank(method="first")
        quali_input.loc[quali_input.isna()] = tail_base + miss_order.values
    quali_rank = _rank_score(quali_input)
    drv_r_rank = _rank_score(target["driver_prev_race_avg"])
    team_r_rank = _rank_score(target["team_prev_race_avg"])
    drv_r_c_rank = _rank_score(target.get("driver_prev_race_circuit", pd.Series(np.nan, index=target.index)))
    team_r_c_rank = _rank_score(target.get("team_prev_race_circuit", pd.Series(np.nan, index=target.index)))
    target["pred_race_score"] = (
        0.30 * pace_rank
        + 0.26 * quali_rank
        + 0.12 * drv_r_rank
        + 0.18 * team_r_rank
        + 0.06 * drv_r_c_rank
        + 0.08 * team_r_c_rank
    )
    # Circuit-aware recovery constraint:
    # back-grid starts are penalized more on overtaking-limited tracks.
    backgrid_depth = (quali_rank - 10.0).clip(lower=0.0)
    backgrid_penalty = overtake_difficulty * (backgrid_depth ** 1.15) * 0.08
    target["pred_race_score"] = target["pred_race_score"] + backgrid_penalty
    target["pred_race_position"] = target["pred_race_score"].rank(method="first").astype(int)

    q_pred = target[["Driver", "pred_quali_position", "quali_position"]].sort_values("pred_quali_position")
    r_pred = target[["Driver", "pred_race_position", "race_position", "quali_position"]].sort_values("pred_race_position")

    if q_pred["quali_position"].notna().any():
        metrics["quali_mae"] = float((q_pred["pred_quali_position"] - q_pred["quali_position"]).abs().mean())
    if r_pred["race_position"].notna().any():
        metrics["race_mae"] = float((r_pred["pred_race_position"] - r_pred["race_position"]).abs().mean())
    metrics["quali_model"] = "weighted_rank:practice+prev_year_quali(driver,team)+circuit_boost"
    metrics["race_model"] = "weighted_rank:practice+quali+prev_year_race(driver,team)+circuit_boost+overtake_penalty"
    metrics["overtake_difficulty"] = float(overtake_difficulty)
    metrics["overtake_difficulty_races_used"] = int(od_races_used)

    return ModelOutputs(
        quali_predictions=q_pred,
        race_predictions=r_pred,
        model_metrics=metrics,
    )
