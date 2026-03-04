from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from typing import Any

import fastf1
import numpy as np
import pandas as pd

from .data import init_fastf1_cache, laps_dataframe, load_sessions, results_dataframe


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
    r_res = results_dataframe(r, "Race")[["Driver", "Position"]].rename(columns={"Position": "race_position"})
    out = feats.merge(q_res, on="Driver", how="left").merge(r_res, on="Driver", how="left")
    out["quali_position"] = pd.to_numeric(out["quali_position"], errors="coerce")
    out["race_position"] = pd.to_numeric(out["race_position"], errors="coerce")

    ps = prior_season if prior_season is not None else max(2018, season - 1)
    out = _attach_priors(out, ps, event_name=str(getattr(b, "event_name", "")))
    out["season"] = season
    out["round"] = round_number
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
    quali_input = pd.to_numeric(target["quali_position"], errors="coerce").fillna(target["pred_quali_position"])
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
    target["pred_race_position"] = target["pred_race_score"].rank(method="first").astype(int)

    q_pred = target[["Driver", "pred_quali_position", "quali_position"]].sort_values("pred_quali_position")
    r_pred = target[["Driver", "pred_race_position", "race_position", "quali_position"]].sort_values("pred_race_position")

    if q_pred["quali_position"].notna().any():
        metrics["quali_mae"] = float((q_pred["pred_quali_position"] - q_pred["quali_position"]).abs().mean())
    if r_pred["race_position"].notna().any():
        metrics["race_mae"] = float((r_pred["pred_race_position"] - r_pred["race_position"]).abs().mean())
    metrics["quali_model"] = "weighted_rank:practice+prev_year_quali(driver,team)+circuit_boost"
    metrics["race_model"] = "weighted_rank:practice+quali+prev_year_race(driver,team)+circuit_boost"

    return ModelOutputs(
        quali_predictions=q_pred,
        race_predictions=r_pred,
        model_metrics=metrics,
    )
