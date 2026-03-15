from __future__ import annotations

from dataclasses import dataclass

import fastf1
import pandas as pd
from fastf1 import plotting
from fastf1.exceptions import DataNotLoadedError

from .settings import PATHS, SPRINT_SESSION_ORDER, STANDARD_SESSION_ORDER
from .utils import ensure_dirs


@dataclass
class SessionBundle:
    season: int
    round_number: int
    event_name: str
    event_format: str
    sessions: dict[str, object]


def init_fastf1_cache() -> None:
    ensure_dirs(PATHS.cache)
    fastf1.Cache.enable_cache(str(PATHS.cache))


def _canonical_session_name(label: str) -> str | None:
    mapping = {
        "Practice 1": "FP1",
        "Practice 2": "FP2",
        "Practice 3": "FP3",
        "Sprint Qualifying": "Sprint Qualifying",
        "Sprint Shootout": "Sprint Qualifying",
        "Sprint": "Sprint",
        "Qualifying": "Qualifying",
        "Race": "Race",
    }
    return mapping.get(str(label).strip())


def _session_order_for_event(season: int, round_number: int) -> tuple[str, str, list[str]]:
    try:
        event = fastf1.get_event(season, round_number)
        event_name = str(event.get("EventName", "")).strip()
        event_format = str(event.get("EventFormat", "")).strip().lower()
        session_names: list[str] = []
        for idx in range(1, 6):
            raw = event.get(f"Session{idx}")
            if raw is None:
                continue
            canonical = _canonical_session_name(str(raw))
            if canonical:
                session_names.append(canonical)
        if session_names:
            return event_name, event_format, session_names
        fallback = SPRINT_SESSION_ORDER if "sprint" in event_format else STANDARD_SESSION_ORDER
        return event_name, event_format, list(fallback)
    except Exception:
        return "", "", list(STANDARD_SESSION_ORDER)


def load_sessions(
    season: int, round_number: int, telemetry_sessions: set[str] | None = None
) -> SessionBundle:
    init_fastf1_cache()
    sessions: dict[str, object] = {}
    event_name, event_format, session_order = _session_order_for_event(season, round_number)
    telemetry_sessions = telemetry_sessions or set()

    for session_name in session_order:
        try:
            sess = fastf1.get_session(season, round_number, session_name)
            sess.load(
                laps=True,
                telemetry=session_name in telemetry_sessions,
                weather=True,
                messages=True,
            )
            sessions[session_name] = sess
            if not event_name:
                event_name = str(sess.event["EventName"])
        except Exception as exc:
            print(f"[WARN] Could not load {session_name}: {exc}")

    if not sessions:
        raise RuntimeError("No sessions loaded. Check season/round and internet connectivity.")

    return SessionBundle(
        season=season,
        round_number=round_number,
        event_name=event_name,
        event_format=event_format,
        sessions=sessions,
    )


def laps_dataframe(session_obj: object, session_name: str) -> pd.DataFrame:
    try:
        laps = session_obj.laps.copy()
    except (DataNotLoadedError, AttributeError):
        return pd.DataFrame()
    return laps_object_to_dataframe(laps, session_name)


def laps_object_to_dataframe(laps: pd.DataFrame, session_name: str) -> pd.DataFrame:
    if laps.empty:
        return pd.DataFrame()

    cols = [
        "Driver",
        "Team",
        "LapNumber",
        "Position",
        "LapTime",
        "SpeedI1",
        "SpeedI2",
        "SpeedFL",
        "SpeedST",
        "Compound",
        "Stint",
        "Sector1Time",
        "Sector2Time",
        "Sector3Time",
        "PitInTime",
        "PitOutTime",
        "TrackStatus",
        "IsAccurate",
    ]
    available = [c for c in cols if c in laps.columns]
    df = laps[available].copy()
    df["session"] = session_name

    if "LapTime" in df.columns:
        df["lap_seconds"] = df["LapTime"].dt.total_seconds()
    if "Sector1Time" in df.columns:
        df["s1_seconds"] = df["Sector1Time"].dt.total_seconds()
    if "Sector2Time" in df.columns:
        df["s2_seconds"] = df["Sector2Time"].dt.total_seconds()
    if "Sector3Time" in df.columns:
        df["s3_seconds"] = df["Sector3Time"].dt.total_seconds()

    if "Driver" in df.columns:
        df = df[df["Driver"].notna()]
    if "lap_seconds" in df.columns:
        df = df[df["lap_seconds"].notna()]

    return df


def qualifying_parts_dataframes(session_obj: object) -> dict[str, pd.DataFrame]:
    out: dict[str, pd.DataFrame] = {}
    try:
        parts = session_obj.laps.split_qualifying_sessions()
    except (DataNotLoadedError, AttributeError, Exception):
        return out

    labels = ["Q1", "Q2", "Q3"]
    for label, part in zip(labels, parts):
        if part is None or len(part) == 0:
            continue
        out[label] = laps_object_to_dataframe(part.copy(), label)
    return out


def results_dataframe(session_obj: object, session_name: str) -> pd.DataFrame:
    candidate_cols = [
        "Abbreviation",
        "FullName",
        "TeamName",
        "Position",
        "GridPosition",
        "Status",
        "Points",
    ]
    empty_out = pd.DataFrame(columns=["Driver", "FullName", "Team", "Position", "GridPosition", "Status", "Points", "session"])
    try:
        res = session_obj.results.copy()
    except (DataNotLoadedError, AttributeError):
        return empty_out
    if res.empty:
        return empty_out

    available = [c for c in candidate_cols if c in res.columns]
    if not available:
        return empty_out
    out = res[available].copy()
    out["session"] = session_name
    out = out.rename(columns={"Abbreviation": "Driver", "TeamName": "Team"})

    return out


def driver_color_map(session_obj: object, drivers: list[str]) -> dict[str, str]:
    colors: dict[str, str] = {}
    for driver in drivers:
        try:
            colors[driver] = plotting.get_driver_color(driver, session_obj)
        except Exception:
            colors[driver] = "#4ea1ff"
    return colors


def team_color_map(session_obj: object, teams: list[str]) -> dict[str, str]:
    colors: dict[str, str] = {}
    for team in teams:
        try:
            colors[team] = plotting.get_team_color(team, session_obj)
        except Exception:
            colors[team] = "#4ea1ff"
    return colors
