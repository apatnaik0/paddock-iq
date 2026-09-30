from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import fastf1
import pandas as pd

from .data import event_session_order, init_fastf1_cache, laps_dataframe, results_dataframe
from .pipeline import run_pipeline
from .settings import PATHS
from .site import update_race_manifest
from .utils import ensure_dirs


@dataclass(frozen=True)
class RaceTarget:
    season: int
    round_number: int
    event_name: str
    event_format: str
    session_order: list[str]
    window_start_utc: datetime | None
    window_end_utc: datetime | None


def _state_path() -> Path:
    return PATHS.site / "races" / "state.json"


def _round_slug(season: int, round_number: int) -> str:
    return f"{season}_round_{round_number:02d}"


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _to_utc(value: Any) -> datetime | None:
    if value is None or pd.isna(value):
        return None
    try:
        ts = pd.Timestamp(value)
    except Exception:
        return None
    if ts.tzinfo is None:
        ts = ts.tz_localize(timezone.utc)
    else:
        ts = ts.tz_convert(timezone.utc)
    return ts.to_pydatetime()


def load_state() -> dict[str, Any]:
    p = _state_path()
    if not p.exists():
        return {"races": {}}
    try:
        payload = json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return {"races": {}}
    if not isinstance(payload, dict):
        return {"races": {}}
    payload.setdefault("races", {})
    return payload


def save_state(payload: dict[str, Any]) -> None:
    ensure_dirs(_state_path().parent)
    _state_path().write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _event_window(row: pd.Series) -> tuple[datetime | None, datetime | None]:
    dates: list[datetime] = []
    for idx in range(1, 6):
        for col in (f"Session{idx}DateUtc", f"Session{idx}Date"):
            dt = _to_utc(row.get(col))
            if dt is not None:
                dates.append(dt)
                break
    for col in ("EventDate", "EventDateUtc"):
        dt = _to_utc(row.get(col))
        if dt is not None:
            dates.append(dt)
    if not dates:
        return None, None
    return min(dates), max(dates) + timedelta(hours=8)


def find_race_target(
    season: int,
    *,
    round_number: int | None = None,
    now_utc: datetime | None = None,
    lookahead_days: int = 7,
) -> RaceTarget:
    init_fastf1_cache()
    if round_number is not None:
        event_name, event_format, session_order = event_session_order(season, round_number)
        event = fastf1.get_event(season, round_number)
        start, end = _event_window(pd.Series(event))
        return RaceTarget(season, round_number, event_name, event_format, session_order, start, end)

    now_utc = now_utc or _utc_now()
    schedule = fastf1.get_event_schedule(season, include_testing=False)
    schedule = schedule[pd.to_numeric(schedule.get("RoundNumber"), errors="coerce").fillna(0).astype(int) > 0].copy()
    candidates: list[tuple[int, pd.Series, datetime | None, datetime | None]] = []
    for _, row in schedule.iterrows():
        rnd = int(row.get("RoundNumber"))
        start, end = _event_window(row)
        candidates.append((rnd, row, start, end))

    active: list[tuple[int, pd.Series, datetime | None, datetime | None]] = []
    for item in candidates:
        _, _, start, end = item
        if start is None or end is None:
            continue
        if start - timedelta(days=1) <= now_utc <= end + timedelta(days=1):
            active.append(item)
    if active:
        rnd, row, start, end = sorted(active, key=lambda x: x[0])[0]
    else:
        upcoming = [
            item
            for item in candidates
            if item[2] is not None and item[2] <= now_utc + timedelta(days=lookahead_days)
            and item[3] is not None and item[3] >= now_utc - timedelta(days=1)
        ]
        if upcoming:
            rnd, row, start, end = sorted(upcoming, key=lambda x: x[2] or datetime.max.replace(tzinfo=timezone.utc))[0]
        else:
            future = [item for item in candidates if item[2] is not None and item[2] > now_utc]
            if future:
                rnd, row, start, end = sorted(future, key=lambda x: x[2] or datetime.max.replace(tzinfo=timezone.utc))[0]
            else:
                rnd, row, start, end = sorted(candidates, key=lambda x: x[0])[-1]

    event_name, event_format, session_order = event_session_order(season, rnd)
    if not event_name:
        event_name = str(row.get("EventName", "")).strip()
    return RaceTarget(season, rnd, event_name, event_format, session_order, start, end)


def _session_is_available(
    season: int,
    round_number: int,
    session_name: str,
    *,
    min_laps: int,
    min_drivers: int,
) -> tuple[bool, dict[str, Any]]:
    try:
        sess = fastf1.get_session(season, round_number, session_name)
        sess.load(laps=True, telemetry=False, weather=False, messages=False)
    except Exception as exc:
        return False, {"session": session_name, "available": False, "reason": str(exc)}
    laps = laps_dataframe(sess, session_name)
    drivers = int(laps["Driver"].nunique()) if not laps.empty and "Driver" in laps.columns else 0
    lap_count = int(laps.shape[0])
    available = lap_count >= min_laps and drivers >= min_drivers
    details: dict[str, Any] = {
        "session": session_name,
        "available": available,
        "laps": lap_count,
        "drivers": drivers,
    }
    if session_name == "Race" and available:
        results = results_dataframe(sess, session_name)
        classified = 0
        if not results.empty and "Position" in results.columns:
            classified = int(pd.to_numeric(results["Position"], errors="coerce").notna().sum())
        details["classified_results"] = classified
        details["race_complete"] = classified >= min_drivers
    return available, details


def available_sessions(
    target: RaceTarget,
    *,
    min_laps: int,
    min_drivers: int,
) -> tuple[list[str], list[dict[str, Any]]]:
    available: list[str] = []
    diagnostics: list[dict[str, Any]] = []
    for session_name in target.session_order:
        ok, detail = _session_is_available(
            target.season,
            target.round_number,
            session_name,
            min_laps=min_laps,
            min_drivers=min_drivers,
        )
        diagnostics.append(detail)
        if ok:
            available.append(session_name)
    return available, diagnostics


def _race_state(state: dict[str, Any], slug: str) -> dict[str, Any]:
    races = state.setdefault("races", {})
    race = races.setdefault(slug, {})
    race.setdefault("processed_sessions", [])
    return race


def _is_race_complete(diagnostics: list[dict[str, Any]]) -> bool:
    return any(str(d.get("session")) == "Race" and bool(d.get("race_complete")) for d in diagnostics)


def auto_update(
    *,
    season: int,
    round_number: int | None,
    train_round_end: int,
    quick: bool,
    force: bool,
    dry_run: bool,
    min_laps: int,
    min_drivers: int,
    lookahead_days: int,
    ga4_measurement_id: str,
    complete_after_race: bool,
) -> int:
    target = find_race_target(season, round_number=round_number, lookahead_days=lookahead_days)
    slug = _round_slug(target.season, target.round_number)
    available, diagnostics = available_sessions(target, min_laps=min_laps, min_drivers=min_drivers)
    state = load_state()
    race_state = _race_state(state, slug)
    processed = set(str(s) for s in race_state.get("processed_sessions", []))
    new_sessions = [s for s in available if s not in processed]
    race_complete = _is_race_complete(diagnostics)
    should_run = force or bool(new_sessions) or not (PATHS.site / "races" / slug / "index.html").exists()

    print(f"[AUTO] Target: {target.season} round {target.round_number} | {target.event_name}")
    print(f"[AUTO] Expected sessions: {', '.join(target.session_order)}")
    print(f"[AUTO] Available sessions: {', '.join(available) if available else 'none'}")
    print(f"[AUTO] New sessions: {', '.join(new_sessions) if new_sessions else 'none'}")
    if not should_run:
        state["last_checked_utc"] = _utc_now().isoformat(timespec="seconds")
        race_state["last_checked_utc"] = state["last_checked_utc"]
        race_state["availability"] = diagnostics
        if not dry_run:
            save_state(state)
        print("[AUTO] No new data detected. Exiting without regenerating the site.")
        return 0

    status = "completed" if race_complete and complete_after_race else "current"
    if dry_run:
        print("[AUTO] Dry run selected. The site, manifest, and state file were not modified.")
        return 0

    update_race_manifest(
        season=target.season,
        round_number=target.round_number,
        event_name=target.event_name,
        is_current=status == "current",
        status=status,
    )

    run_pipeline(
        season=target.season,
        round_number=target.round_number,
        train_round_end=train_round_end,
        quick=quick,
        ga4_measurement_id=ga4_measurement_id,
    )

    # Pipeline rendering preserves the manifest row state that was set above.
    now = _utc_now().isoformat(timespec="seconds")
    race_state.update(
        {
            "season": target.season,
            "round_number": target.round_number,
            "event_name": target.event_name,
            "event_format": target.event_format,
            "status": status,
            "is_current": status == "current",
            "processed_sessions": available,
            "last_successful_update_utc": now,
            "last_checked_utc": now,
            "availability": diagnostics,
        }
    )
    state["current_round"] = slug if status == "current" else None
    state["last_checked_utc"] = now
    state["last_successful_update_utc"] = now
    save_state(state)
    update_race_manifest(
        season=target.season,
        round_number=target.round_number,
        event_name=target.event_name,
        is_current=status == "current",
        status=status,
    )
    print(f"[AUTO] Updated {slug}. Processed sessions: {', '.join(available) if available else 'none'}")
    return 0


def main() -> None:
    parser = argparse.ArgumentParser(description="Automatically update the current F1 dashboard race.")
    parser.add_argument("--season", type=int, default=_utc_now().year)
    parser.add_argument("--round", dest="round_number", type=int, default=None)
    parser.add_argument("--train-round-end", type=int, default=2)
    parser.add_argument("--quick", action="store_true", help="Use the pipeline's faster historical window.")
    parser.add_argument("--force", action="store_true", help="Regenerate even if no new sessions are detected.")
    parser.add_argument("--dry-run", action="store_true", help="Detect target/session state without running the pipeline.")
    parser.add_argument("--min-laps", type=int, default=10)
    parser.add_argument("--min-drivers", type=int, default=5)
    parser.add_argument("--lookahead-days", type=int, default=7)
    parser.add_argument(
        "--ga4-measurement-id",
        type=str,
        default=os.getenv("PADDOCK_GA4_MEASUREMENT_ID", ""),
    )
    parser.add_argument(
        "--no-complete-after-race",
        action="store_true",
        help="Keep the race marked current even after classified race data is available.",
    )
    args = parser.parse_args()
    raise SystemExit(
        auto_update(
            season=args.season,
            round_number=args.round_number,
            train_round_end=args.train_round_end,
            quick=args.quick,
            force=args.force,
            dry_run=args.dry_run,
            min_laps=args.min_laps,
            min_drivers=args.min_drivers,
            lookahead_days=args.lookahead_days,
            ga4_measurement_id=args.ga4_measurement_id,
            complete_after_race=not args.no_complete_after_race,
        )
    )


if __name__ == "__main__":
    main()
