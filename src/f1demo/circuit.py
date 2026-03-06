from __future__ import annotations

from typing import Any

import fastf1
import numpy as np
import pandas as pd

from .data import init_fastf1_cache, laps_dataframe, results_dataframe


def _format_laptime(seconds: float | int | None) -> str:
    if seconds is None or (isinstance(seconds, float) and pd.isna(seconds)):
        return ""
    total = float(seconds)
    mins = int(total // 60)
    secs = total - mins * 60
    return f"{mins}:{secs:06.3f}"


def _has_status_code(status: object, codes: set[str]) -> bool:
    s = str(status) if status is not None else ""
    return any(c in s for c in codes)


def _event_track_notes(event_name: str) -> dict[str, list[str]]:
    name = (event_name or "").strip().lower()
    # Same-GP specific notes where available, with safe generic fallback.
    if "australian" in name or "albert park" in name:
        return {
            "key_overtake_zones": [
                "T3 braking zone after the DRS straight is the primary passing opportunity.",
                "T11 can create late-braking moves when exit from the previous sector is strong.",
                "T1 first-lap compression frequently opens opportunistic side-by-side entries.",
            ],
            "ideal_driver_characteristics": [
                "Strong confidence in high-speed direction changes to maximize final-sector commitment.",
                "Precise trail-braking into medium-speed corners to protect front tyres over stints.",
                "Consistent traction management on corner exits to defend and attack effectively under DRS.",
            ],
            "ideal_car_characteristics": [
                "Balanced low-drag efficiency for long straights without giving up mid-speed platform stability.",
                "Strong front-end response for quick rotation through linked corner sequences.",
                "Stable kerb behavior and rear traction to protect tyre life across long runs.",
            ],
            "track_evolution_notes": [
                "Track evolution is usually significant across the weekend as rubber builds on a semi-street surface.",
                "Early sessions can be grip-limited and noisy, so FP long-run trends are often more reliable than single-lap headlines.",
                "Late-session pace spikes in qualifying are common as temperature and grip windows align.",
            ],
        }
    return {
        "key_overtake_zones": [
            "Primary overtakes typically occur at the longest DRS-assisted braking zone.",
            "Secondary passing chances come from traction-sensitive exits leading into medium-length straights.",
            "Lap-one and restart phases usually create the highest transient overtake probability.",
        ],
        "ideal_driver_characteristics": [
            "High confidence under braking and consistency in tyre-limited phases.",
            "Strong corner-exit management to convert pace into defendable straight-line performance.",
            "Low-variance lap execution to preserve strategy flexibility.",
        ],
        "ideal_car_characteristics": [
            "Balanced aero efficiency: enough top speed for attack with stable platform in direction changes.",
            "Good traction and tyre thermal control over medium and long stints.",
            "Predictable balance under fuel and compound transitions.",
        ],
        "track_evolution_notes": [
            "Grip generally builds through the weekend, increasing the value of late-session runs.",
            "Session-to-session temperature shifts can reorder competitive balance more than headline averages suggest.",
            "Race stint behavior should be weighted more heavily than isolated qualifying peaks for strategy calls.",
        ],
    }


def _turn_count_from_session(session_obj: object) -> int | None:
    if session_obj is None:
        return None
    try:
        info = session_obj.get_circuit_info()
        if info is None or getattr(info, "corners", None) is None:
            return None
        corners = info.corners
        if corners.empty or "Number" not in corners.columns:
            return None
        return int(pd.to_numeric(corners["Number"], errors="coerce").dropna().nunique())
    except Exception:
        return None


def _turn_count_from_bundle(bundle: object) -> int | None:
    # Prefer qualifying; fallback to practice/race if qualifying not available yet.
    for key in ["Qualifying", "FP2", "FP1", "FP3", "Race"]:
        turns = _turn_count_from_session(bundle.sessions.get(key))
        if turns is not None:
            return turns
    return None


def _gear_changes_from_session(session_obj: object) -> int | None:
    if session_obj is None:
        return None
    try:
        fastest = session_obj.laps.pick_fastest()
        if fastest is None:
            return None
        tel = fastest.get_car_data()
        if tel is None or tel.empty or "nGear" not in tel.columns:
            return None
        g = pd.to_numeric(tel["nGear"], errors="coerce").dropna().astype(int)
        if g.empty:
            return None
        return int((g.diff().fillna(0) != 0).sum())
    except Exception:
        return None


def _gear_changes_from_bundle(bundle: object) -> int | None:
    # Prefer qualifying telemetry; fallback to FP2/FP1 if qualifying isn't available yet.
    for key in ["Qualifying", "FP2", "FP1", "FP3", "Race"]:
        gear = _gear_changes_from_session(bundle.sessions.get(key))
        if gear is not None:
            return gear
    return None


def build_circuit_overview(bundle: object, history_years: int = 5) -> dict[str, Any]:
    init_fastf1_cache()
    season = int(bundle.season)
    round_number = int(bundle.round_number)
    event_name = str(getattr(bundle, "event_name", "")).strip()

    winners: list[dict[str, Any]] = []
    fastest_records: list[dict[str, Any]] = []
    historical_pole_records: list[dict[str, Any]] = []
    pit_stops: list[float] = []
    pit_loss_green: list[float] = []
    pit_loss_sc: list[float] = []
    pit_loss_vsc: list[float] = []
    pit_durations_s: list[float] = []
    car_adj_rows: list[dict[str, Any]] = []
    sc_races = 0
    vsc_races = 0
    race_count = 0

    max_lookback_years = max(history_years + 8, 12)
    for yr in range(season - 1, season - max_lookback_years - 1, -1):
        if race_count >= history_years:
            break
        hist_round = round_number
        try:
            if event_name:
                sched = fastf1.get_event_schedule(yr, include_testing=False)
                if sched is None or sched.empty or "EventName" not in sched.columns or "RoundNumber" not in sched.columns:
                    continue
                name_norm = sched["EventName"].astype(str).str.strip().str.lower()
                match = sched[name_norm == event_name.lower()]
                if match.empty:
                    continue
                match_round = pd.to_numeric(match.iloc[0]["RoundNumber"], errors="coerce")
                if pd.isna(match_round):
                    continue
                hist_round = int(match_round)
                sess = fastf1.get_session(yr, hist_round, "Race")
            else:
                sess = fastf1.get_session(yr, round_number, "Race")
            sess.load(laps=True, telemetry=False, weather=False, messages=False)
        except Exception:
            continue

        res = results_dataframe(sess, "Race")
        laps = laps_dataframe(sess, "Race")
        if res.empty or laps.empty:
            continue
        race_count += 1

        # Winner history
        winner = res.copy()
        winner["Position"] = pd.to_numeric(winner["Position"], errors="coerce")
        winner = winner.dropna(subset=["Position"]).sort_values("Position")
        if not winner.empty:
            w = winner.iloc[0]
            winners.append({"season": yr, "driver": str(w.get("Driver", "")), "team": str(w.get("Team", ""))})

        # Fastest race lap record
        laps_num = laps.copy()
        laps_num["lap_seconds"] = pd.to_numeric(laps_num["lap_seconds"], errors="coerce")
        laps_num = laps_num.dropna(subset=["lap_seconds"])
        if not laps_num.empty:
            i = laps_num["lap_seconds"].idxmin()
            fr = laps_num.loc[i]
            fastest_records.append(
                {
                    "season": yr,
                    "driver": str(fr.get("Driver", "")),
                    "lap_seconds": float(fr.get("lap_seconds", np.nan)),
                }
            )

        # Avg pit stops
        laps_num["Stint"] = pd.to_numeric(laps_num.get("Stint"), errors="coerce")
        valid = laps_num.dropna(subset=["Driver", "Stint"])
        if not valid.empty:
            c = valid.groupby("Driver", as_index=False)["Stint"].nunique()
            vals = (c["Stint"] - 1).clip(lower=0).astype(float).tolist()
            pit_stops.extend(vals)

        # Average pit-lane time proxy from pit-in to next pit-out (same driver).
        for _, g in laps_num.groupby("Driver"):
            in_times = pd.to_timedelta(g.get("PitInTime"), errors="coerce").dropna().sort_values().tolist()
            out_times = pd.to_timedelta(g.get("PitOutTime"), errors="coerce").dropna().sort_values().tolist()
            if not in_times or not out_times:
                continue
            j = 0
            for tin in in_times:
                while j < len(out_times) and out_times[j] <= tin:
                    j += 1
                if j >= len(out_times):
                    break
                dt = (out_times[j] - tin).total_seconds()
                if 10.0 <= dt <= 80.0:
                    pit_durations_s.append(float(dt))
                j += 1

        # Car-adjusted driver score: position delta versus team average position in the race.
        rr = res.copy()
        rr["Position"] = pd.to_numeric(rr["Position"], errors="coerce")
        rr = rr.dropna(subset=["Driver", "Team", "Position"])
        if not rr.empty:
            team_counts = rr.groupby("Team")["Driver"].count()
            valid_teams = team_counts[team_counts >= 2].index.tolist()
            rr = rr[rr["Team"].isin(valid_teams)]
            if not rr.empty:
                team_mean = rr.groupby("Team", as_index=False).agg(team_pos_avg=("Position", "mean"))
                rr = rr.merge(team_mean, on="Team", how="left")
                rr["car_adjusted_delta"] = rr["team_pos_avg"] - rr["Position"]
                car_adj_rows.extend(
                    rr[["Driver", "car_adjusted_delta"]].to_dict(orient="records")
                )

        # SC/VSC occurrence
        st = laps_num.get("TrackStatus")
        if st is not None:
            track = st.astype(str)
            has_sc = track.apply(lambda x: _has_status_code(x, {"4", "5"})).any()
            has_vsc = track.apply(lambda x: _has_status_code(x, {"6", "7"})).any()
            sc_races += int(has_sc)
            vsc_races += int(has_vsc)

        # Pit time saved proxy under SC/VSC vs green
        laps_num["is_pit_lap"] = laps_num.get("PitInTime").notna() | laps_num.get("PitOutTime").notna()
        for _, g in laps_num.groupby("Driver"):
            g = g.dropna(subset=["lap_seconds"])
            if g.empty:
                continue
            clean = g[
                (~g["is_pit_lap"])
                & (~g["TrackStatus"].astype(str).apply(lambda x: _has_status_code(x, {"4", "5", "6", "7"})))
            ]["lap_seconds"]
            if clean.empty:
                continue
            clean_ref = float(clean.median())
            stops = g[g["is_pit_lap"]].copy()
            if stops.empty:
                continue
            for _, r in stops.iterrows():
                loss = float(r["lap_seconds"] - clean_ref)
                if loss < 0.0 or loss > 80.0:
                    continue
                status = str(r.get("TrackStatus", ""))
                if _has_status_code(status, {"4", "5"}):
                    pit_loss_sc.append(loss)
                elif _has_status_code(status, {"6", "7"}):
                    pit_loss_vsc.append(loss)
                else:
                    pit_loss_green.append(loss)

        # Historical qualifying pole reference for comparable fastest-lap basis.
        try:
            qsess = fastf1.get_session(yr, int(hist_round), "Qualifying")
            qsess.load(laps=True, telemetry=False, weather=False, messages=False)
            qlaps = laps_dataframe(qsess, "Qualifying").copy()
            if not qlaps.empty:
                qlaps["lap_seconds"] = pd.to_numeric(qlaps["lap_seconds"], errors="coerce")
                qlaps = qlaps.dropna(subset=["lap_seconds"])
                if not qlaps.empty:
                    qi = qlaps["lap_seconds"].idxmin()
                    qrow = qlaps.loc[qi]
                    historical_pole_records.append(
                        {
                            "season": yr,
                            "driver": str(qrow.get("Driver", "")),
                            "lap_seconds": float(qrow.get("lap_seconds", np.nan)),
                        }
                    )
        except Exception:
            continue

    sc_save = None
    if pit_loss_green and pit_loss_sc:
        sc_save = round(float(np.median(pit_loss_green) - np.median(pit_loss_sc)), 2)
    vsc_save = None
    if pit_loss_green and pit_loss_vsc:
        vsc_save = round(float(np.median(pit_loss_green) - np.median(pit_loss_vsc)), 2)

    hist_fast = None
    if historical_pole_records:
        best = min(historical_pole_records, key=lambda r: r["lap_seconds"])
        hist_fast = f"{_format_laptime(best['lap_seconds'])} ({best['driver']} - {best['season']})"
    elif fastest_records:
        best = min(fastest_records, key=lambda r: r["lap_seconds"])
        hist_fast = f"{_format_laptime(best['lap_seconds'])} ({best['driver']} - {best['season']})"

    # Current weekend fastest lap: prefer qualifying pole; fallback to FP2/FP1/FP3 then race.
    weekend_fast = None
    for sname in ["Qualifying", "FP2", "FP1", "FP3", "Race"]:
        sess = bundle.sessions.get(sname)
        if sess is None:
            continue
        slaps = laps_dataframe(sess, sname).copy()
        if slaps.empty:
            continue
        slaps["lap_seconds"] = pd.to_numeric(slaps["lap_seconds"], errors="coerce")
        slaps = slaps.dropna(subset=["lap_seconds"])
        if slaps.empty:
            continue
        j = slaps["lap_seconds"].idxmin()
        row = slaps.loc[j]
        weekend_fast = f"{_format_laptime(row['lap_seconds'])} ({row.get('Driver', '')} - {sname})"
        break

    turns_val = _turn_count_from_bundle(bundle)
    gear_val = _gear_changes_from_bundle(bundle)

    # Backfill with historical same-GP qualifying if current weekend sessions are incomplete.
    if turns_val is None or gear_val is None:
        for yr in range(season - 1, season - max_lookback_years - 1, -1):
            try:
                sched = fastf1.get_event_schedule(yr, include_testing=False)
                if sched is None or sched.empty or "EventName" not in sched.columns or "RoundNumber" not in sched.columns:
                    continue
                name_norm = sched["EventName"].astype(str).str.strip().str.lower()
                match = sched[name_norm == event_name.lower()]
                if match.empty:
                    continue
                hist_round = pd.to_numeric(match.iloc[0]["RoundNumber"], errors="coerce")
                if pd.isna(hist_round):
                    continue
                qs = fastf1.get_session(yr, int(hist_round), "Qualifying")
                qs.load(laps=True, telemetry=True, weather=False, messages=False)
            except Exception:
                continue
            if turns_val is None:
                turns_val = _turn_count_from_session(qs)
            if gear_val is None:
                gear_val = _gear_changes_from_session(qs)
            if turns_val is not None and gear_val is not None:
                break

    combined_sc_vsc = 0.0
    if race_count > 0:
        combined_sc_vsc = round(100.0 * ((sc_races + vsc_races - min(sc_races, vsc_races)) / race_count), 1)

    best_driver_adjusted = None
    if car_adj_rows:
        cadf = pd.DataFrame(car_adj_rows)
        cadf["car_adjusted_delta"] = pd.to_numeric(cadf["car_adjusted_delta"], errors="coerce")
        cadf = cadf.dropna(subset=["Driver", "car_adjusted_delta"])
        if not cadf.empty:
            agg = cadf.groupby("Driver", as_index=False).agg(
                car_adjusted_score=("car_adjusted_delta", "mean"),
                races=("car_adjusted_delta", "count"),
            )
            agg = agg.sort_values(["car_adjusted_score", "races"], ascending=[False, False])
            top = agg.iloc[0]
            best_driver_adjusted = {
                "driver": str(top["Driver"]),
                "score": round(float(top["car_adjusted_score"]), 2),
                "races": int(top["races"]),
            }

    notes = _event_track_notes(event_name)

    return {
        "turns": turns_val,
        "gear_changes_per_lap": gear_val,
        "weekend_fastest_lap": weekend_fast,
        "historical_fastest_lap": hist_fast,
        "avg_pit_stops": round(float(np.mean(pit_stops)), 2) if pit_stops else None,
        "sc_likelihood_pct": round(100.0 * sc_races / race_count, 1) if race_count else None,
        "vsc_likelihood_pct": round(100.0 * vsc_races / race_count, 1) if race_count else None,
        "combined_neutralization_likelihood_pct": combined_sc_vsc if race_count else None,
        "sc_pit_time_saved_s": sc_save,
        "vsc_pit_time_saved_s": vsc_save,
        "avg_pitstop_time_s": round(float(np.mean(pit_durations_s)), 2) if pit_durations_s else None,
        "best_driver_adjusted": best_driver_adjusted,
        "previous_winners": winners[:5],
        "history_races_used": race_count,
        "key_overtake_zones": notes["key_overtake_zones"],
        "ideal_driver_characteristics": notes["ideal_driver_characteristics"],
        "ideal_car_characteristics": notes["ideal_car_characteristics"],
        "track_evolution_notes": notes["track_evolution_notes"],
    }
