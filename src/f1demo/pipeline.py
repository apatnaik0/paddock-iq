from __future__ import annotations

import argparse
import os
import time
from pathlib import Path

import fastf1
import pandas as pd

from .analysis import (
    build_session_charts,
    export_session_tables,
    plot_quali_fastest_two_lap_delta,
    session_summary,
    stint_summary,
    teammate_delta,
)
from .circuit import build_circuit_overview
from .data import (
    driver_color_map,
    laps_dataframe,
    load_sessions,
    qualifying_parts_dataframes,
    results_dataframe,
    team_color_map,
)
from .modeling import _round_feature_frame, train_and_predict
from .settings import PATHS
from .site import copy_plot_assets, render_site
from .strategy import build_strategy_overview
from .utils import ensure_dirs, write_json


def _chart_highlights(chart_title: str, session_name: str, top5: list[dict], laps_count: int) -> list[str]:
    chart_title_l = chart_title.lower()
    rows = [r for r in top5 if isinstance(r, dict)]

    def _num(v: object) -> float | None:
        if v is None:
            return None
        try:
            f = float(v)
            if pd.isna(f):
                return None
            return f
        except (TypeError, ValueError):
            return None

    def _rank_value(r: dict) -> float:
        for k in ("session_position", "race_position"):
            n = _num(r.get(k))
            if n is not None:
                return n
        return float("inf")

    ranked = sorted(rows, key=_rank_value)
    leader = ranked[0] if ranked else {}
    runner = ranked[1] if len(ranked) > 1 else {}
    third = ranked[2] if len(ranked) > 2 else {}
    fifth = ranked[4] if len(ranked) > 4 else {}
    tenth = ranked[9] if len(ranked) > 9 else (ranked[-1] if ranked else {})
    leader_name = str(leader.get("Driver", "N/A"))
    leader_team = str(leader.get("Team", ""))
    runner_name = str(runner.get("Driver", "N/A")) if runner else "N/A"
    runner_team = str(runner.get("Team", "")) if runner else ""
    third_name = str(third.get("Driver", "N/A")) if third else "N/A"

    pace_candidates = ("top5_avg_pace_s", "best_lap_time_s", "median_pace_s", "best_lap_s", "median_lap_s")
    pace_key = next((k for k in pace_candidates if any(_num(r.get(k)) is not None for r in ranked)), None)
    lead_pace = _num(leader.get(pace_key)) if pace_key else None
    run_pace = _num(runner.get(pace_key)) if pace_key else None
    third_pace = _num(third.get(pace_key)) if pace_key else None
    fifth_pace = _num(fifth.get(pace_key)) if pace_key else None
    tenth_pace = _num(tenth.get(pace_key)) if pace_key else None
    pace_gap = (run_pace - lead_pace) if (run_pace is not None and lead_pace is not None) else None
    p3_gap = (third_pace - lead_pace) if (third_pace is not None and lead_pace is not None) else None
    p5_gap = (fifth_pace - lead_pace) if (fifth_pace is not None and lead_pace is not None) else None
    p10_gap = (tenth_pace - lead_pace) if (tenth_pace is not None and lead_pace is not None) else None
    lead_pace_fmt = _format_laptime(lead_pace) if lead_pace is not None else "N/A"
    gap_fmt = f"{pace_gap:.3f}s" if pace_gap is not None else "N/A"
    p3_gap_fmt = f"{p3_gap:.3f}s" if p3_gap is not None else "N/A"
    p5_gap_fmt = f"{p5_gap:.3f}s" if p5_gap is not None else "N/A"
    p10_gap_fmt = f"{p10_gap:.3f}s" if p10_gap is not None else "N/A"

    if "fastest two lap delta" in chart_title_l:
        return [
            f"{session_name} reference lap is {leader_name}{f' ({leader_team})' if leader_team else ''} at {lead_pace_fmt}, with {runner_name}{f' ({runner_team})' if runner_team else ''} next at +{gap_fmt}.",
            "Long same-direction delta segments indicate where one lap held a sustained gain rather than isolated corner-only spikes.",
            f"P1-to-P3 separation is +{p3_gap_fmt}, so even small mid-lap gains can materially change the front order in this split.",
        ]

    driver_team = {str(r.get("Driver", "")): str(r.get("Team", "")) for r in ranked if r.get("Driver")}
    s1 = [(_num(r.get("s1_time_s")), str(r.get("Driver", ""))) for r in ranked if _num(r.get("s1_time_s")) is not None]
    s2 = [(_num(r.get("s2_time_s")), str(r.get("Driver", ""))) for r in ranked if _num(r.get("s2_time_s")) is not None]
    s3 = [(_num(r.get("s3_time_s")), str(r.get("Driver", ""))) for r in ranked if _num(r.get("s3_time_s")) is not None]
    has_sector_triplet = bool(s1 and s2 and s3)
    if has_sector_triplet:
        s1_best = min(s1, key=lambda x: x[0])
        s2_best = min(s2, key=lambda x: x[0])
        s3_best = min(s3, key=lambda x: x[0])
        unique_sector_leaders = len({s1_best[1], s2_best[1], s3_best[1]})
        s1_team = driver_team.get(s1_best[1], "")
        s2_team = driver_team.get(s2_best[1], "")
        s3_team = driver_team.get(s3_best[1], "")

    if "sector delta comparison" in chart_title_l and has_sector_triplet:
        return [
            f"S1/S2/S3 benchmarks are {s1_best[1]}{f' ({s1_team})' if s1_team else ''} {s1_best[0]:.3f}s, {s2_best[1]}{f' ({s2_team})' if s2_team else ''} {s2_best[0]:.3f}s, and {s3_best[1]}{f' ({s3_team})' if s3_team else ''} {s3_best[0]:.3f}s.",
            f"{unique_sector_leaders} different driver(s) hold sector-best times, so lap ranking is shaped by sector balance rather than a single dominant split.",
            f"With a {gap_fmt} table gap from P1 to P2, even one mid-sector correction can materially change grid order.",
        ]

    if "sector execution gap" in chart_title_l and has_sector_triplet:
        return [
            f"{leader_name}{f' ({leader_team})' if leader_team else ''} leads the split at {lead_pace_fmt}, and this stack shows where rivals leave lap potential unused versus their own best sectors.",
            f"Sector best references are S1 {s1_best[1]} ({s1_best[0]:.3f}s), S2 {s2_best[1]} ({s2_best[0]:.3f}s), and S3 {s3_best[1]} ({s3_best[0]:.3f}s).",
            "The tallest color block in each stack marks the biggest execution leak and should be the first setup/driver target.",
        ]

    if "sector delta heatmap" in chart_title_l and has_sector_triplet:
        return [
            f"The heatmap is anchored to S1/S2/S3 references of {s1_best[0]:.3f}s, {s2_best[0]:.3f}s, and {s3_best[0]:.3f}s respectively.",
            f"Because {unique_sector_leaders} driver(s) share sector benchmarks, the chart highlights where contenders trade strengths across the lap.",
            "Rows with one bright sector and two dark sectors usually indicate setup specialization that limits full-lap conversion.",
        ]

    if "sector delta heatmap" in chart_title_l:
        return [
            f"{session_name} pace leader is {leader_name}{f' ({leader_team})' if leader_team else ''} at {lead_pace_fmt}, and the heatmap maps where others lose time by sector.",
            f"Front reference spread is +{gap_fmt} to P2 and +{p3_gap_fmt} to P3, so concentrated hot cells usually explain most of that deficit.",
            "Prioritize drivers with one dominant weak sector first, because that is typically the fastest setup or execution gain to recover.",
        ]

    if "mean vs top speed" in chart_title_l:
        return [
            f"The pace reference is {leader_name}{f' ({leader_team})' if leader_team else ''}, with {runner_name} close behind at +{gap_fmt}.",
            f"Top-table spread to P5 is +{p5_gap_fmt}, so this plot is best read as an explanation of why similarly ranked teams reach pace differently.",
            f"The front order in this session is {leader_name}, {runner_name}, {third_name}; compare those teams first for the clearest drag-versus-cornering tradeoff contrast.",
        ]

    if "improvement analysis" in chart_title_l:
        return [
            f"{leader_name}{f' ({leader_team})' if leader_team else ''} sets the reference at {lead_pace_fmt}, while this chart isolates who improved most run-to-run in {session_name}.",
            "Large positive improvement with low lap count often signals one optimized push-lap, while repeated medium gains suggest sustainable pace unlocking.",
            f"The current top-table separation is {gap_fmt}, so incremental gains in final attempts can still reorder nearby positions.",
        ]

    if "ideal vs best lap" in chart_title_l:
        return [
            f"{leader_name} leads at {lead_pace_fmt}; this view measures the execution gap between each driver’s theoretical ideal and achieved best lap.",
            "A near-zero gap indicates clean sector stitching, while larger gaps point to unrealized potential from traffic, tyre prep, or sequencing.",
            "Use this to distinguish true car-speed deficit from lap assembly deficit before making setup conclusions.",
        ]

    if "position trace" in chart_title_l:
        top3 = [str(r.get("Driver", "")) for r in ranked[:3] if r.get("Driver")]
        podium = ", ".join(top3) if top3 else "N/A"
        return [
            f"Final order reference is {podium}, and the trace shows how each driver reached that finish rather than only the classified result.",
            "Large slope changes typically align with pit phases, traffic release, or neutralization effects.",
            f"Compare the winner {leader_name} versus nearest finishers ({runner_name}, {third_name}) to separate strategy timing from raw pace.",
        ]

    if "stint timeline" in chart_title_l:
        stint_vals = [int(v) for v in (_num(r.get("stints")) for r in ranked) if v is not None]
        if stint_vals:
            if min(stint_vals) == max(stint_vals):
                stint_obs = f"Observed stop profile is {stint_vals[0]} stints for all listed runners"
            else:
                stint_obs = f"Observed stop profile spans {min(stint_vals)} to {max(stint_vals)} stints across the listed runners"
            return [
                f"{stint_obs}, and the front reference is {leader_name} with a median pace of {lead_pace_fmt}.",
                "Longer opening or middle segments usually indicate teams protecting track position and stretching tyre life before committing to stop windows.",
                "Shorter repeated segments usually reflect aggressive offset attempts or response calls to direct rivals in the same position battle.",
            ]
        return [
            "The stint timeline maps each driver’s tyre phases, making undercut/overcut timing differences easier to interpret.",
            "Aligned pit windows suggest convergent strategy constraints, while staggered windows signal intentional offset play.",
            "Use this alongside lap-time trend charts to judge whether each stop timing call converted into net pace.",
        ]

    if "lap time trace" in chart_title_l:
        return [
            f"{leader_name}{f' ({leader_team})' if leader_team else ''} leads at {lead_pace_fmt}, with {runner_name} at +{gap_fmt}, and this trace shows how that gap evolved lap-by-lap.",
            "Step-like improvements often mark post-stop clean-air phases, while spikes usually point to traffic or tyre drop-off.",
            f"P1-to-P3 pace spread is +{p3_gap_fmt}, which indicates how much variation contenders could tolerate during pit cycles.",
        ]

    if "degradation-corrected pace" in chart_title_l:
        return [
            f"{leader_name}{f' ({leader_team})' if leader_team else ''} is the corrected baseline at {lead_pace_fmt}, with {runner_name} at +{gap_fmt} after tyre-age normalization.",
            "Drivers whose corrected distributions stay tight and low typically had strong stint control independent of compound age profile.",
            f"The corrected benchmark gap of {gap_fmt} to P2 indicates whether raw result order is supported by sustainable pace.",
        ]

    if "degradation" in chart_title_l:
        return [
            f"{leader_name}{f' ({leader_team})' if leader_team else ''} is the pace reference at {lead_pace_fmt}, and the degradation trend tests whether that edge survives tyre-age accumulation.",
            "Flatter trend lines indicate stronger stint management, while steeper slopes usually force earlier stops or larger late-stint deficits.",
            f"Use slope differences between {leader_name}, {runner_name}, and {third_name} to estimate who can extend windows without a major pace penalty.",
        ]

    if "consistency" in chart_title_l:
        laps_leader = _num(leader.get("num_laps"))
        laps_runner = _num(runner.get("num_laps"))
        laps_third = _num(third.get("num_laps"))
        laps_txt = f" over {int(laps_leader)} laps" if laps_leader is not None else ""
        return [
            f"{leader_name}{f' ({leader_team})' if leader_team else ''} leads on table pace at {lead_pace_fmt}{laps_txt}, providing the benchmark point on this scatter.",
            f"{runner_name} is +{gap_fmt} on {int(laps_runner) if laps_runner is not None else 'N/A'} laps, while {third_name} is +{p3_gap_fmt} on {int(laps_third) if laps_third is not None else 'N/A'} laps.",
            f"Pace spread grows to +{p10_gap_fmt} by P10 in this session, so high vertical scatter at small gaps is the main consistency risk signal here.",
        ]

    if "pace distribution" in chart_title_l or "pace box" in chart_title_l:
        return [
            f"{leader_name}{f' ({leader_team})' if leader_team else ''} sets the pace reference at {lead_pace_fmt}; {runner_name} is +{gap_fmt} and {third_name} is +{p3_gap_fmt}.",
            f"Session spread is +{p5_gap_fmt} by P5 and +{p10_gap_fmt} by P10, which quantifies how quickly the pace ladder opens in this chart.",
            f"This distribution is computed from {laps_count} cleaned laps in {session_name}, so it represents timed-run pace rather than cooldown or in/out-lap noise.",
        ]

    if "median pace delta to fastest" in chart_title_l:
        return [
            f"Median pace baseline is {leader_name}{f' ({leader_team})' if leader_team else ''} at {lead_pace_fmt}, with {runner_name} and {third_name} at +{gap_fmt} and +{p3_gap_fmt}.",
            "Small median deltas indicate sustainable run-level pace, not just one-off peak lap execution.",
            "If the delta curve opens gradually, the field is compressed; sharp breaks usually indicate pace tier boundaries.",
        ]

    if "best lap delta to fastest" in chart_title_l:
        return [
            f"Best-lap benchmark is {leader_name}{f' ({leader_team})' if leader_team else ''} at {lead_pace_fmt}; closest challenger is {runner_name} at +{gap_fmt}.",
            "This reflects peak single-lap execution potential, so a low delta does not always imply equivalent long-run pace.",
            f"With P1-to-P3 only +{p3_gap_fmt}, marginal sector improvements can quickly reshuffle this split.",
        ]

    if "delta" in chart_title_l:
        return [
            f"The delta baseline is the table leader, {leader_name}, and each bar/point shows how far others are from that reference.",
            f"The closest table delta is {gap_fmt}, which is small enough that minor sector execution shifts can reorder positions.",
            "If deltas expand sharply down the order, that usually indicates a clear pace tier break rather than random lap variance.",
        ]

    return [
        f"{leader_name}{f' ({leader_team})' if leader_team else ''} is the primary session reference in the table, and this chart explains how that advantage is expressed.",
        "The most useful read is usually the first separation cluster, because it identifies the competitive front before midfield noise dominates.",
        f"Interpret these patterns with the table together; the cleaned sample here uses {laps_count} laps from {session_name}.",
    ]


def _format_laptime(seconds: float | int | None) -> str:
    if seconds is None or (isinstance(seconds, float) and pd.isna(seconds)):
        return ""
    total = float(seconds)
    mins = int(total // 60)
    secs = total - mins * 60
    return f"{mins}:{secs:06.3f}"


def _colors_for_laps(session_obj: object, laps: pd.DataFrame) -> dict[str, str]:
    drivers = sorted(laps["Driver"].dropna().astype(str).unique().tolist()) if "Driver" in laps.columns else []
    return driver_color_map(session_obj, drivers)


def _team_colors_for_laps(session_obj: object, laps: pd.DataFrame) -> dict[str, str]:
    teams = sorted(laps["Team"].dropna().astype(str).unique().tolist()) if "Team" in laps.columns else []
    return team_color_map(session_obj, teams)


def _previous_five_rounds(season: int) -> tuple[int, ...]:
    try:
        sched = fastf1.get_event_schedule(season, include_testing=False)
        if sched is None or sched.empty or "RoundNumber" not in sched.columns:
            return tuple()
        rounds = (
            pd.to_numeric(sched["RoundNumber"], errors="coerce")
            .dropna()
            .astype(int)
            .tolist()
        )
        rounds = sorted(set(r for r in rounds if r > 0))
        if not rounds:
            return tuple()
        return tuple(rounds[-5:])
    except Exception:
        return tuple()


def _same_gp_last_editions(event_name: str, season: int, count: int = 5) -> tuple[tuple[int, int], ...]:
    editions: list[tuple[int, int]] = []
    target = event_name.strip().lower()
    if not target:
        return tuple()
    for yr in range(season - 1, max(1950, season - 30), -1):
        if len(editions) >= count:
            break
        try:
            sched = fastf1.get_event_schedule(yr, include_testing=False)
            if sched is None or sched.empty or "EventName" not in sched.columns or "RoundNumber" not in sched.columns:
                continue
            match = sched[sched["EventName"].astype(str).str.strip().str.lower() == target]
            if match.empty:
                continue
            rnd = pd.to_numeric(match.iloc[0]["RoundNumber"], errors="coerce")
            if pd.isna(rnd):
                continue
            editions.append((yr, int(rnd)))
        except Exception:
            continue
    return tuple(editions)


def _strategy_highlights(
    strategy_rows: list[dict],
    undercut_rows: list[dict],
    stint_extension_rows: list[dict],
    race_mode_rows: list[dict],
    stint_pace_rows: list[dict],
) -> dict[str, list[str]]:
    def _num(v: object) -> float | None:
        try:
            f = float(v)
            if pd.isna(f):
                return None
            return f
        except (TypeError, ValueError):
            return None

    def _window_center(window: str | None) -> float | None:
        if not window:
            return None
        try:
            w = str(window).strip().upper().replace(" ", "")
            if "-" in w:
                a, b = w.split("-", 1)
                a = float(a.replace("L", ""))
                b = float(b.replace("L", ""))
                return (a + b) / 2.0
            return float(w.replace("L", ""))
        except Exception:
            return None

    pit_hl: list[str] = []
    edge_hl: list[str] = []
    ext_hl: list[str] = []
    mode_hl: list[str] = []
    compound_hl: list[str] = []

    if strategy_rows:
        with_p1 = [(r, _window_center(r.get("pit_window_1"))) for r in strategy_rows]
        with_p1 = [(r, c) for r, c in with_p1 if c is not None]
        if with_p1:
            early = min(with_p1, key=lambda x: x[1])[0]
            late = max(with_p1, key=lambda x: x[1])[0]
            pit_hl.append(
                f"Earliest projected first stop is {early.get('Team', 'N/A')} ({early.get('pit_window_1', '-')}); latest is {late.get('Team', 'N/A')} ({late.get('pit_window_1', '-')})."
            )
        max_stops = max(strategy_rows, key=lambda r: _num(r.get("expected_stops")) if _num(r.get("expected_stops")) is not None else -1)
        min_stops = min(strategy_rows, key=lambda r: _num(r.get("expected_stops")) if _num(r.get("expected_stops")) is not None else 999)
        pit_hl.append(
            f"Highest stop load is {max_stops.get('Team', 'N/A')} at {max_stops.get('expected_stops', 'N/A')} stops, while {min_stops.get('Team', 'N/A')} is lowest at {min_stops.get('expected_stops', 'N/A')}."
        )
        transitions: list[str] = []
        for r in strategy_rows[:3]:
            tail = f", then {r.get('pit2_transition')}" if r.get("pit2_transition") else ""
            transitions.append(f"{r.get('Team', 'N/A')}: {r.get('pit1_transition', '-')}{tail}")
        pit_hl.append(
            f"Opening compound paths show early intent: {' | '.join(transitions)}."
        )

    if undercut_rows:
        best_u = max(undercut_rows, key=lambda r: _num(r.get("undercut_gain_s")) if _num(r.get("undercut_gain_s")) is not None else -1)
        best_o = max(undercut_rows, key=lambda r: _num(r.get("overcut_gain_s")) if _num(r.get("overcut_gain_s")) is not None else -1)
        edge_hl.append(
            f"Strongest undercut edge is {best_u.get('Team', 'N/A')} at {best_u.get('undercut_gain_s', 'N/A')}s around {best_u.get('pit_trigger_window', '-')}; strongest overcut edge is {best_o.get('Team', 'N/A')} at {best_o.get('overcut_gain_s', 'N/A')}s."
        )
        bal = [r for r in undercut_rows if str(r.get("preferred_call", "")).upper() == "BALANCED"]
        edge_hl.append(
            f"{len(bal)} team(s) project a BALANCED call, while others show clearer bias between undercut and overcut viability."
        )
        top_pref = sorted(
            undercut_rows,
            key=lambda r: abs(float(r.get("undercut_viability_pct", 50)) - float(r.get("overcut_viability_pct", 50))),
            reverse=True,
        )[:2]
        if top_pref:
            edge_hl.append(
                "Highest call asymmetry: "
                + " | ".join(
                    f"{r.get('Team', 'N/A')} U/O {r.get('undercut_viability_pct', 'N/A')}%/{r.get('overcut_viability_pct', 'N/A')}%"
                    for r in top_pref
                )
                + "."
            )

    if stint_extension_rows:
        best_ext = max(stint_extension_rows, key=lambda r: _num(r.get("extension_viability_pct")) if _num(r.get("extension_viability_pct")) is not None else -1)
        max_pen = max(stint_extension_rows, key=lambda r: _num(r.get("expected_penalty_s")) if _num(r.get("expected_penalty_s")) is not None else -1)
        ext_hl.append(
            f"Best extension case is {best_ext.get('Team', 'N/A')} ({best_ext.get('extension_call', 'N/A')}) with {best_ext.get('extension_viability_pct', 'N/A')}% viability and +{best_ext.get('max_extend_laps', 'N/A')} laps on {best_ext.get('stint_compound', 'N/A')}."
        )
        ext_hl.append(
            f"Highest extension penalty appears at {max_pen.get('Team', 'N/A')} ({max_pen.get('expected_penalty_s', 'N/A')}s), so stretching beyond {max_pen.get('pit_window', '-')} is likely costly."
        )
        calls = {}
        for r in stint_extension_rows:
            calls[str(r.get("extension_call", "N/A"))] = calls.get(str(r.get("extension_call", "N/A")), 0) + 1
        call_mix = ", ".join(f"{k}: {v}" for k, v in sorted(calls.items()))
        ext_hl.append(f"Extension call mix across teams is {call_mix}, showing how constrained the first-stop phase is.")

    if race_mode_rows:
        aggr = max(race_mode_rows, key=lambda r: _num(r.get("aggressive_score_pct")) if _num(r.get("aggressive_score_pct")) is not None else -1)
        defn = max(race_mode_rows, key=lambda r: _num(r.get("defensive_score_pct")) if _num(r.get("defensive_score_pct")) is not None else -1)
        mode_hl.append(
            f"Most aggressive profile is {aggr.get('Team', 'N/A')} ({aggr.get('aggressive_score_pct', 'N/A')}%) around {aggr.get('primary_trigger', '-')}; most defensive is {defn.get('Team', 'N/A')} ({defn.get('defensive_score_pct', 'N/A')}%)."
        )
        balanced = [r.get("Team", "N/A") for r in race_mode_rows if str(r.get("mode", "")).upper() == "BALANCED"]
        if balanced:
            mode_hl.append(f"Balanced-mode teams are {', '.join(balanced)}, indicating higher adaptability to SC/VSC or traffic disruptions.")
        else:
            mode_hl.append("No team is fully balanced in this projection; race mode is polarized between attack and cover approaches.")
        mode_hl.append("Primary trigger windows should be treated as action zones, with final call gated by tyre state and release traffic.")

    if stint_pace_rows:
        def _best_compound(r: dict) -> tuple[str, float]:
            vals = {
                "Soft": float(r.get("soft_delta_ms", 9999)),
                "Medium": float(r.get("medium_delta_ms", 9999)),
                "Hard": float(r.get("hard_delta_ms", 9999)),
            }
            comp = min(vals, key=vals.get)
            return comp, vals[comp]

        per_team = [(r.get("Team", "N/A"),) + _best_compound(r) for r in stint_pace_rows]
        mix = {}
        for _, c, _ in per_team:
            mix[c] = mix.get(c, 0) + 1
        compound_hl.append(
            "Team-best compound split is "
            + ", ".join(f"{k}: {v} team(s)" for k, v in sorted(mix.items()))
            + "."
        )
        best_team = min(per_team, key=lambda x: x[2])
        compound_hl.append(
            f"Strongest absolute compound delta is {best_team[0]} on {best_team[1]} (+{best_team[2]/1000.0:.3f}s to own best baseline)."
        )
        sample = sorted(per_team, key=lambda x: x[2])[:3]
        compound_hl.append(
            "Top compound-efficient teams: " + " | ".join(f"{t} ({c}, +{d/1000.0:.3f}s)" for t, c, d in sample) + "."
        )

    return {
        "pit_windows": pit_hl,
        "undercut_overcut": edge_hl,
        "stint_extension": ext_hl,
        "race_mode": mode_hl,
        "compound_pace": compound_hl,
    }


def run_pipeline(
    season: int,
    round_number: int,
    train_round_end: int,
    quick: bool = False,
    ga4_measurement_id: str = "",
) -> None:
    ensure_dirs(PATHS.outputs, PATHS.tables, PATHS.plots, PATHS.models, PATHS.site)
    round_slug = f"{season}_round_{round_number:02d}"
    round_table_dir = PATHS.tables / round_slug
    round_plot_dir = PATHS.plots / round_slug
    ensure_dirs(round_table_dir, round_plot_dir)
    for old in round_table_dir.glob("*.csv"):
        old.unlink()
    for old in round_plot_dir.glob("*.png"):
        old.unlink()

    bundle = load_sessions(season, round_number, telemetry_sessions={"Qualifying"})
    practice_payload: list[dict] = []
    quali_payload: list[dict] = []
    race_payload: dict | None = None

    for session_name, sess in bundle.sessions.items():
        if session_name.startswith("FP"):
            laps = laps_dataframe(sess, session_name)
            colors = _colors_for_laps(sess, laps)
            team_colors = _team_colors_for_laps(sess, laps)
            summary = session_summary(laps)
            teammate = teammate_delta(summary)
            stints = stint_summary(laps)
            export_session_tables(round_table_dir, session_name, summary, teammate, stints)
            top5_avg = (
                laps.sort_values(["Driver", "lap_seconds"])
                .groupby("Driver", as_index=False)
                .head(5)
                .groupby("Driver", as_index=False)["lap_seconds"]
                .mean()
                .rename(columns={"lap_seconds": "top5_avg_pace_s"})
            )
            res = results_dataframe(sess, session_name)
            if not res.empty and "Position" in res.columns:
                res["Position"] = pd.to_numeric(res["Position"], errors="coerce")
                order_df = res.dropna(subset=["Position"]).sort_values("Position")[["Driver", "Position"]]
                if not order_df.empty:
                    table_df = (
                        order_df.merge(
                            summary[["Driver", "Team", "laps"]].rename(columns={"laps": "num_laps"}),
                            on="Driver",
                            how="left",
                        )
                        .merge(top5_avg, on="Driver", how="left")
                        .head(10)
                        .rename(columns={"Position": "session_position"})
                    )
                else:
                    table_df = (
                        summary[["Driver", "Team", "laps"]]
                        .rename(columns={"laps": "num_laps"})
                        .merge(top5_avg, on="Driver", how="left")
                        .sort_values("top5_avg_pace_s")
                        .head(10)
                    )
                    table_df.insert(0, "session_position", range(1, len(table_df) + 1))
            else:
                table_df = (
                    summary[["Driver", "Team", "laps"]]
                    .rename(columns={"laps": "num_laps"})
                    .merge(top5_avg, on="Driver", how="left")
                    .sort_values("top5_avg_pace_s")
                    .head(10)
                )
                table_df.insert(0, "session_position", range(1, len(table_df) + 1))
            table_rows = (
                table_df.to_dict(orient="records") if not summary.empty else []
            )
            for row in table_rows:
                row["top5_avg_pace_fmt"] = _format_laptime(row.get("top5_avg_pace_s"))
            charts = build_session_charts(
                round_plot_dir,
                session_name,
                laps,
                stints,
                driver_colors=colors,
                team_colors=team_colors,
            )
            for c in charts:
                c["insights"] = _chart_highlights(c["title"], session_name, table_rows, int(laps.shape[0]))
            practice_payload.append(
                {
                    "session_name": session_name,
                    "drivers": int(summary.shape[0]),
                    "laps": int(laps.shape[0]),
                    "table_rows": table_rows,
                    "charts": charts,
                }
            )
            continue

        if session_name == "Qualifying":
            q_colors = _colors_for_laps(sess, laps_dataframe(sess, session_name))
            q_team_colors = _team_colors_for_laps(sess, laps_dataframe(sess, session_name))
            q_parts = qualifying_parts_dataframes(sess)
            for label in ["Q1", "Q2", "Q3"]:
                laps = q_parts.get(label)
                if laps is None or laps.empty:
                    continue
                summary = session_summary(laps)
                teammate = teammate_delta(summary)
                stints = stint_summary(laps)
                export_session_tables(round_table_dir, label, summary, teammate, stints)
                # Qualifying session order by best lap in this split.
                q_order = (
                    summary[["Driver", "Team", "best_lap_s", "median_lap_s", "laps"]]
                    .sort_values("best_lap_s")
                    .head(10)
                    .rename(
                        columns={
                            "best_lap_s": "best_lap_time_s",
                            "median_lap_s": "median_pace_s",
                            "laps": "num_laps",
                        }
                    )
                )
                q_order.insert(0, "session_position", range(1, len(q_order) + 1))
                sector_min = (
                    laps.groupby("Driver", as_index=False)
                    .agg(
                        s1_time_s=("s1_seconds", "min"),
                        s2_time_s=("s2_seconds", "min"),
                        s3_time_s=("s3_seconds", "min"),
                    )
                )
                q_table = q_order.merge(sector_min, on="Driver", how="left")
                table_rows = q_table.to_dict(orient="records")
                for row in table_rows:
                    row["s1_time_fmt"] = _format_laptime(row.get("s1_time_s"))
                    row["s2_time_fmt"] = _format_laptime(row.get("s2_time_s"))
                    row["s3_time_fmt"] = _format_laptime(row.get("s3_time_s"))
                    row["best_lap_time_fmt"] = _format_laptime(row.get("best_lap_time_s"))
                charts = build_session_charts(
                    round_plot_dir,
                    label,
                    laps,
                    stints,
                    driver_colors=q_colors,
                    team_colors=q_team_colors,
                )
                telem_chart = plot_quali_fastest_two_lap_delta(
                    round_plot_dir, label, sess, label, q_colors
                )
                if telem_chart:
                    charts.append(telem_chart)
                for c in charts:
                    c["insights"] = _chart_highlights(c["title"], label, table_rows, int(laps.shape[0]))
                quali_payload.append(
                    {
                        "session_name": label,
                        "drivers": int(summary.shape[0]),
                        "laps": int(laps.shape[0]),
                        "table_rows": table_rows,
                        "charts": charts,
                    }
                )
            continue

        if session_name == "Race":
            laps = laps_dataframe(sess, session_name)
            colors = _colors_for_laps(sess, laps)
            summary = session_summary(laps)
            teammate = teammate_delta(summary)
            stints = stint_summary(laps)
            export_session_tables(round_table_dir, session_name, summary, teammate, stints)
            race_results = results_dataframe(sess, session_name)
            stint_counts = (
                stints[stints["lap_count"] >= 2]
                .groupby("Driver", as_index=False)["Stint"]
                .nunique()
                .rename(columns={"Stint": "stints"})
            )
            race_table = (
                race_results[["Driver", "Team", "Position"]]
                .rename(columns={"Position": "race_position"})
                .merge(
                    summary[["Driver", "median_lap_s", "best_lap_s"]].rename(
                        columns={"median_lap_s": "median_pace_s", "best_lap_s": "fastest_lap_s"}
                    ),
                    on="Driver",
                    how="left",
                )
                .merge(stint_counts, on="Driver", how="left")
            )
            race_table["race_position"] = pd.to_numeric(race_table["race_position"], errors="coerce")
            race_table["stints"] = pd.to_numeric(race_table["stints"], errors="coerce").astype("Int64")
            race_table = race_table.dropna(subset=["race_position"]).sort_values("race_position").head(10)
            race_driver_order = race_table["Driver"].dropna().astype(str).tolist()
            table_rows = race_table.to_dict(orient="records")
            for row in table_rows:
                row["median_pace_fmt"] = _format_laptime(row.get("median_pace_s"))
                row["fastest_lap_fmt"] = _format_laptime(row.get("fastest_lap_s"))
            charts = build_session_charts(
                round_plot_dir,
                session_name,
                laps,
                stints,
                driver_order=race_driver_order,
                driver_colors=colors,
            )
            for c in charts:
                c["insights"] = _chart_highlights(c["title"], session_name, table_rows, int(laps.shape[0]))
            race_payload = {
                "session_name": session_name,
                "drivers": int(summary.shape[0]),
                "laps": int(laps.shape[0]),
                "table_rows": table_rows,
                "charts": charts,
            }

    # ML component
    train_season = max(2018, season - 1)
    train_round_end_eff = train_round_end if quick else max(train_round_end, 24)
    target_df = _round_feature_frame(
        season,
        round_number,
        prior_season=train_season,
        bundle=bundle,
    )
    model_out = train_and_predict(target_df, pd.DataFrame())

    model_out.quali_predictions.to_csv(PATHS.models / f"{round_slug}_quali_predictions.csv", index=False)
    model_out.race_predictions.to_csv(PATHS.models / f"{round_slug}_race_predictions.csv", index=False)
    write_json(PATHS.models / f"{round_slug}_metrics.json", model_out.model_metrics)
    same_gp_prev5 = _same_gp_last_editions(bundle.event_name, season, count=5)
    strategy_overview = build_strategy_overview(
        bundle,
        model_out.race_predictions,
        strategy_train_season=train_season,
        strategy_train_round_end=train_round_end_eff,
        strategy_train_rounds=None,
        strategy_train_events=(same_gp_prev5 if same_gp_prev5 else None),
    )
    strategy_highlights = _strategy_highlights(
        strategy_rows=strategy_overview.strategy_rows,
        undercut_rows=strategy_overview.undercut_rows,
        stint_extension_rows=strategy_overview.stint_extension_rows,
        race_mode_rows=strategy_overview.race_mode_rows,
        stint_pace_rows=strategy_overview.stint_pace_rows,
    )
    circuit_overview = build_circuit_overview(bundle, history_years=5)
    pace_delta_max = 1.0
    if strategy_overview.stint_pace_rows:
        vals = []
        for r in strategy_overview.stint_pace_rows:
            vals.extend([float(r.get("soft_delta_ms", 0.0)), float(r.get("medium_delta_ms", 0.0)), float(r.get("hard_delta_ms", 0.0))])
        pace_delta_max = max(1.0, max(vals))
    strategy_edge_max = 0.1
    if strategy_overview.undercut_rows:
        edge_vals = []
        for r in strategy_overview.undercut_rows:
            edge_vals.extend([abs(float(r.get("undercut_gain_s", 0.0))), abs(float(r.get("overcut_gain_s", 0.0)))])
        strategy_edge_max = max(0.1, max(edge_vals))

    # Site
    race_site_slug = round_slug
    race_site_dir = PATHS.site / "races" / race_site_slug
    copy_plot_assets(round_plot_dir, race_site_dir)
    ctx = {
        "season": season,
        "round_number": round_number,
        "event_name": bundle.event_name,
        "asset_version": int(time.time()),
        "ga4_measurement_id": ga4_measurement_id.strip(),
        "practice_sessions": practice_payload,
        "qualifying_sessions": quali_payload,
        "race_session": race_payload,
        "quali_predictions": model_out.quali_predictions.to_dict(orient="records"),
        "race_predictions": model_out.race_predictions.to_dict(orient="records"),
        "metrics": model_out.model_metrics,
        "strategy_rows": strategy_overview.strategy_rows,
        "stint_pace_rows": strategy_overview.stint_pace_rows,
        "team_outlook_rows": strategy_overview.team_outlook_rows,
        "undercut_rows": strategy_overview.undercut_rows,
        "stint_extension_rows": strategy_overview.stint_extension_rows,
        "race_mode_rows": strategy_overview.race_mode_rows,
        "strategy_hybrid_meta": strategy_overview.hybrid_meta,
        "strategy_highlights": strategy_highlights,
        "circuit_overview": circuit_overview,
        "pit_window_reference": strategy_overview.pit_window_reference,
        "strategy_total_laps": strategy_overview.total_laps,
        "strategy_pace_delta_max": pace_delta_max,
        "strategy_edge_max": strategy_edge_max,
    }
    render_site(ctx, round_slug=race_site_slug)

    # Convenience copies so tables are easy to browse from site folder
    site_outputs = race_site_dir / "outputs"
    ensure_dirs(site_outputs)
    if site_outputs.exists():
        # Shallow sync of generated tables/models for local inspection
        for f in round_table_dir.glob("*.csv"):
            out = site_outputs / f.name
            out.write_bytes(f.read_bytes())
        for f in (PATHS.models).glob(f"{round_slug}*"):
            out = site_outputs / f.name
            out.write_bytes(f.read_bytes())

    print(f"[DONE] Built local demo site at: {PATHS.site / 'index.html'}")
    print(f"[DONE] Race pages at: {race_site_dir / 'index.html'}")


def main() -> None:
    parser = argparse.ArgumentParser(description="F1 analysis + prediction demo pipeline")
    parser.add_argument("--season", type=int, default=2025)
    parser.add_argument("--round", dest="round_number", type=int, default=1)
    parser.add_argument("--train-round-end", type=int, default=2)
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Use a short historical window for faster local iteration.",
    )
    parser.add_argument(
        "--ga4-measurement-id",
        type=str,
        default=os.getenv("PADDOCK_GA4_MEASUREMENT_ID", ""),
        help="Optional GA4 Measurement ID (e.g., G-XXXXXXX) for private analytics.",
    )
    args = parser.parse_args()
    run_pipeline(
        season=args.season,
        round_number=args.round_number,
        train_round_end=args.train_round_end,
        quick=args.quick,
        ga4_measurement_id=args.ga4_measurement_id,
    )


if __name__ == "__main__":
    main()
