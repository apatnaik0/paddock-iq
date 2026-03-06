from __future__ import annotations

from pathlib import Path

import fastf1
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from .utils import ensure_dirs

sns.set_theme(style="darkgrid")

COMPOUND_COLORS = {
    "SOFT": "#FF0000",
    "MEDIUM": "#FFD54F",
    "HARD": "#F5F5F5",
    "INTERMEDIATE": "#00C853",
    "WET": "#1E88E5",
    "UNKNOWN": "#8E8E8E",
}

QUALI_SECTOR_COLORS = {
    "Sector 1": "#4ea1ff",
    "Sector 2": "#00c2a8",
    "Sector 3": "#ff8a3d",
}


def _style_ax(ax: plt.Axes) -> None:
    ax.set_facecolor("#121820")
    for spine in ax.spines.values():
        spine.set_color("#3a4454")
    ax.tick_params(colors="#d9e1ee")
    ax.xaxis.label.set_color("#d9e1ee")
    ax.yaxis.label.set_color("#d9e1ee")
    ax.title.set_color("#f4f7fb")


def _style_legend(leg: plt.Legend | None) -> None:
    if leg is None:
        return
    leg.get_frame().set_facecolor("#121820")
    leg.get_frame().set_edgecolor("#3a4454")
    leg.get_frame().set_alpha(0.95)
    title = leg.get_title()
    if title:
        title.set_color("#FFFFFF")
    for txt in leg.get_texts():
        txt.set_color("#FFFFFF")


def _save_fig(fig: plt.Figure, path: Path) -> str:
    fig.patch.set_facecolor("#0d1117")
    fig.savefig(path, dpi=160, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    return path.name


def _resolve_driver_order(df: pd.DataFrame, driver_order: list[str] | None, limit: int = 10) -> list[str]:
    if df.empty or "Driver" not in df.columns:
        return []
    available = set(df["Driver"].dropna().astype(str).unique().tolist())
    if driver_order:
        order = [d for d in driver_order if d in available]
        return order[:limit]
    if "lap_seconds" in df.columns:
        return (
            df.groupby("Driver", as_index=False)["lap_seconds"]
            .median()
            .sort_values("lap_seconds")
            .head(limit)["Driver"]
            .tolist()
        )
    return df["Driver"].dropna().astype(str).drop_duplicates().head(limit).tolist()


def _clean_laps(laps_df: pd.DataFrame, session_name: str) -> pd.DataFrame:
    if laps_df.empty:
        return laps_df

    df = laps_df.copy()
    df = df[df["lap_seconds"].notna()]

    if "IsAccurate" in df.columns:
        df = df[df["IsAccurate"].fillna(False)]

    if "PitInTime" in df.columns:
        df = df[df["PitInTime"].isna()]
    if "PitOutTime" in df.columns:
        df = df[df["PitOutTime"].isna()]

    # Remove extreme outliers while keeping realistic long runs.
    # In qualifying, never trim the fast tail or we can drop the pole lap.
    session_group = _session_group(session_name)
    if session_group == "qualifying":
        high = df["lap_seconds"].quantile(0.995)
        df = df[df["lap_seconds"] <= high]
    else:
        # Practice and race: keep the full fast tail, trim only extreme slow laps.
        high = df["lap_seconds"].quantile(0.97)
        df = df[df["lap_seconds"] <= high]

    # Practice-only cleanup: remove obvious cooldown/very slow laps per driver.
    # Keep laps within a driver-relative window from their session-best lap.
    if session_group == "practice" and not df.empty:
        filtered_parts: list[pd.DataFrame] = []
        for _, g in df.groupby("Driver"):
            g = g.copy()
            if g.empty:
                continue
            best = g["lap_seconds"].min()
            # Dynamic tolerance: around 5-8s depending on lap length.
            tol = max(5.0, 0.08 * best)
            g = g[g["lap_seconds"] <= best + tol]
            filtered_parts.append(g)
        if filtered_parts:
            df = pd.concat(filtered_parts, ignore_index=True)
        else:
            df = pd.DataFrame(columns=df.columns)

    if session_name.lower() == "race":
        df = df[df["LapNumber"] > 1]

    return df


def clean_session_laps(laps_df: pd.DataFrame, session_name: str) -> pd.DataFrame:
    """Public wrapper to apply the exact lap-cleaning logic used for plotting."""
    return _clean_laps(laps_df, session_name)


def _add_tyre_age(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty or "Stint" not in df.columns:
        return df
    out = df.sort_values(["Driver", "Stint", "LapNumber"]).copy()
    out["tyre_age_lap"] = out.groupby(["Driver", "Stint"]).cumcount() + 1
    out["Compound"] = out["Compound"].fillna("UNKNOWN")
    return out


def _session_group(session_name: str) -> str:
    low = session_name.lower()
    if "race" in low:
        return "race"
    if "qual" in low or low in {"q1", "q2", "q3"}:
        return "qualifying"
    return "practice"


def session_summary(laps_df: pd.DataFrame) -> pd.DataFrame:
    if laps_df.empty:
        return pd.DataFrame()

    grp = (
        laps_df.groupby(["Driver", "Team"], as_index=False)
        .agg(
            laps=("lap_seconds", "count"),
            best_lap_s=("lap_seconds", "min"),
            median_lap_s=("lap_seconds", "median"),
            consistency_std_s=("lap_seconds", "std"),
        )
        .sort_values("median_lap_s")
    )
    grp["consistency_std_s"] = grp["consistency_std_s"].fillna(0.0)
    grp["pace_rank"] = range(1, len(grp) + 1)
    return grp


def teammate_delta(summary_df: pd.DataFrame) -> pd.DataFrame:
    if summary_df.empty:
        return pd.DataFrame()

    team_best = summary_df.groupby("Team")["median_lap_s"].transform("min")
    out = summary_df[["Driver", "Team", "median_lap_s"]].copy()
    out["delta_to_team_best_s"] = out["median_lap_s"] - team_best
    return out.sort_values(["Team", "delta_to_team_best_s"])


def stint_summary(laps_df: pd.DataFrame) -> pd.DataFrame:
    if laps_df.empty or "Stint" not in laps_df.columns:
        return pd.DataFrame()

    base = laps_df[["Driver", "Compound", "Stint", "LapNumber"]].dropna(subset=["Stint"]).copy()
    if base.empty:
        return pd.DataFrame()

    base["Compound"] = base["Compound"].fillna("UNKNOWN")
    out = (
        base.groupby(["Driver", "Stint", "Compound"], as_index=False)
        .agg(start_lap=("LapNumber", "min"), end_lap=("LapNumber", "max"), lap_count=("LapNumber", "count"))
        .sort_values(["Driver", "Stint"])
    )
    return out


def export_session_tables(round_dir: Path, session_name: str, summary: pd.DataFrame, teammate: pd.DataFrame, stints: pd.DataFrame) -> None:
    ensure_dirs(round_dir)
    summary.to_csv(round_dir / f"{session_name.lower()}_summary.csv", index=False)
    teammate.to_csv(round_dir / f"{session_name.lower()}_teammate_delta.csv", index=False)
    stints.to_csv(round_dir / f"{session_name.lower()}_stints.csv", index=False)


def _plot_pace_box(
    round_plot_dir: Path,
    session_name: str,
    df: pd.DataFrame,
    driver_order: list[str] | None = None,
    driver_colors: dict[str, str] | None = None,
) -> dict:
    order = _resolve_driver_order(df, driver_order, limit=10)
    use = df[df["Driver"].isin(order)].copy()
    palette = {d: (driver_colors or {}).get(d, "#4ea1ff") for d in order}

    fig, ax = plt.subplots(figsize=(11, 5))
    sns.boxplot(
        data=use,
        x="Driver",
        y="lap_seconds",
        hue="Driver",
        order=order,
        hue_order=order,
        palette=palette,
        ax=ax,
        legend=False,
        flierprops={"marker": "o", "markerfacecolor": "#f4f7fb", "markeredgecolor": "#f4f7fb", "markersize": 3.5, "alpha": 0.9},
        whiskerprops={"color": "#e8effc", "linewidth": 1.1},
        capprops={"color": "#e8effc", "linewidth": 1.1},
        medianprops={"color": "#ffffff", "linewidth": 1.3},
        boxprops={"edgecolor": "#e8effc", "linewidth": 1.0},
    )
    ax.set_title(f"{session_name}: Lap Time Distribution (Top 10)")
    ax.set_xlabel("Driver")
    ax.set_ylabel("Lap Time (s)")
    ax.tick_params(axis="x", rotation=45)
    _style_ax(ax)

    name = _save_fig(fig, round_plot_dir / f"{session_name.lower()}_pace_box.png")
    return {"title": "Pace Distribution", "file": name}


def _plot_delta_to_fastest(
    round_plot_dir: Path, session_name: str, df: pd.DataFrame, metric: str = "median", driver_order: list[str] | None = None
) -> dict:
    if metric == "best":
        ranked = df.groupby("Driver", as_index=False)["lap_seconds"].min().sort_values("lap_seconds").head(12)
        y_title = "Best Lap Delta (s)"
        title = "Best Lap Delta to Fastest"
        file_suffix = "best_delta"
    else:
        ranked = df.groupby("Driver", as_index=False)["lap_seconds"].median().sort_values("lap_seconds").head(12)
        y_title = "Median Pace Delta (s)"
        title = "Median Pace Delta to Fastest"
        file_suffix = "median_delta"

    if driver_order:
        ranked = ranked.set_index("Driver").reindex(driver_order[:12]).dropna().reset_index()
    order = ranked["Driver"].tolist()
    ranked["delta_s"] = ranked["lap_seconds"] - ranked["lap_seconds"].min()
    fig, ax = plt.subplots(figsize=(11, 5))
    sns.barplot(data=ranked, x="Driver", y="delta_s", order=order, ax=ax, color="#4ea1ff")
    ax.axhline(0.0, color="#9fb0c9", linewidth=1.0, linestyle="--")
    ax.set_title(f"{session_name}: {title}")
    ax.set_xlabel("Driver")
    ax.set_ylabel(y_title)
    ax.tick_params(axis="x", rotation=45)
    _style_ax(ax)

    name = _save_fig(fig, round_plot_dir / f"{session_name.lower()}_{file_suffix}.png")
    return {"title": title, "file": name}


def _plot_sector_heatmap(round_plot_dir: Path, session_name: str, df: pd.DataFrame, driver_order: list[str] | None = None) -> dict | None:
    req = {"s1_seconds", "s2_seconds", "s3_seconds", "Driver"}
    if not req.issubset(set(df.columns)):
        return None

    sec = (
        df.groupby("Driver", as_index=False)[["s1_seconds", "s2_seconds", "s3_seconds"]]
        .min()
        .dropna()
    )
    if sec.empty:
        return None
    if driver_order:
        sec = sec.set_index("Driver").reindex(driver_order[:10]).dropna()
    else:
        sec = sec.sort_values("s1_seconds").head(10).set_index("Driver")
    delta = sec - sec.min(axis=0)

    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(delta, annot=True, fmt=".3f", cmap="mako", cbar_kws={"label": "Delta to best sector (s)"}, ax=ax)
    ax.set_title(f"{session_name}: Sector Delta Heatmap")
    _style_ax(ax)

    name = _save_fig(fig, round_plot_dir / f"{session_name.lower()}_sector_heatmap.png")
    return {"title": "Sector Delta Heatmap", "file": name}


def _plot_lap_evolution(
    round_plot_dir: Path, session_name: str, df: pd.DataFrame, driver_colors: dict[str, str] | None = None
) -> dict:
    ranked = df.groupby("Driver", as_index=False)["lap_seconds"].median().sort_values("lap_seconds").head(8)
    keep = ranked["Driver"].tolist()
    use = df[df["Driver"].isin(keep)].sort_values("LapNumber")
    palette = {d: (driver_colors or {}).get(d, "#4ea1ff") for d in keep}

    fig, ax = plt.subplots(figsize=(11, 5))
    sns.lineplot(
        data=use,
        x="LapNumber",
        y="lap_seconds",
        hue="Driver",
        hue_order=keep,
        palette=palette,
        estimator="median",
        errorbar=None,
        ax=ax,
    )
    ax.set_title(f"{session_name}: Lap Evolution")
    ax.set_xlabel("Lap Number")
    ax.set_ylabel("Lap Time (s)")
    _style_ax(ax)

    name = _save_fig(fig, round_plot_dir / f"{session_name.lower()}_lap_evolution.png")
    return {"title": "Lap Evolution", "file": name}


def _plot_practice_consistency(
    round_plot_dir: Path,
    session_name: str,
    df: pd.DataFrame,
    driver_order: list[str] | None = None,
    driver_colors: dict[str, str] | None = None,
) -> dict:
    use = df.copy()
    summ = (
        use.groupby(["Driver", "Team"], as_index=False)
        .agg(median_lap_s=("lap_seconds", "median"), consistency_std_s=("lap_seconds", "std"), laps=("lap_seconds", "count"))
        .dropna()
    )
    summ = summ[summ["laps"] >= 5]
    if driver_order:
        summ = summ.set_index("Driver").reindex(driver_order[:12]).dropna().reset_index()
    else:
        summ = summ.sort_values("median_lap_s").head(12)
    if summ.empty:
        return _plot_lap_evolution(round_plot_dir, session_name, use, driver_colors)

    fig, ax = plt.subplots(figsize=(11, 5))
    drivers = summ["Driver"].tolist()
    palette_colors = sns.color_palette("tab20", n_colors=max(len(drivers), 3))
    palette = {d: (driver_colors or {}).get(d, palette_colors[i]) for i, d in enumerate(drivers)}
    marker_map: dict[str, str] = {d: "o" for d in drivers}
    for team, team_drivers in summ.groupby("Team")["Driver"]:
        ordered = [d for d in drivers if d in set(team_drivers.tolist())]
        for idx, drv in enumerate(ordered):
            marker_map[drv] = "s" if idx % 2 == 1 else "o"

    min_laps = float(summ["laps"].min())
    max_laps = float(summ["laps"].max())

    def _size_from_laps(laps: float) -> float:
        if max_laps <= min_laps:
            return 140.0
        return 70.0 + ((float(laps) - min_laps) / (max_laps - min_laps)) * 240.0

    for _, row in summ.iterrows():
        driver = str(row["Driver"])
        ax.scatter(
            float(row["median_lap_s"]),
            float(row["consistency_std_s"]),
            s=_size_from_laps(float(row["laps"])),
            marker=marker_map.get(driver, "o"),
            c=[palette.get(driver, "#4ea1ff")],
            edgecolors="#f4f7fb",
            linewidths=0.9,
            alpha=0.95,
            label=driver,
            zorder=3,
        )
    ax.set_title(f"{session_name}: Pace vs Consistency")
    ax.set_xlabel("Median Lap Time (s)")
    ax.set_ylabel("Lap Time Std Dev (s)")
    from matplotlib.lines import Line2D
    handles = [
        Line2D(
            [0],
            [0],
            marker=marker_map.get(d, "o"),
            color="none",
            markerfacecolor=palette[d],
            markeredgecolor="#f4f7fb",
            markeredgewidth=0.9,
            markersize=7,
            label=d,
        )
        for d in drivers
    ]
    leg = ax.legend(handles=handles, title="Driver", bbox_to_anchor=(1.02, 1), loc="upper left", borderaxespad=0.0, framealpha=0.85)
    _style_legend(leg)
    _style_ax(ax)

    name = _save_fig(fig, round_plot_dir / f"{session_name.lower()}_consistency.png")
    return {"title": "Pace vs Consistency", "file": name}


def _plot_practice_team_speed_profile(
    round_plot_dir: Path, session_name: str, df: pd.DataFrame, team_colors: dict[str, str] | None = None
) -> dict | None:
    speed_cols = [c for c in ["SpeedI1", "SpeedI2", "SpeedFL", "SpeedST"] if c in df.columns]
    if not speed_cols or "Team" not in df.columns:
        return None

    use = df.dropna(subset=["Team"]).copy()
    if use.empty:
        return None

    # Focus on fast laps only: quickest 30% per team based on lap time.
    fast_parts: list[pd.DataFrame] = []
    for _, g in use.groupby("Team"):
        g = g.dropna(subset=["lap_seconds"]).copy()
        if g.empty:
            continue
        cutoff = g["lap_seconds"].quantile(0.30)
        fast_parts.append(g[g["lap_seconds"] <= cutoff])
    if not fast_parts:
        return None
    fast = pd.concat(fast_parts, ignore_index=True)

    fast["lap_mean_speed"] = fast[speed_cols].mean(axis=1, skipna=True)
    fast["lap_top_speed"] = fast[speed_cols].max(axis=1, skipna=True)
    team = (
        fast.groupby("Team", as_index=False)
        .agg(mean_speed_kph=("lap_mean_speed", "median"), top_speed_kph=("lap_top_speed", "max"), laps=("lap_seconds", "count"))
        .dropna(subset=["mean_speed_kph", "top_speed_kph"])
    )
    if team.empty:
        return None

    fig, ax = plt.subplots(figsize=(11, 6))
    colors = team["Team"].map(lambda t: (team_colors or {}).get(str(t), "#4ea1ff"))
    ax.scatter(
        team["mean_speed_kph"],
        team["top_speed_kph"],
        s=70 + team["laps"].clip(upper=40) * 6,
        c=colors.tolist(),
        edgecolors="#f4f7fb",
        linewidths=0.8,
        alpha=0.9,
    )
    for _, r in team.iterrows():
        ax.text(float(r["mean_speed_kph"]) + 0.15, float(r["top_speed_kph"]) + 0.08, str(r["Team"]), fontsize=8, color="#dce8f9")

    ax.set_title(f"{session_name}: Team Mean Speed vs Top Speed (Fast Laps)")
    ax.set_xlabel("Team Mean Speed (km/h)")
    ax.set_ylabel("Team Top Speed (km/h)")
    from matplotlib.lines import Line2D
    handles = [
        Line2D(
            [0],
            [0],
            marker="o",
            color="none",
            markerfacecolor=(team_colors or {}).get(str(t), "#4ea1ff"),
            markeredgecolor="#f4f7fb",
            markeredgewidth=0.8,
            markersize=7,
            label=str(t),
        )
        for t in sorted(team["Team"].astype(str).unique().tolist())
    ]
    leg = ax.legend(handles=handles, title="Team", bbox_to_anchor=(1.02, 1), loc="upper left", borderaxespad=0.0, framealpha=0.85)
    _style_legend(leg)
    _style_ax(ax)

    name = _save_fig(fig, round_plot_dir / f"{session_name.lower()}_team_speed_profile.png")
    return {"title": "Team Mean vs Top Speed", "file": name}


def _plot_practice_long_run_pace(
    round_plot_dir: Path,
    session_name: str,
    df: pd.DataFrame,
    driver_order: list[str] | None = None,
    driver_colors: dict[str, str] | None = None,
) -> dict | None:
    req = {"Driver", "Stint", "LapNumber", "lap_seconds"}
    if not req.issubset(set(df.columns)):
        return None

    use = df.dropna(subset=["Driver", "Stint", "LapNumber", "lap_seconds"]).copy()
    if use.empty:
        return None
    use["Driver"] = use["Driver"].astype(str)
    use["Stint"] = pd.to_numeric(use["Stint"], errors="coerce")
    use = use[use["Stint"].notna()].copy()
    if use.empty:
        return None
    use["Stint"] = use["Stint"].astype(int)
    use = use.sort_values(["Driver", "Stint", "LapNumber"])
    use["stint_lap_idx"] = use.groupby(["Driver", "Stint"]).cumcount() + 1
    # Remove launch/out-lap effects from long-run estimate.
    use = use[use["stint_lap_idx"] >= 3].copy()
    if use.empty:
        return None

    if "Compound" not in use.columns:
        use["Compound"] = "UNKNOWN"
    use["Compound"] = use["Compound"].fillna("UNKNOWN").astype(str).str.upper()

    stints = (
        use.groupby(["Driver", "Team", "Stint", "Compound"], as_index=False)
        .agg(lap_count=("lap_seconds", "count"), median_lap_s=("lap_seconds", "median"))
    )
    stints = stints[stints["lap_count"] >= 3].copy()
    if stints.empty:
        return None

    # One representative long run per driver: longest stint, then quickest median.
    stints = stints.sort_values(["Driver", "lap_count", "median_lap_s"], ascending=[True, False, True])
    rep = stints.groupby("Driver", as_index=False).first()
    if driver_order:
        rep = rep.set_index("Driver").reindex(driver_order[:12]).dropna(subset=["Team", "median_lap_s"]).reset_index()
    else:
        rep = rep.sort_values("median_lap_s").head(12)
    if rep.empty:
        return None

    rep = rep.sort_values("median_lap_s")
    rep["delta_s"] = rep["median_lap_s"] - rep["median_lap_s"].min()

    fig, ax = plt.subplots(figsize=(11, 5.8))
    colors = rep["Driver"].map(lambda d: (driver_colors or {}).get(str(d), "#4ea1ff")).tolist()
    ax.bar(rep["Driver"], rep["delta_s"], color=colors, edgecolor="#f4f7fb", linewidth=0.7)
    ax.axhline(0.0, color="#9fb0c9", linewidth=1.0, linestyle="--")
    for i, row in rep.reset_index(drop=True).iterrows():
        ax.text(
            i,
            float(row["delta_s"]) + 0.01,
            f"{int(row['lap_count'])}L {str(row['Compound'])[:3]}",
            ha="center",
            va="bottom",
            fontsize=8.5,
            color="#dce8f9",
        )

    ax.set_title(f"{session_name}: Long-Run Pace (Representative Stint)")
    ax.set_xlabel("Driver")
    ax.set_ylabel("Median Stint Pace Delta (s)")
    ax.tick_params(axis="x", rotation=45)
    _style_ax(ax)

    name = _save_fig(fig, round_plot_dir / f"{session_name.lower()}_long_run_pace.png")

    leader = rep.iloc[0]
    p2 = rep.iloc[1] if len(rep) > 1 else None
    p3 = rep.iloc[2] if len(rep) > 2 else None
    rank_for_gap = min(5, len(rep))
    rank_delta = float(rep.iloc[rank_for_gap - 1]["delta_s"]) if len(rep) > 0 else 0.0
    tail_delta = float(rep.iloc[-1]["delta_s"]) if len(rep) > 0 else 0.0
    insights = [
        (
            f"Long-run reference is {leader['Driver']} ({leader['Team']}) at {leader['median_lap_s']:.3f}s"
            + (
                f", with {p2['Driver']} at +{float(p2['delta_s']):.3f}s"
                if p2 is not None
                else ""
            )
            + (
                f" and {p3['Driver']} at +{float(p3['delta_s']):.3f}s."
                if p3 is not None
                else "."
            )
        ),
        "Each bar uses one representative race-run stint per driver (minimum 3 laps, excluding opening stint laps).",
        f"Within the plotted long-run group, spread is +{rank_delta:.3f}s to P{rank_for_gap} and +{tail_delta:.3f}s from fastest to slowest.",
    ]
    return {"title": "Long-Run Pace", "file": name, "insights": insights}


def _plot_practice_compound_usage(
    round_plot_dir: Path,
    session_name: str,
    df: pd.DataFrame,
    driver_order: list[str] | None = None,
) -> dict | None:
    if "Compound" not in df.columns or "Driver" not in df.columns:
        return None

    use = df.dropna(subset=["Driver", "Compound"]).copy()
    if use.empty:
        return None

    use["Driver"] = use["Driver"].astype(str)
    use["Compound"] = use["Compound"].astype(str).str.upper()
    use["Compound"] = use["Compound"].replace(
        {
            "C1": "HARD",
            "C2": "HARD",
            "C3": "MEDIUM",
            "C4": "SOFT",
            "C5": "SOFT",
        }
    )
    use = use[use["Compound"].isin(["SOFT", "MEDIUM", "HARD"])].copy()
    if use.empty:
        return None

    if driver_order:
        keep = [d for d in driver_order[:10] if d in set(use["Driver"].unique().tolist())]
    else:
        keep = (
            use.groupby("Driver", as_index=False)["lap_seconds"]
            .median()
            .sort_values("lap_seconds")
            .head(10)["Driver"]
            .tolist()
        )
    if not keep:
        return None
    use = use[use["Driver"].isin(keep)].copy()

    counts = (
        use.groupby(["Driver", "Compound"], as_index=False)
        .size()
        .rename(columns={"size": "laps"})
    )
    piv = counts.pivot(index="Driver", columns="Compound", values="laps").fillna(0.0)
    for c in ["SOFT", "MEDIUM", "HARD"]:
        if c not in piv.columns:
            piv[c] = 0.0
    piv = piv[["SOFT", "MEDIUM", "HARD"]]
    piv["total"] = piv.sum(axis=1)
    piv = piv[piv["total"] > 0].copy()
    if piv.empty:
        return None

    pct = piv[["SOFT", "MEDIUM", "HARD"]].div(piv["total"], axis=0) * 100.0
    if driver_order:
        pct = pct.reindex([d for d in keep if d in pct.index]).dropna(how="all")
    else:
        pct = pct.sort_values(["SOFT", "MEDIUM", "HARD"], ascending=False)
    if pct.empty:
        return None

    fig, ax = plt.subplots(figsize=(12, 6))
    y = np.arange(len(pct))
    left = np.zeros(len(pct))
    comp_order = ["SOFT", "MEDIUM", "HARD"]
    comp_colors = [COMPOUND_COLORS["SOFT"], COMPOUND_COLORS["MEDIUM"], COMPOUND_COLORS["HARD"]]

    for comp, color in zip(comp_order, comp_colors):
        vals = pct[comp].to_numpy()
        bars = ax.barh(y, vals, left=left, color=color, edgecolor="#f4f7fb", linewidth=0.5, label=comp.title())
        for i, (bar, v) in enumerate(zip(bars, vals)):
            if v >= 10.0:
                ax.text(left[i] + v / 2.0, bar.get_y() + bar.get_height() / 2.0, f"{v:.0f}%", ha="center", va="center", color="#091018", fontsize=8, fontweight="bold")
        left += vals

    ax.set_yticks(y)
    ax.set_yticklabels(pct.index.tolist())
    ax.invert_yaxis()
    ax.set_xlim(0, 100)
    ax.set_xlabel("Share of Timed Running (%)")
    ax.set_ylabel("Driver")
    ax.set_title(f"{session_name}: Compound Usage Mix (Soft / Medium / Hard)")
    leg = ax.legend(title="Compound", bbox_to_anchor=(1.02, 1), loc="upper left", borderaxespad=0.0, framealpha=0.9)
    _style_legend(leg)
    _style_ax(ax)

    name = _save_fig(fig, round_plot_dir / f"{session_name.lower()}_compound_usage.png")

    soft_leader = pct["SOFT"].idxmax()
    med_leader = pct["MEDIUM"].idxmax()
    hard_leader = pct["HARD"].idxmax()
    balance = (pct[["SOFT", "MEDIUM", "HARD"]].max(axis=1) - pct[["SOFT", "MEDIUM", "HARD"]].min(axis=1)).idxmin()
    max_hard = float(pct["HARD"].max())
    hard_line = (
        f"Highest hard-running share: {hard_leader} ({max_hard:.1f}%), indicating the strongest bias to longer-stint baseline work in this session."
        if max_hard > 0.0
        else "No hard-compound running is present for the selected drivers in this session; programs are concentrated on soft/medium work."
    )
    insights = [
        f"Highest soft-running share: {soft_leader} ({pct.loc[soft_leader, 'SOFT']:.1f}%). Highest medium share: {med_leader} ({pct.loc[med_leader, 'MEDIUM']:.1f}%).",
        hard_line,
        f"Most balanced S/M/H split is {balance} ({pct.loc[balance, 'SOFT']:.1f}% / {pct.loc[balance, 'MEDIUM']:.1f}% / {pct.loc[balance, 'HARD']:.1f}%); compound bias can reflect deliberate program masking as much as pure pace.",
    ]
    return {"title": "Compound Usage Mix", "file": name, "insights": insights}


def _plot_quali_improvement(round_plot_dir: Path, session_name: str, df: pd.DataFrame, driver_order: list[str] | None = None) -> dict | None:
    use = df.copy()
    grp = use.sort_values(["Driver", "LapNumber"]).groupby("Driver")
    imp = grp["lap_seconds"].agg(first="first", last="last", best="min").dropna().reset_index()
    if imp.empty:
        return None
    imp["improvement"] = imp["first"] - imp["best"]
    if driver_order:
        imp = imp.set_index("Driver").reindex(driver_order[:12]).dropna().reset_index()
    else:
        imp = imp.sort_values("improvement", ascending=False).head(12)

    fig, ax = plt.subplots(figsize=(10, 5))
    sns.barplot(data=imp, x="Driver", y="improvement", ax=ax, color="#00c2a8")
    ax.set_title(f"{session_name}: Improvement from First Timed Lap")
    ax.set_xlabel("Driver")
    ax.set_ylabel("Improvement (s)")
    ax.tick_params(axis="x", rotation=45)
    _style_ax(ax)

    name = _save_fig(fig, round_plot_dir / f"{session_name.lower()}_improvement.png")
    return {"title": "Improvement Analysis", "file": name}


def _plot_quali_sector_delta_comparison(
    round_plot_dir: Path,
    session_name: str,
    df: pd.DataFrame,
    driver_order: list[str] | None = None,
    team_colors: dict[str, str] | None = None,
) -> dict | None:
    req = {"Driver", "s1_seconds", "s2_seconds", "s3_seconds"}
    if not req.issubset(set(df.columns)):
        return None

    sec = (
        df.groupby(["Driver", "Team"], as_index=False)[["s1_seconds", "s2_seconds", "s3_seconds"]]
        .min()
        .dropna()
    )
    if sec.empty:
        return None
    if driver_order:
        sec = sec.set_index("Driver").reindex(driver_order[:10]).dropna().reset_index()
    else:
        sec = sec.nsmallest(10, "s1_seconds")

    fig, axes = plt.subplots(1, 3, figsize=(18, 6), sharex=False, sharey=False)
    sector_specs = [
        ("s1_seconds", "Sector 1", QUALI_SECTOR_COLORS["Sector 1"]),
        ("s2_seconds", "Sector 2", QUALI_SECTOR_COLORS["Sector 2"]),
        ("s3_seconds", "Sector 3", QUALI_SECTOR_COLORS["Sector 3"]),
    ]
    for ax, (col, title, color) in zip(axes, sector_specs):
        ranked = sec[["Driver", "Team", col]].sort_values(col).copy()
        ranked["delta_ms"] = (ranked[col] - ranked[col].min()) * 1000.0
        bar_colors = ranked["Team"].map(lambda t: (team_colors or {}).get(str(t), color)).tolist()
        ax.barh(ranked["Driver"], ranked["delta_ms"], color=bar_colors, edgecolor="#f4f7fb", linewidth=0.5)
        ax.invert_yaxis()  # fastest at top
        ax.set_title(title)
        ax.set_xlabel("Delta to Fastest (ms)")
        if col != "s1_seconds":
            ax.set_ylabel("")
        else:
            ax.set_ylabel("Driver")
        _style_ax(ax)

    fig.suptitle(f"{session_name}: Sector Ranking (Best Sector Times)", y=1.02, color="#f4f7fb")
    fig.tight_layout()

    name = _save_fig(fig, round_plot_dir / f"{session_name.lower()}_sector_delta_comparison.png")
    return {"title": "Sector Delta Comparison", "file": name}


def _plot_quali_ideal_vs_best(
    round_plot_dir: Path, session_name: str, df: pd.DataFrame, driver_order: list[str] | None = None
) -> dict | None:
    req = {"Driver", "lap_seconds", "s1_seconds", "s2_seconds", "s3_seconds"}
    if not req.issubset(set(df.columns)):
        return None

    base = df.dropna(subset=["Driver", "lap_seconds", "s1_seconds", "s2_seconds", "s3_seconds"]).copy()
    if base.empty:
        return None

    stats = (
        base.groupby("Driver", as_index=False)
        .agg(
            best_lap_s=("lap_seconds", "min"),
            best_s1_s=("s1_seconds", "min"),
            best_s2_s=("s2_seconds", "min"),
            best_s3_s=("s3_seconds", "min"),
        )
    )
    stats["ideal_lap_s"] = stats["best_s1_s"] + stats["best_s2_s"] + stats["best_s3_s"]
    stats["realization_gap_s"] = stats["best_lap_s"] - stats["ideal_lap_s"]

    if driver_order:
        stats = stats.set_index("Driver").reindex(driver_order[:10]).dropna().reset_index()
    else:
        stats = stats.sort_values("best_lap_s").head(10)
    if stats.empty:
        return None

    order = stats["Driver"].tolist()
    stats["gap_ms"] = stats["realization_gap_s"] * 1000.0

    fig, ax = plt.subplots(figsize=(11, 5))
    sns.barplot(
        data=stats,
        x="Driver",
        y="gap_ms",
        order=order,
        color="#4ea1ff",
        ax=ax,
    )
    ax.axhline(0.0, color="#9fb0c9", linewidth=1.0, linestyle="--")
    ax.set_title(f"{session_name}: Ideal vs Best Lap (Execution Gap)")
    ax.set_xlabel("Driver")
    ax.set_ylabel("Best - Ideal (ms)")
    ax.tick_params(axis="x", rotation=45)
    # Tight y-range so differences are readable.
    ymax = float(stats["gap_ms"].max()) if not stats.empty else 0.0
    ax.set_ylim(0, max(30.0, ymax * 1.25))
    _style_ax(ax)

    name = _save_fig(fig, round_plot_dir / f"{session_name.lower()}_ideal_vs_best.png")
    return {"title": "Ideal vs Best Lap", "file": name}


def _plot_quali_sector_execution_gap(
    round_plot_dir: Path, session_name: str, df: pd.DataFrame, driver_order: list[str] | None = None
) -> dict | None:
    req = {"Driver", "lap_seconds", "s1_seconds", "s2_seconds", "s3_seconds"}
    if not req.issubset(set(df.columns)):
        return None

    base = df.dropna(subset=["Driver", "lap_seconds", "s1_seconds", "s2_seconds", "s3_seconds"]).copy()
    if base.empty:
        return None

    # Driver's best single sectors across the session split
    mins = (
        base.groupby("Driver", as_index=False)
        .agg(best_s1_s=("s1_seconds", "min"), best_s2_s=("s2_seconds", "min"), best_s3_s=("s3_seconds", "min"))
    )
    # Sector values on each driver's best full lap
    best_lap_rows = (
        base.sort_values(["Driver", "lap_seconds"])
        .drop_duplicates(subset=["Driver"], keep="first")[["Driver", "s1_seconds", "s2_seconds", "s3_seconds"]]
        .rename(
            columns={
                "s1_seconds": "bestlap_s1_s",
                "s2_seconds": "bestlap_s2_s",
                "s3_seconds": "bestlap_s3_s",
            }
        )
    )
    gaps = mins.merge(best_lap_rows, on="Driver", how="inner")
    gaps["s1_gap_ms"] = (gaps["bestlap_s1_s"] - gaps["best_s1_s"]) * 1000.0
    gaps["s2_gap_ms"] = (gaps["bestlap_s2_s"] - gaps["best_s2_s"]) * 1000.0
    gaps["s3_gap_ms"] = (gaps["bestlap_s3_s"] - gaps["best_s3_s"]) * 1000.0
    for col in ["s1_gap_ms", "s2_gap_ms", "s3_gap_ms"]:
        gaps[col] = gaps[col].clip(lower=0.0)

    if driver_order:
        gaps = gaps.set_index("Driver").reindex(driver_order[:10]).dropna().reset_index()
    else:
        gaps["total_gap_ms"] = gaps["s1_gap_ms"] + gaps["s2_gap_ms"] + gaps["s3_gap_ms"]
        gaps = gaps.sort_values("total_gap_ms").head(10)
    if gaps.empty:
        return None

    order = gaps["Driver"].tolist()
    g = gaps.set_index("Driver").reindex(order)
    x = np.arange(len(order))
    s1 = g["s1_gap_ms"].values
    s2 = g["s2_gap_ms"].values
    s3 = g["s3_gap_ms"].values

    fig, ax = plt.subplots(figsize=(11, 5))
    ax.bar(x, s1, color=QUALI_SECTOR_COLORS["Sector 1"], label="Sector 1")
    ax.bar(x, s2, bottom=s1, color=QUALI_SECTOR_COLORS["Sector 2"], label="Sector 2")
    ax.bar(x, s3, bottom=s1 + s2, color=QUALI_SECTOR_COLORS["Sector 3"], label="Sector 3")
    ax.set_title(f"{session_name}: Sector Execution Gap")
    ax.set_xlabel("Driver")
    ax.set_ylabel("Best-lap Sector Gap to Driver Sector Best (ms)")
    ax.set_xticks(x)
    ax.set_xticklabels(order, rotation=45)
    ax.tick_params(axis="x", rotation=45)
    leg = ax.legend(title="Sector", bbox_to_anchor=(1.02, 1), loc="upper left", borderaxespad=0.0, framealpha=0.85)
    _style_legend(leg)
    _style_ax(ax)

    name = _save_fig(fig, round_plot_dir / f"{session_name.lower()}_sector_execution_gap.png")
    return {"title": "Sector Execution Gap", "file": name}


def plot_quali_fastest_two_lap_delta(
    round_plot_dir: Path,
    session_name: str,
    qual_session_obj: object,
    split_label: str,
    driver_colors: dict[str, str] | None = None,
) -> dict | None:
    label_idx = {"Q1": 0, "Q2": 1, "Q3": 2}
    if split_label not in label_idx:
        return None

    try:
        parts = qual_session_obj.laps.split_qualifying_sessions()
    except Exception:
        return None
    if label_idx[split_label] >= len(parts):
        return None
    split = parts[label_idx[split_label]]
    if split is None or len(split) == 0:
        return None

    ranked = split[split["LapTime"].notna()].sort_values("LapTime")
    if ranked.empty:
        return None
    fastest = ranked.drop_duplicates(subset=["Driver"]).head(2)
    if len(fastest) < 2:
        return None
    drv_ref = str(fastest.iloc[0]["Driver"])
    drv_cmp = str(fastest.iloc[1]["Driver"])
    team_ref = str(fastest.iloc[0].get("Team", ""))
    team_cmp = str(fastest.iloc[1].get("Team", ""))

    try:
        lap_ref = split.pick_drivers(drv_ref).pick_fastest()
        lap_cmp = split.pick_drivers(drv_cmp).pick_fastest()
        delta, ref_tel, cmp_tel = fastf1.utils.delta_time(lap_ref, lap_cmp)
    except Exception:
        return None

    c_ref = (driver_colors or {}).get(drv_ref, "#4ea1ff")
    c_cmp = (driver_colors or {}).get(drv_cmp, "#00c2a8")
    # If both laps are from the same team (or same mapped color), force contrast.
    if (team_ref and team_cmp and team_ref == team_cmp) or (c_ref.lower() == c_cmp.lower()):
        c_cmp = "#ff4d6d"

    fig, axes = plt.subplots(2, 1, figsize=(15, 10), sharex=True, gridspec_kw={"height_ratios": [3, 2]})
    ax_speed, ax_delta = axes
    ax_speed.plot(ref_tel["Distance"], ref_tel["Speed"], color=c_ref, linewidth=2.0, label=f"{drv_ref} (faster)")
    ax_speed.plot(cmp_tel["Distance"], cmp_tel["Speed"], color=c_cmp, linewidth=2.0, label=f"{drv_cmp}")
    ax_speed.set_title(f"{split_label}: Fastest Two Best-Lap Comparison")
    ax_speed.set_ylabel("Speed (km/h)")
    leg1 = ax_speed.legend(
        title="Driver",
        bbox_to_anchor=(1.02, 1),
        loc="upper left",
        borderaxespad=0.0,
        framealpha=0.85,
    )
    _style_legend(leg1)
    _style_ax(ax_speed)

    corners = None
    # Corner markers/labels like FastF1 speed-trace example.
    try:
        circuit_info = qual_session_obj.get_circuit_info()
        corners = circuit_info.corners.dropna(subset=["Distance"]).copy()
        v_min = float(min(ref_tel["Speed"].min(), cmp_tel["Speed"].min()))
        v_max = float(max(ref_tel["Speed"].max(), cmp_tel["Speed"].max()))
        if not corners.empty:
            ax_speed.vlines(
                x=corners["Distance"],
                ymin=v_min - 20.0,
                ymax=v_max + 20.0,
                linestyles="dotted",
                colors="#ffffff",
                linewidth=0.7,
                alpha=0.9,
            )
            for _, corner in corners.iterrows():
                label = f"{int(corner['Number'])}{str(corner['Letter']).strip()}"
                ax_speed.text(
                    float(corner["Distance"]),
                    v_min - 28.0,
                    label,
                    ha="center",
                    va="center_baseline",
                    fontsize=8,
                    color="#c7d4e8",
                )
            ax_speed.set_ylim(v_min - 36.0, v_max + 24.0)
    except Exception:
        corners = None

    delta_fast_minus_slow_ms = (-delta) * 1000.0
    ax_delta.plot(ref_tel["Distance"], delta_fast_minus_slow_ms, color="#f4f7fb", linewidth=1.8)
    ax_delta.axhline(0.0, color="#9fb0c9", linewidth=1.0, linestyle="--")
    ax_delta.set_xlabel("Distance (m)")
    ax_delta.set_ylabel(f"{drv_ref} - {drv_cmp} (ms)")
    if corners is not None and not corners.empty:
        d_min = float(delta_fast_minus_slow_ms.min())
        d_max = float(delta_fast_minus_slow_ms.max())
        ax_delta.vlines(
            x=corners["Distance"],
            ymin=d_min - 12.0,
            ymax=d_max + 12.0,
            linestyles="dotted",
            colors="#ffffff",
            linewidth=0.7,
            alpha=0.9,
        )
        for _, corner in corners.iterrows():
            label = f"{int(corner['Number'])}{str(corner['Letter']).strip()}"
            ax_delta.text(
                float(corner["Distance"]),
                d_min - 18.0,
                label,
                ha="center",
                va="center_baseline",
                fontsize=8,
                color="#c7d4e8",
            )
        ax_delta.set_ylim(d_min - 24.0, d_max + 14.0)
    _style_ax(ax_delta)

    name = _save_fig(fig, round_plot_dir / f"{session_name.lower()}_fastest_two_delta.png")
    return {"title": "Fastest Two Lap Delta", "file": name}


def _plot_stint_timeline(
    round_plot_dir: Path, session_name: str, stints: pd.DataFrame, driver_order: list[str] | None = None
) -> dict | None:
    if stints.empty:
        return None

    stints_sorted = stints.sort_values(["Driver", "Stint", "start_lap"]).copy()
    if driver_order:
        stints_sorted = stints_sorted[stints_sorted["Driver"].isin(driver_order[:10])]
    stints_sorted["Compound"] = stints_sorted["Compound"].str.upper()
    stints_sorted["Compound"] = stints_sorted["Compound"].where(
        stints_sorted["Compound"].isin(["SOFT", "MEDIUM", "HARD", "INTERMEDIATE", "INTER", "WET"]), "UNKNOWN"
    )
    if driver_order:
        drivers = [d for d in driver_order[:10] if d in set(stints_sorted["Driver"].tolist())]
    else:
        drivers = sorted(stints_sorted["Driver"].dropna().astype(str).unique().tolist())
    if not drivers:
        return None

    fig, ax = plt.subplots(figsize=(12, 6))
    compounds_seen: set[str] = set()
    total_by_driver: dict[str, float] = {}
    driver_stint_rows: dict[str, list[dict]] = {}
    for driver in drivers:
        drows = (
            stints_sorted[stints_sorted["Driver"] == driver]
            .sort_values("Stint")
            .to_dict(orient="records")
        )
        driver_stint_rows[driver] = drows
        total_by_driver[driver] = float(sum(float(r.get("lap_count", 0) or 0) for r in drows))

    # If totals differ only slightly, align bar end points to the max total.
    # This avoids visual mismatch caused by occasional missing lap records.
    totals = list(total_by_driver.values())
    if totals:
        max_total = max(totals)
        min_total = min(totals)
        if max_total - min_total <= 2.0:
            for driver in drivers:
                gap = max_total - total_by_driver.get(driver, 0.0)
                if gap > 0.0 and driver_stint_rows.get(driver):
                    driver_stint_rows[driver][-1]["lap_count"] = float(driver_stint_rows[driver][-1].get("lap_count", 0) or 0) + gap

    for driver in drivers:
        driver_stints = driver_stint_rows.get(driver, [])
        previous_stint_end = 0.0
        for row in driver_stints:
            compound = str(row.get("Compound", "UNKNOWN")).upper()
            if compound == "INTER":
                compound = "INTERMEDIATE"
            if compound not in {"SOFT", "MEDIUM", "HARD", "INTERMEDIATE", "WET", "UNKNOWN"}:
                compound = "UNKNOWN"
            compounds_seen.add(compound)
            width = float(row.get("lap_count", 0))
            if width <= 0:
                continue
            ax.barh(
                y=driver,
                width=width,
                left=previous_stint_end,
                color=COMPOUND_COLORS[compound],
                edgecolor="#10151d",
                linewidth=0.8,
                height=0.72,
            )
            previous_stint_end += width

    ax.set_title(f"{session_name}: Stint Timeline by Compound")
    ax.set_xlabel("Lap")
    ax.set_ylabel("Driver")
    ax.grid(axis="x", color="#2b3647", alpha=0.7, linewidth=0.6)
    ax.grid(axis="y", visible=False)
    ax.invert_yaxis()
    from matplotlib.lines import Line2D
    legend_order = [c for c in ["SOFT", "MEDIUM", "HARD", "INTERMEDIATE", "WET", "UNKNOWN"] if c in compounds_seen]
    handles = [Line2D([0], [0], color=COMPOUND_COLORS[c], lw=6, label=c.title()) for c in legend_order]
    if handles:
        leg = ax.legend(
            handles=handles,
            title="Tyre",
            bbox_to_anchor=(1.02, 1),
            loc="upper left",
            borderaxespad=0.0,
            framealpha=0.9,
        )
        _style_legend(leg)
    _style_ax(ax)

    name = _save_fig(fig, round_plot_dir / f"{session_name.lower()}_stints_colored.png")
    return {"title": "Stint Timeline (Compound Coded)", "file": name}


def _driver_degradation_metrics(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame()

    work = _add_tyre_age(df)
    rows: list[dict] = []

    for driver, ddf in work.groupby("Driver"):
        ddf = ddf.dropna(subset=["lap_seconds", "tyre_age_lap"]).copy()
        if len(ddf) < 8:
            continue

        x = ddf["tyre_age_lap"].astype(float).values
        y = ddf["lap_seconds"].astype(float).values
        if np.unique(x).shape[0] < 3:
            continue

        slope, intercept = np.polyfit(x, y, 1)
        corrected = y - slope * (x - 1.0)

        rows.append(
            {
                "Driver": driver,
                "degradation_s_per_lap": slope,
                "corrected_pace_s": float(np.nanmedian(corrected)),
                "raw_pace_s": float(np.nanmedian(y)),
            }
        )

    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows)


def _plot_degradation(
    round_plot_dir: Path, session_name: str, race_df: pd.DataFrame, driver_order: list[str] | None = None
) -> dict | None:
    m = _driver_degradation_metrics(race_df)
    if m.empty:
        return None

    if driver_order:
        m = m.set_index("Driver").reindex(driver_order[:12]).dropna().reset_index()
    else:
        m = m.sort_values("degradation_s_per_lap").head(12)
    fig, ax = plt.subplots(figsize=(11, 5))
    sns.barplot(data=m, x="Driver", y="degradation_s_per_lap", ax=ax, color="#ff8a3d")
    ax.set_title(f"{session_name}: Tyre Degradation Slope")
    ax.set_xlabel("Driver")
    ax.set_ylabel("s/lap (higher = more degradation)")
    ax.tick_params(axis="x", rotation=45)
    _style_ax(ax)

    name = _save_fig(fig, round_plot_dir / f"{session_name.lower()}_degradation.png")
    return {"title": "Tyre Degradation", "file": name}


def _plot_corrected_pace(
    round_plot_dir: Path,
    session_name: str,
    race_df: pd.DataFrame,
    driver_order: list[str] | None = None,
    driver_colors: dict[str, str] | None = None,
) -> dict | None:
    work = _add_tyre_age(race_df)
    rows: list[dict] = []
    for driver, ddf in work.groupby("Driver"):
        ddf = ddf.dropna(subset=["lap_seconds", "tyre_age_lap"]).copy()
        if len(ddf) < 8:
            continue
        x = ddf["tyre_age_lap"].astype(float).values
        y = ddf["lap_seconds"].astype(float).values
        if np.unique(x).shape[0] < 3:
            continue
        slope, _ = np.polyfit(x, y, 1)
        corrected = y - slope * (x - 1.0)
        for v in corrected:
            rows.append({"Driver": driver, "corrected_lap_s": float(v)})

    if not rows:
        return None
    cdf = pd.DataFrame(rows)
    ranked = cdf.groupby("Driver", as_index=False)["corrected_lap_s"].median()
    if driver_order:
        ranked = ranked.set_index("Driver").reindex(driver_order[:12]).dropna().reset_index()
    else:
        ranked = ranked.sort_values("corrected_lap_s").head(12)
    order = ranked["Driver"].tolist()
    use = cdf[cdf["Driver"].isin(order)]
    palette = {d: (driver_colors or {}).get(d, "#4ea1ff") for d in order}
    fig, ax = plt.subplots(figsize=(11, 5))
    sns.boxplot(
        data=use,
        x="Driver",
        y="corrected_lap_s",
        order=order,
        hue="Driver",
        hue_order=order,
        palette=palette,
        legend=False,
        ax=ax,
        flierprops={"marker": "o", "markerfacecolor": "#f4f7fb", "markeredgecolor": "#f4f7fb", "markersize": 3.5, "alpha": 0.9},
        whiskerprops={"color": "#e8effc", "linewidth": 1.1},
        capprops={"color": "#e8effc", "linewidth": 1.1},
        medianprops={"color": "#ffffff", "linewidth": 1.3},
        boxprops={"edgecolor": "#e8effc", "linewidth": 1.0},
    )
    ax.set_title(f"{session_name}: Degradation-Corrected Pace Distribution")
    ax.set_xlabel("Driver")
    ax.set_ylabel("Corrected Lap Time (s)")
    ax.tick_params(axis="x", rotation=45)
    _style_ax(ax)

    name = _save_fig(fig, round_plot_dir / f"{session_name.lower()}_corrected_pace.png")
    return {"title": "Degradation-Corrected Pace", "file": name}


def _plot_position_trace(
    round_plot_dir: Path,
    session_name: str,
    race_df: pd.DataFrame,
    driver_order: list[str] | None = None,
    driver_colors: dict[str, str] | None = None,
) -> dict | None:
    if "Position" not in race_df.columns:
        return None

    pos = race_df.dropna(subset=["Position"]).copy()
    if pos.empty:
        return None

    if driver_order:
        keep = [d for d in driver_order[:10] if d in set(pos["Driver"].tolist())]
    else:
        ranked = pos.groupby("Driver", as_index=False)["Position"].median().sort_values("Position").head(10)
        keep = ranked["Driver"].tolist()
    use = pos[pos["Driver"].isin(keep)].sort_values("LapNumber")
    palette = {d: (driver_colors or {}).get(d, "#4ea1ff") for d in keep}

    style_map: dict[str, str] = {d: "-" for d in keep}
    for team, drivers in use.groupby("Team")["Driver"]:
        ordered = [d for d in keep if d in set(drivers.tolist())]
        for idx, drv in enumerate(ordered):
            style_map[drv] = "--" if idx % 2 == 1 else "-"

    fig, ax = plt.subplots(figsize=(11, 5))
    for driver in keep:
        ddf = use[use["Driver"] == driver].sort_values("LapNumber")
        if ddf.empty:
            continue
        ax.plot(
            ddf["LapNumber"],
            ddf["Position"],
            color=palette.get(driver, "#4ea1ff"),
            linestyle=style_map.get(driver, "-"),
            linewidth=2.0,
            label=driver,
        )
    ax.invert_yaxis()
    ax.set_title(f"{session_name}: Position Trace")
    ax.set_xlabel("Lap")
    ax.set_ylabel("Track Position")
    leg = ax.legend(title="Driver", bbox_to_anchor=(1.02, 1), loc="upper left", borderaxespad=0.0, framealpha=0.85)
    _style_legend(leg)
    _style_ax(ax)

    name = _save_fig(fig, round_plot_dir / f"{session_name.lower()}_position_trace.png")
    return {"title": "Position Trace", "file": name}


def _plot_race_lap_time_trace(
    round_plot_dir: Path,
    session_name: str,
    race_df: pd.DataFrame,
    driver_order: list[str] | None = None,
    driver_colors: dict[str, str] | None = None,
) -> dict | None:
    req = {"Driver", "Team", "LapNumber", "lap_seconds"}
    if not req.issubset(set(race_df.columns)):
        return None

    use = race_df.dropna(subset=["Driver", "LapNumber", "lap_seconds"]).copy()
    if use.empty:
        return None
    if driver_order:
        keep = [d for d in driver_order[:10] if d in set(use["Driver"].tolist())]
    else:
        ranked = use.groupby("Driver", as_index=False)["lap_seconds"].median().sort_values("lap_seconds").head(10)
        keep = ranked["Driver"].tolist()
    if not keep:
        return None

    use = use[use["Driver"].isin(keep)].sort_values("LapNumber")
    palette = {d: (driver_colors or {}).get(d, "#4ea1ff") for d in keep}
    style_map: dict[str, str] = {d: "-" for d in keep}
    for _, drivers in use.groupby("Team")["Driver"]:
        ordered = [d for d in keep if d in set(drivers.tolist())]
        for idx, drv in enumerate(ordered):
            style_map[drv] = "--" if idx % 2 == 1 else "-"

    fig, ax = plt.subplots(figsize=(12, 6))
    for driver in keep:
        ddf = use[use["Driver"] == driver].sort_values("LapNumber")
        if ddf.empty:
            continue
        ax.plot(
            ddf["LapNumber"],
            ddf["lap_seconds"],
            color=palette.get(driver, "#4ea1ff"),
            linestyle=style_map.get(driver, "-"),
            linewidth=1.8,
            alpha=0.95,
            label=driver,
        )

    ax.set_title(f"{session_name}: Lap Time Trace")
    ax.set_xlabel("Lap Number")
    ax.set_ylabel("Lap Time (s)")
    leg = ax.legend(title="Driver", bbox_to_anchor=(1.02, 1), loc="upper left", borderaxespad=0.0, framealpha=0.85)
    _style_legend(leg)
    _style_ax(ax)

    name = _save_fig(fig, round_plot_dir / f"{session_name.lower()}_lap_time_trace.png")
    return {"title": "Lap Time Trace", "file": name}


def build_session_charts(
    round_plot_dir: Path,
    session_name: str,
    laps_df: pd.DataFrame,
    stints: pd.DataFrame,
    driver_order: list[str] | None = None,
    precleaned: bool = False,
    driver_colors: dict[str, str] | None = None,
    team_colors: dict[str, str] | None = None,
) -> list[dict]:
    ensure_dirs(round_plot_dir)
    cleaned = laps_df.copy() if precleaned else _clean_laps(laps_df, session_name)
    if cleaned.empty:
        return []

    charts: list[dict] = []
    session_group = _session_group(session_name)

    if session_group == "practice":
        charts.append(_plot_pace_box(round_plot_dir, session_name, cleaned, driver_order, driver_colors))
        charts.append(_plot_delta_to_fastest(round_plot_dir, session_name, cleaned, metric="median", driver_order=driver_order))
        sector_chart = _plot_sector_heatmap(round_plot_dir, session_name, cleaned, driver_order)
        if sector_chart:
            charts.append(sector_chart)
        charts.append(_plot_practice_consistency(round_plot_dir, session_name, cleaned, driver_order, driver_colors))
        lrp = _plot_practice_long_run_pace(round_plot_dir, session_name, cleaned, driver_order, driver_colors)
        if lrp:
            charts.append(lrp)
        comp_mix = _plot_practice_compound_usage(round_plot_dir, session_name, cleaned, driver_order)
        if comp_mix:
            charts.append(comp_mix)
        sp = _plot_practice_team_speed_profile(round_plot_dir, session_name, cleaned, team_colors)
        if sp:
            charts.append(sp)
        return charts[:7]

    elif session_group == "qualifying":
        sec_comp = _plot_quali_sector_delta_comparison(round_plot_dir, session_name, cleaned, driver_order, team_colors)
        if sec_comp:
            charts.append(sec_comp)
        else:
            charts.append(_plot_delta_to_fastest(round_plot_dir, session_name, cleaned, metric="median", driver_order=driver_order))
        charts.append(_plot_delta_to_fastest(round_plot_dir, session_name, cleaned, metric="best", driver_order=driver_order))
        sector_chart = _plot_sector_heatmap(round_plot_dir, session_name, cleaned, driver_order)
        if sector_chart:
            charts.append(sector_chart)
        imp = _plot_quali_improvement(round_plot_dir, session_name, cleaned, driver_order)
        if imp:
            charts.append(imp)
        else:
            charts.append(_plot_delta_to_fastest(round_plot_dir, session_name, cleaned, metric="median", driver_order=driver_order))
        seg = _plot_quali_sector_execution_gap(round_plot_dir, session_name, cleaned, driver_order)
        if seg:
            charts.append(seg)
        return charts[:5]

    elif session_group == "race":
        pos = _plot_position_trace(round_plot_dir, session_name, cleaned, driver_order, driver_colors)
        if pos:
            charts.append(pos)
        ltt = _plot_race_lap_time_trace(round_plot_dir, session_name, cleaned, driver_order, driver_colors)
        if ltt:
            charts.append(ltt)

        stint_chart = _plot_stint_timeline(round_plot_dir, session_name, stints, driver_order)
        if stint_chart:
            charts.append(stint_chart)

        deg = _plot_degradation(round_plot_dir, session_name, cleaned, driver_order)
        if deg:
            charts.append(deg)

        corr = _plot_corrected_pace(round_plot_dir, session_name, cleaned, driver_order, driver_colors)
        if corr:
            charts.append(corr)
        if len(charts) < 5:
            charts.append(_plot_pace_box(round_plot_dir, session_name, cleaned, driver_order, driver_colors))
        return charts[:5]

    return charts
