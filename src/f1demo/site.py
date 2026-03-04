from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from jinja2 import Environment, FileSystemLoader, select_autoescape

from .settings import PATHS
from .utils import ensure_dirs


def _env() -> Environment:
    return Environment(
        loader=FileSystemLoader(str(PATHS.templates)),
        autoescape=select_autoescape(["html", "xml"]),
    )


def _manifest_path() -> Path:
    return PATHS.site / "races" / "manifest.json"


def _load_manifest() -> list[dict[str, Any]]:
    p = _manifest_path()
    if not p.exists():
        return []
    try:
        payload = json.loads(p.read_text(encoding="utf-8"))
        if isinstance(payload, list):
            return [r for r in payload if isinstance(r, dict)]
    except Exception:
        return []
    return []


def _save_manifest(rows: list[dict[str, Any]]) -> None:
    p = _manifest_path()
    ensure_dirs(p.parent)
    p.write_text(json.dumps(rows, indent=2), encoding="utf-8")


def _render_home(env: Environment, races: list[dict[str, Any]]) -> None:
    home_tpl = env.get_template("home.html.j2")
    current = next((r for r in races if bool(r.get("is_current", False))), None)
    available = [r for r in races if not bool(r.get("is_current", False))]
    (PATHS.site / "index.html").write_text(
        home_tpl.render(current_race=current, available_races=available),
        encoding="utf-8",
    )


def render_site(context: dict[str, Any], round_slug: str) -> Path:
    ensure_dirs(PATHS.site)
    env = _env()
    race_dir = PATHS.site / "races" / round_slug
    ensure_dirs(race_dir)

    index_tpl = env.get_template("index.html.j2")
    round_tpl = env.get_template("round.html.j2")
    strategy_tpl = env.get_template("strategy.html.j2")

    (race_dir / "index.html").write_text(index_tpl.render(**context), encoding="utf-8")
    (race_dir / "round.html").write_text(round_tpl.render(**context), encoding="utf-8")
    (race_dir / "strategy.html").write_text(strategy_tpl.render(**context), encoding="utf-8")

    manifest = _load_manifest()
    prev = next((r for r in manifest if str(r.get("slug", "")) == round_slug), {})
    row = {
        "slug": round_slug,
        "season": int(context.get("season", 0)),
        "round_number": int(context.get("round_number", 0)),
        "event_name": str(context.get("event_name", "")),
        "overview_path": f"races/{round_slug}/index.html",
        "updated_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "is_current": bool(prev.get("is_current", False)),
        "status": str(prev.get("status", "completed")),
    }
    filtered = [r for r in manifest if str(r.get("slug", "")) != round_slug]
    filtered.append(row)
    races = sorted(
        filtered,
        key=lambda r: (int(r.get("season", 0)), int(r.get("round_number", 0))),
        reverse=True,
    )
    _save_manifest(races)
    _render_home(env, races)
    return race_dir


def copy_plot_assets(round_plot_dir: Path, race_dir: Path) -> None:
    ensure_dirs(race_dir / "assets")
    for png in round_plot_dir.glob("*.png"):
        dst = race_dir / "assets" / png.name
        dst.write_bytes(png.read_bytes())
