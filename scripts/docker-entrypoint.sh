#!/usr/bin/env sh
set -eu

mode="${PADDOCK_MODE:-auto}"
season="${SEASON:-$(date -u +%Y)}"
train_round_end="${TRAIN_ROUND_END:-2}"
min_laps="${MIN_LAPS:-10}"
min_drivers="${MIN_DRIVERS:-5}"
lookahead_days="${LOOKAHEAD_DAYS:-7}"
ga4="${PADDOCK_GA4_MEASUREMENT_ID:-}"

bool_flag() {
  value="${1:-}"
  case "$(printf '%s' "$value" | tr '[:upper:]' '[:lower:]')" in
    1|true|yes|y|on) return 0 ;;
    *) return 1 ;;
  esac
}

if [ "$#" -gt 0 ]; then
  exec "$@"
fi

case "$mode" in
  auto)
    set -- python -m src.f1demo.auto_update --season "$season" --train-round-end "$train_round_end" --min-laps "$min_laps" --min-drivers "$min_drivers" --lookahead-days "$lookahead_days"
    if [ -n "${ROUND:-}" ]; then set -- "$@" --round "$ROUND"; fi
    if bool_flag "${QUICK:-true}"; then set -- "$@" --quick; fi
    if bool_flag "${FORCE_UPDATE:-false}"; then set -- "$@" --force; fi
    if bool_flag "${DRY_RUN:-false}"; then set -- "$@" --dry-run; fi
    if bool_flag "${NO_COMPLETE_AFTER_RACE:-false}"; then set -- "$@" --no-complete-after-race; fi
    if [ -n "$ga4" ]; then set -- "$@" --ga4-measurement-id "$ga4"; fi
    exec "$@"
    ;;
  pipeline)
    if [ -z "${ROUND:-}" ]; then
      echo "ROUND is required when PADDOCK_MODE=pipeline" >&2
      exit 2
    fi
    set -- python -m src.f1demo.pipeline --season "$season" --round "$ROUND" --train-round-end "$train_round_end"
    if bool_flag "${QUICK:-true}"; then set -- "$@" --quick; fi
    if [ -n "$ga4" ]; then set -- "$@" --ga4-measurement-id "$ga4"; fi
    exec "$@"
    ;;
  serve)
    exec python -m http.server "${PORT:-8000}" -d site
    ;;
  *)
    echo "Unsupported PADDOCK_MODE: $mode" >&2
    echo "Supported modes: auto, pipeline, serve" >&2
    exit 2
    ;;
esac
