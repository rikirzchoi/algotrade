"""
macro.regime — manual macro-regime configuration, updated weekly by a human.

The MacroRegime is a rich data structure that captures the operator's
view of the current market environment across multiple dimensions:
risk appetite, trend character, sector bias, geopolitical tension,
Fed stance, and upcoming high-impact events.

Strategies read this at the start of each session to decide whether
to trade at full size, reduced size, or avoid certain sectors entirely.

This is intentionally kept simple and human-driven — no automated regime
detection — because misclassification by an algorithm in a fast-moving macro
environment is more dangerous than the minor delay of a weekly manual update.

Edit macro/regime.json weekly after reviewing your World Monitor dashboard.
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass, field
from datetime import date, datetime
from pathlib import Path

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Valid value sets
# ---------------------------------------------------------------------------

_VALID_RISK_APPETITE = ("low", "neutral", "high")
_VALID_TREND_REGIME = ("trending", "ranging", "volatile")
_VALID_GEO_TENSION = ("low", "elevated", "high")
_VALID_FED_STANCE = ("dovish", "neutral", "hawkish")
_VALID_IMPACT = ("low", "medium", "high")

_SECTOR_MAP: dict[str, list[str]] = {
    "energy":      ["XLE"],
    "tech":        ["XLK", "QQQ", "NVDA"],
    "defensive":   ["SPY", "IWM"],
    "financials":  ["XLF"],
    "healthcare":  ["XLV"],
    "commodities": ["GLD", "USO"],
}


# ---------------------------------------------------------------------------
# Dataclass
# ---------------------------------------------------------------------------

@dataclass
class MacroRegime:
    """
    Current macro-regime configuration.

    Loaded from a JSON file (macro/regime.json) at engine start.  The JSON
    file is the single source of truth and is edited manually by the operator
    after a weekly World Monitor review.

    Attributes
    ----------
    risk_appetite : str
        Overall risk tolerance: "low" | "neutral" | "high"
    trend_regime : str
        Prevailing market character: "trending" | "ranging" | "volatile"
    sector_bias : list[str]
        Preferred sectors to concentrate exposure in.  Empty → trade full
        universe.  Valid tags: "energy", "tech", "defensive", "financials",
        "healthcare", "commodities".
    geopolitical_tension : str
        Current geopolitical risk level: "low" | "elevated" | "high"
    fed_stance : str
        Fed policy lean: "dovish" | "neutral" | "hawkish"
    upcoming_events : list[dict]
        Calendar of upcoming macro events.  Each dict requires keys
        ``date`` (YYYY-MM-DD), ``event`` (str), ``impact`` ("low"|"medium"|"high").
    notes : str
        Free-text weekly summary written by the operator.
    last_updated : str
        ISO timestamp set automatically by :func:`save_regime`.
    """

    risk_appetite: str = "neutral"
    trend_regime: str = "ranging"
    sector_bias: list[str] = field(default_factory=list)
    geopolitical_tension: str = "low"
    fed_stance: str = "neutral"
    upcoming_events: list[dict] = field(default_factory=list)
    notes: str = ""
    last_updated: str = ""

    def __post_init__(self) -> None:
        """Validate all fields; reset invalid values to defaults and log warnings."""
        if self.risk_appetite not in _VALID_RISK_APPETITE:
            logger.warning(
                "MacroRegime: invalid risk_appetite %r — resetting to 'neutral'",
                self.risk_appetite,
            )
            self.risk_appetite = "neutral"

        if self.trend_regime not in _VALID_TREND_REGIME:
            logger.warning(
                "MacroRegime: invalid trend_regime %r — resetting to 'ranging'",
                self.trend_regime,
            )
            self.trend_regime = "ranging"

        if self.geopolitical_tension not in _VALID_GEO_TENSION:
            logger.warning(
                "MacroRegime: invalid geopolitical_tension %r — resetting to 'low'",
                self.geopolitical_tension,
            )
            self.geopolitical_tension = "low"

        if self.fed_stance not in _VALID_FED_STANCE:
            logger.warning(
                "MacroRegime: invalid fed_stance %r — resetting to 'neutral'",
                self.fed_stance,
            )
            self.fed_stance = "neutral"

        clean_events: list[dict] = []
        for evt in self.upcoming_events:
            if not isinstance(evt, dict):
                logger.warning("MacroRegime: upcoming_events item is not a dict — skipping: %r", evt)
                continue
            missing = {"date", "event", "impact"} - evt.keys()
            if missing:
                logger.warning(
                    "MacroRegime: upcoming_events item missing keys %s — skipping: %r",
                    missing,
                    evt,
                )
                continue
            if evt["impact"] not in _VALID_IMPACT:
                logger.warning(
                    "MacroRegime: upcoming_events item has invalid impact %r — skipping: %r",
                    evt["impact"],
                    evt,
                )
                continue
            clean_events.append(evt)
        self.upcoming_events = clean_events


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------

def load_regime(path: str | Path = "macro/regime.json") -> MacroRegime:
    """
    Load a :class:`MacroRegime` from *path*.

    Returns a default :class:`MacroRegime` if the file does not exist or
    contains malformed JSON.  Never raises.

    Parameters
    ----------
    path : str | Path
        Path to the JSON file.  Defaults to ``macro/regime.json``.
    """
    p = Path(path)
    if not p.exists():
        logger.info("macro/regime.json not found — using defaults")
        return MacroRegime()
    try:
        with p.open("r", encoding="utf-8") as fh:
            data = json.load(fh)
        return MacroRegime(**data)
    except json.JSONDecodeError as exc:
        logger.warning("MacroRegime: malformed JSON in %s (%s) — using defaults", p, exc)
        return MacroRegime()
    except Exception as exc:  # noqa: BLE001
        logger.warning("MacroRegime: failed to load %s (%s) — using defaults", p, exc)
        return MacroRegime()


def save_regime(
    regime: MacroRegime,
    path: str | Path = "macro/regime.json",
) -> None:
    """
    Persist *regime* to *path* as indented JSON.

    Sets :attr:`MacroRegime.last_updated` to the current local time in ISO
    format before writing.  Writes atomically via a ``.tmp`` file + rename.
    Creates parent directories as needed.  Never raises.

    Parameters
    ----------
    regime : MacroRegime
        The regime instance to serialise.
    path : str | Path
        Destination file path.  Defaults to ``macro/regime.json``.
    """
    try:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)

        regime.last_updated = datetime.now().isoformat()

        data = {
            "risk_appetite": regime.risk_appetite,
            "trend_regime": regime.trend_regime,
            "sector_bias": regime.sector_bias,
            "geopolitical_tension": regime.geopolitical_tension,
            "fed_stance": regime.fed_stance,
            "upcoming_events": regime.upcoming_events,
            "notes": regime.notes,
            "last_updated": regime.last_updated,
        }

        tmp = p.with_suffix(".tmp")
        with tmp.open("w", encoding="utf-8") as fh:
            json.dump(data, fh, indent=2)
        os.replace(tmp, p)
        logger.debug("MacroRegime saved to %s", p)
    except Exception as exc:  # noqa: BLE001
        logger.error("MacroRegime: failed to save to %s: %s", path, exc)


# ---------------------------------------------------------------------------
# Query helpers
# ---------------------------------------------------------------------------

def get_position_size_multiplier(regime: MacroRegime) -> float:
    """
    Return a position-size multiplier in ``[0.1, 1.0]`` derived from
    *regime*.

    Base multiplier by risk_appetite
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    * ``"high"``    → 1.00 (full size)
    * ``"neutral"`` → 0.75 (reduced)
    * ``"low"``     → 0.50 (half size)

    Geopolitical discount applied on top
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    * ``"elevated"`` → × 0.9
    * ``"high"``     → × 0.8

    The result is clamped to ``[0.1, 1.0]``.

    Parameters
    ----------
    regime : MacroRegime
        Current regime configuration.
    """
    base = {"high": 1.0, "neutral": 0.75, "low": 0.5}.get(regime.risk_appetite, 0.75)

    geo_discount = {"elevated": 0.9, "high": 0.8}.get(regime.geopolitical_tension, 1.0)

    result = base * geo_discount
    return max(0.1, min(1.0, result))


def get_preferred_symbols(
    regime: MacroRegime,
    universe: list[str],
) -> list[str]:
    """
    Filter *universe* to symbols that match the regime's sector bias.

    If :attr:`MacroRegime.sector_bias` is empty the full *universe* is
    returned unchanged.

    If a sector_bias is set but none of its mapped symbols appear in
    *universe*, the full *universe* is returned as a fallback.

    Parameters
    ----------
    regime : MacroRegime
        Current regime configuration.
    universe : list[str]
        Full symbol universe to filter (e.g. from ``cfg.instruments.swing_symbols``).
    """
    if not regime.sector_bias:
        return universe

    universe_set = set(universe)
    seen: set[str] = set()
    preferred: list[str] = []

    for tag in regime.sector_bias:
        for sym in _SECTOR_MAP.get(tag, []):
            if sym in universe_set and sym not in seen:
                preferred.append(sym)
                seen.add(sym)

    if not preferred:
        return universe

    return preferred


def has_high_impact_event_soon(
    regime: MacroRegime,
    days_ahead: int = 2,
    reference_date: date | None = None,
) -> bool:
    """
    Return ``True`` if any high-impact event falls within the look-ahead window.

    The window is ``[reference_date, reference_date + days_ahead]`` inclusive.
    Malformed date strings in :attr:`MacroRegime.upcoming_events` are silently
    skipped.

    Parameters
    ----------
    regime : MacroRegime
        Current regime configuration.
    days_ahead : int
        Number of calendar days to look ahead (inclusive).  Default: 2.
    reference_date : date | None
        The anchor date.  Defaults to :func:`datetime.date.today`.
    """
    from datetime import timedelta

    ref = reference_date if reference_date is not None else date.today()
    cutoff = ref + timedelta(days=days_ahead)

    for evt in regime.upcoming_events:
        if evt.get("impact") != "high":
            continue
        try:
            evt_date = date.fromisoformat(evt["date"])
        except (KeyError, ValueError, TypeError):
            continue
        if ref <= evt_date <= cutoff:
            return True
    return False


def get_regime_summary(regime: MacroRegime) -> str:
    """
    Return a compact one-line summary of *regime* suitable for logging.

    Format::

        Regime: risk=neutral | trend=ranging | fed=neutral | geo=low | bias=none

    Parameters
    ----------
    regime : MacroRegime
        Current regime configuration.
    """
    return (
        f"Regime: risk={regime.risk_appetite} | "
        f"trend={regime.trend_regime} | "
        f"fed={regime.fed_stance} | "
        f"geo={regime.geopolitical_tension} | "
        f"bias={regime.sector_bias or 'none'}"
    )
