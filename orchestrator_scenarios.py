#!/usr/bin/env python3
# orchestrator_scenarios.py
# ^ Shebang so the file can be run directly on Unix-like systems (chmod +x).
#   Keeps the script portable for CLI demos in your white paper repo.

"""
Behavioral Liquidity Scenario Engine — Public, Fully-Commented Edition

This script demonstrates an AI-driven workflow for generating and analyzing
behavioral liquidity scenarios and comparing them to a deterministic baseline.
It is designed to support a white paper that argues: AI-curated macro scenarios
+ instrument-impact reasoning + behavioral overlays can outperform traditional,
static, rules-only approaches for liquidity risk.

High-level flow (why each step exists):
1) Load a liquidity profile (banking-book positions & metadata).
2) Fetch a minimal macro "stub" (or live macro inputs in production).
3) Ask an AI agent to propose scenarios (macro shocks, instrument impacts,
   severity labels, and rationale).
4) Convert portfolio into contractual daily cashflows (deterministic baseline).
5) Compute HQLA (eligibility rules) and baseline KPIs (LCR, survival days, etc.).
6) For each AI scenario:
   a) Translate macro shocks -> behavior parameters (deposit runoffs, roll/non-roll).
   b) Apply instrument impacts (AI proposes trade or facility actions) to cashflows.
   c) Overlay behavioral outflows and recompute KPIs.
   d) Summarize effects (plain-English + numbers) and collect AI narrative.
7) Compare gap-to-targets (e.g., LCR >= 130%, survival >= 180d) to show how AI
   improves outcomes and surfaces binding constraints earlier.

External dependencies (what they provide):
- ai_agents.scenario_generator.generate_ai_scenario:
    Returns AI-proposed scenarios with macro_shocks, instrument_impacts, rationale.
- ai_agents.ai_narratives.ai_macro_view / ai_explain_scenario:
    Produces explainable narratives for macro context and scenario result interpretation.
- portfolio_engine.*:
    Deterministic engines for cashflow rolling, behavioral estimation, KPI computation,
    HQLA eligibility, and various helper transforms.

Data assumptions:
- The liquidity profile JSON contains the portfolio and instrument metadata your
  portfolio_engine expects (e.g., balances, maturities, product types, CSA flags).
- KPIs returned by compute_kpis_from_daily_detail include keys used below:
  'lcr', 'hqla', 'worst_30d_outflow', 'peak_cumulative_outflow', 'survival_days'.

This code is purposely verbose in comments to serve as documentation in public repos.
"""

import json  # Parse/serialize JSON inputs/outputs for portability and auditability.
from pathlib import Path  # Robust, cross-platform filesystem paths.
from datetime import datetime  # Stamp outputs with an as-of date for reproducibility.

# AI scenario generation (macro -> scenarios with impacts and rationale).
from ai_agents.scenario_generator import generate_ai_scenario

# Deterministic portfolio & liquidity engines (clear separation of concerns):
from portfolio_engine import (
    portfolio_to_flows,                     # Converts position inventory -> contractual CF series.
    roll_daily_contractual_flows_detail,    # Expands CF series into daily buckets over horizon.
    estimate_behavioral_series_detail,      # Adds behavioral overlays (e.g., deposit runoff).
    combine_daily_detail,                   # Merges contractual and behavioral CFs.
    compute_kpis_from_daily_detail,         # Calculates LCR, survival days, peak outflows, etc.
    compute_size_proxies,                   # Derives size proxies (deposit base, notionals) for summaries.
    scenario_to_behavior_params,            # Maps macro shocks/severity -> behavior parameters.
    compute_hqla_eligible,                  # Computes HQLA eligible stock from portfolio.
    apply_instrument_impacts_to_cashflows,  # Applies AI-proposed instrument impacts before behavior.
)

# AI narrative helpers to make results explainable to executives & regulators.
from ai_agents.ai_narratives import ai_macro_view, ai_explain_scenario


# ---------- Simple macro stub ----------
def fetch_macro_data():
    """
    WHY:
      Provide a minimal macro snapshot to seed AI scenario generation in demos.
      In production, replace with live data (FRED/Bloomberg/Refinitiv) or your macro desk.
      This can be replaced by calls to real-time API economic data providers.

    RETURNS:
      dict with macro indicators used by the AI agent (units noted for clarity):
        - fed_funds_rate: float, % (upper bound or effective)
        - us10y_yield: float, % (benchmark duration anchor)
        - vix: float, index level (risk sentiment / volatility proxy)
        - credit_spreads_baa: int, basis points (IG proxy)
        - credit_spreads_hy: int, basis points (HY proxy)
    """
    return {
        "fed_funds_rate": 5.25,   # %
        "us10y_yield": 4.40,      # %
        "vix": 18.2,              # index
        "credit_spreads_baa": 190,  # bps
        "credit_spreads_hy": 410    # bps
    }


def load_liquidity_profile(file_path: str):
    """
    WHY:
      Centralized file loader to keep I/O concerns separated and testable.

    PARAMS:
      file_path: str - path to JSON file containing the liquidity profile.

    RETURNS:
      dict - parsed JSON representing positions and metadata required by portfolio_engine.

    NOTES:
      - Validates existence at call site for clearer error message.
      - In controlled demos, the file is 'sample_portfolio.json' in project root.
    """
    with open(file_path, "r") as f:  # Open file in read mode (text).
        return json.load(f)          # Parse JSON into Python dict.


def compute_gap_to_targets(kpis: dict, lcr_target: float = 1.30, survival_target_days: int = 180) -> dict:
    """
    WHY:
      Convert KPI results into an actionable "gap-to-target" view that executives use:
      - How much additional HQLA (USD) is needed to meet LCR and Survival targets?
      - Which constraint is binding today (LCR vs. Survival)?

    PARAMS:
      kpis: dict
        Expected keys:
          'hqla': float (USD) — stock of High-Quality Liquid Assets
          'worst_30d_outflow': float (USD) — 30-day net outflow for LCR denominator
          'peak_cumulative_outflow': float (USD) — worst cumulative outflow over the horizon
          'survival_days': int — days until cash runs out (given HQLA and modeled flows)
          'lcr': float — LCR ratio (e.g., 1.18 = 118%)
      lcr_target: float
        Target LCR ratio (e.g., 1.30 for a 130% management buffer).
      survival_target_days: int
        Target survival horizon in days (e.g., 180 days internal limit/management target).

    RETURNS:
      dict summarizing:
        - targets (ratios/days),
        - additive HQLA needed to meet each target,
        - which metric is binding,
        - the binding gap in USD.

    RATIONALE:
      - LCR gap: ensure HQLA >= target_ratio × 30d outflow.
      - Survival gap: ensure HQLA >= peak cumulative outflow over chosen horizon.
      - Binding rule prioritizes the metric currently missing its target.
    """
    # Pull inputs with safe defaults to prevent KeyError on malformed KPI dicts.
    hqla = float(kpis.get("hqla", 0.0))
    worst_30 = float(kpis.get("worst_30d_outflow", 0.0))
    survival_days = int(kpis.get("survival_days", survival_target_days))
    peak_cum = float(kpis.get("peak_cumulative_outflow", 0.0))

    # HQLA shortfall to hit LCR target. If negative, no additional HQLA is needed for LCR.
    lcr_gap = max(0.0, lcr_target * worst_30 - hqla)

    # Survival constraint: HQLA must at least cover the maximum funding hole over the horizon.
    survival_gap = max(0.0, peak_cum - hqla)

    # Decide which constraint is binding for communication & action.
    lcr_ratio = float(kpis.get("lcr", 0.0))
    if lcr_ratio < lcr_target:
        binding = "lcr"
        binding_gap = lcr_gap
    elif survival_days < survival_target_days:
        binding = "survival"
        binding_gap = survival_gap
    else:
        binding = "none"
        binding_gap = 0.0

    # Rounded for human-friendly reporting; keep raw precision upstream if needed.
    return {
        "targets": {
            "lcr_target_ratio": lcr_target,
            "survival_target_days": survival_target_days
        },
        "addl_hqla_needed_for_lcr_target_usd": round(lcr_gap, 2),
        "addl_hqla_needed_for_survival_target_usd": round(survival_gap, 2),
        "binding_metric": binding,
        "binding_gap_usd": round(binding_gap, 2)
    }


def build_effects_summary(base_kpis, kpis, behavior, proxies):
    """
    WHY:
      Provide an executive-friendly "what it will do" summary for each scenario:
      deposit runoff $, wholesale non-roll $, expected margin calls, KPI deltas, and
      a plain-language string to embed into packs.

    PARAMS:
      base_kpis: dict — baseline KPIs for delta computations.
      kpis: dict — scenario KPIs after impacts + behavior.
      behavior: dict — behavior parameters (e.g., deposit_runoff_30d_pct, notroll_prob_*).
      proxies: dict — size proxies from portfolio (deposit_base, wholesale_base,
                       ir_notionals, fx_notionals, etc.) for quick magnitude estimates.

    RETURNS:
      dict with numeric summaries and a preformatted English sentence for slides.

    RATIONALE:
      - Stakeholders want both numbers and a compact narrative.
      - Uses proxies to avoid re-running heavy models for communication-only metrics.
    """
    # Estimate 30-day deposit runoff in absolute USD from base × runoff %.
    dep_runoff_amt = proxies["deposit_base"] * behavior["deposit_runoff_30d_pct"]

    # Approximate wholesale not-roll over 0–90d, split into <30d and 30–90d buckets.
    wholesale_30 = proxies["wholesale_base"] * behavior["notroll_prob_30d"] * 0.5
    wholesale_90 = proxies["wholesale_base"] * max(0.0, (behavior["notroll_prob_90d"] - behavior["notroll_prob_30d"])) * 0.5
    wholesale_notroll_total = wholesale_30 + wholesale_90

    # Margin calls stub: scale by aggregate notionals × scenario margin factor.
    margin_calls = 0.001 * (proxies["ir_notionals"] + proxies["fx_notionals"]) * behavior["margin_factor"]

    # KPI deltas relative to the baseline for direction & magnitude.
    delta_lcr_pp = (kpis["lcr"] - base_kpis["lcr"]) * 100.0
    delta_survival = kpis["survival_days"] - base_kpis["survival_days"]

    # Package both machine-readable numbers and human-readable text.
    return {
        "30d_deposit_outflow_usd": round(dep_runoff_amt, 2),
        "0_90d_wholesale_notroll_usd": round(wholesale_notroll_total, 2),
        "expected_margin_calls_usd": round(margin_calls, 2),
        "worst_30d_outflow_usd": round(kpis["worst_30d_outflow"], 2),
        "peak_cumulative_outflow_usd": round(kpis["peak_cumulative_outflow"], 2),
        "delta_lcr_percentage_points": round(delta_lcr_pp, 1),
        "delta_survival_days": int(delta_survival),
        "plain_language": (
            "Assumes {d:.1f}% deposit run-off, {n30:.0f}% wholesale not-roll by 30d and {n90:.0f}% by 90d, "
            "margin factor {m:.1f}. LCR {dl:+.1f}pp, survival {ds:+d} days vs baseline."
        ).format(
            d=behavior["deposit_runoff_30d_pct"]*100.0,   # Convert to %
            n30=behavior["notroll_prob_30d"]*100.0,       # Convert to %
            n90=behavior["notroll_prob_90d"]*100.0,       # Convert to %
            m=behavior["margin_factor"],                  # Factor as-is
            dl=delta_lcr_pp,                              # LCR delta in pp
            ds=int(delta_survival),                       # Survival delta in days
        )
    }


def main():
    """
    WHY:
      Orchestrate the entire run: data load, AI scenario generation, deterministic
      modeling, KPI computation, gaps-to-targets, and reporting.

    SIDE EFFECTS:
      - Prints a human-readable JSON summary to stdout.
      - Writes a machine-readable JSON artifact in ./out for auditability/re-runs.

    AUDITABILITY:
      - 'as_of' date stamps results to link with market data snapshots and approval packs.
      - The JSON output can be versioned and attached to governance workflows.
    """
    # Stamp each run with a date (YYYY-MM-DD). In production, consider a timezone-aware timestamp.
    as_of = datetime.today().strftime("%Y-%m-%d")

    # Input path for the liquidity profile. Kept near top for quick demo edits.
    liquidity_file = Path("sample_portfolio.json")

    # Fail fast with a clear message if the demo input is missing.
    if not liquidity_file.exists():
        raise FileNotFoundError(f"Missing {liquidity_file}")

    # --------------- Load data + macro context ---------------
    portfolio = load_liquidity_profile(str(liquidity_file))  # Structured positions & attributes.
    macro_data = fetch_macro_data()                          # Minimal macro state for AI to react to.

    # --------------- AI scenario generation ---------------
    # WHY: Leverage AI to propose macro shocks, instrument impacts, and rationale tailored to current macro.
    scenarios_out = generate_ai_scenario(macro_data, portfolio)  # External agent call.
    # Robustly accept both {"scenarios":[...]} and bare {...} for single-scenario returns.
    scenarios = scenarios_out.get("scenarios", [scenarios_out])

    # --------------- Baseline (deterministic reference) ---------------
    # WHY: Provide a classical, rules-only benchmark to compare AI-enhanced outcomes against.
    hqla = compute_hqla_eligible(portfolio)  # Apply eligibility rules to compute HQLA stock (USD).
    base_contract = roll_daily_contractual_flows_detail(
        portfolio_to_flows(portfolio),       # Transform positions into CFs, then into daily buckets.
        as_of                                # Anchor horizon relative to this run date.
    )
    base_kpis = compute_kpis_from_daily_detail(
        base_contract,                       # Deterministic daily CFs (no behavior or impacts).
        as_of,
        hqla                                 # Pass HQLA so LCR/survival are computed consistently.
    )

    # Quantify how far the baseline is from management targets.
    base_gaps = compute_gap_to_targets(base_kpis, lcr_target=1.30, survival_target_days=180)

    # --------------- Size proxies & macro narrative ---------------
    proxies = compute_size_proxies(portfolio)  # Deposit base, wholesale base, notionals, etc.
    macro_narrative = ai_macro_view(macro_data)  # Executive-readable 3–5 line macro brief.

    # Initialize the top-level result container (human & machine readable).
    results = {
        "as_of": as_of,
        "macro_environment": macro_narrative,
        "baseline_kpis": base_kpis,
        "baseline_gaps_to_targets": base_gaps,
        "scenarios": []  # List to be populated per scenario.
    }

    # --------------- Scenario loop ---------------
    for scn in scenarios:
        # Required metadata with defaults for robustness in public demos.
        sev = scn.get("severity", "base")  # e.g., 'mild', 'base', 'severe'.

        # 1) Macro shocks -> behavior parameters (WHY: make behavior traceable to macro conditions).
        beh = scenario_to_behavior_params(scn.get("macro_shocks", {}), sev)

        # 2) Apply AI-proposed instrument impacts to contractual CFs *before* behavior overlays.
        #    WHY: trades and facility tweaks change the deterministic path first (then behavior compounds).
        impacted_cfs = apply_instrument_impacts_to_cashflows(
            portfolio_to_flows(portfolio),         # Start from raw contractual CFs.
            portfolio,                             # Need instrument metadata to map impacts correctly.
            scn.get("instrument_impacts", []),     # List of impacts (e.g., issue CP, draw lines, swap in HQLA).
            as_of                                  # Anchor any timing logic to run date.
        )
        impacted_contract = roll_daily_contractual_flows_detail(impacted_cfs, as_of)

        # 3) Behavioral overlays (WHY: capture customer/wholesale behaviors under stress).
        behavior_series = estimate_behavioral_series_detail(portfolio, as_of, beh)

        # 4) Combine deterministic and behavioral cashflows, then compute KPIs.
        combined = combine_daily_detail(impacted_contract, behavior_series)
        kpis = compute_kpis_from_daily_detail(combined, as_of, hqla)
        gaps = compute_gap_to_targets(kpis, lcr_target=1.30, survival_target_days=180)

        # 5) Communicate impacts & add AI narrative (WHY: transparency for management & regulators).
        effects = build_effects_summary(base_kpis, kpis, beh, proxies)
        ai_note = ai_explain_scenario(
            {
                "severity": sev,
                "scenario_name": scn.get("scenario_name",""),
                "macro_shocks": scn.get("macro_shocks",{}),
                "behavior_params": beh,
                "instrument_impacts": scn.get("instrument_impacts",[]),
            },
            kpis  # Provide actual results so the narrative reflects facts, not guesses.
        )

        # Append a self-contained record for this scenario to results.
        results["scenarios"].append({
            "severity": sev,
            "scenario_name": scn.get("scenario_name", ""),
            "macro_shocks": scn.get("macro_shocks", {}),
            "behavior_params": beh,
            "kpis": kpis,
            "gap_to_targets": gaps,
            "what_it_will_do": effects,
            "ai_note": ai_note,
            "rationale": scn.get("rationale", {})  # Keep AI reasoning trace for audit.
        })

    # --------------- Output (human + machine) ---------------
    print(f"=== Behavioral Scenario Run (as-of {as_of}) ===")  # Console banner for clarity.
    print(json.dumps(results, indent=2))                    # Pretty JSON for readability in demos.

    # Persist artifacts in ./out for re-use in decks, audits, and regression tests.
    outdir = Path("out"); outdir.mkdir(exist_ok=True)       # Ensure output directory exists.
    with open(outdir / f"scenario_run_{as_of}.json", "w") as f:
        json.dump(results, f, indent=2)                     # Write canonical results to file.

# Python entrypoint guard to allow import without side effects and CLI usage.
if __name__ == "__main__":
    main()
