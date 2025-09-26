# portfolio_engine.py
# -*- coding: utf-8 -*-
"""
Deterministic Liquidity Engine — Fully Commented Public Edition

Purpose
-------
This module provides the *deterministic* leg of your liquidity stress framework.
It converts a portfolio into daily cashflows, applies Basel-style HQLA rules,
overlays behavior (deposit runoffs, wholesale not-roll, margin calls),
and computes KPIs (LCR, survival days, peak cumulative outflow).

Why this matters
----------------
Traditional stress programs rely on deterministic mechanics (eligibility rules,
haircuts, inflow caps). Paired with AI-generated scenarios (macro -> behavior
and instrument impacts), these functions create an auditable baseline to quantify
how AI proposals change cashflows, KPIs, and gap-to-targets.

Data expectations
-----------------
`portfolio` (dict) commonly includes:
- "intraday_liquidity": {"reserve": <float USD>}  # reserve balances treated as Level 1 HQLA
- "liquidity_profile": [ { instrument records ... } ]
    Required fields by function:
      * compute_hqla_eligible: "type", "hql_level", "notional"
      * compute_size_proxies:  "type", "notional"
      * estimate_behavioral_series_detail: "type", "notional", optional "stable_funding_factor"
- "cashflows" or "liquidity_profile_cashflows":
    List of cashflow dicts: {"instrument_id", "type", "currency", "date" (ISO), "amount" (±USD)}

KPI conventions
---------------
- Inflows are positive (+), outflows negative (-) in raw cashflows.
- Engine stores daily buckets as {'in': <sum>, 'out': <sum>} with both >= 0.
- LCR denominator applies a 75% inflow cap to worst 30-day net outflow window.
- Survival days: earliest day when cumulative (out - in) exceeds HQLA.

Compliance note
---------------
Haircuts and caps mirror Basel LCR logic: 15% haircut for Level 2A, 50% for Level 2B,
Level 2B ≤ 15% of total HQLA, and total Level 2 (2A+2B) ≤ 40% of HQLA.
"""


from collections import defaultdict  # Aggregates daily inflow/outflow without predefining all dates.
from datetime import datetime, timedelta  # Date math for rolling horizons and parsing ISO timestamps.


# ---------- Type sets ----------
# WHY: Centralized product taxonomy allows concise membership checks across functions.
ASSET_TYPES = {"bond", "mortgage_backed_security", "loan"}
LIAB_TYPES  = {
    "certificate_of_deposit", "commercial_paper", "repo", "interbank_borrowing", "fed_funds",
    "fed_discount_window", "corporate_deposits", "retail_deposits", "sme_deposits"
}
DERIV_TYPES = {"interest_rate_swap", "futures", "fx_forward", "cross_currency_swap"}


def _dt(s):
    """
    Parse input into a `datetime`.

    WHY:
      Functions accept either datetime objects or ISO8601 strings; this helper normalizes.

    PARAMS:
      s: datetime | str — already a datetime or an ISO-like string (YYYY-MM-DD or full ISO).

    RETURNS:
      datetime: parsed value.

    NOTE:
      Using `datetime.fromisoformat` keeps parsing strict and reproducible.
    """
    if isinstance(s, datetime):
        return s
    return datetime.fromisoformat(str(s))


# ---------- HQLA with haircuts & caps ----------
def compute_hqla_eligible(portfolio: dict) -> float:
    """
    Compute HQLA stock after Basel-style haircuts and caps.

    LOGIC:
      - Start with Level 1 from reserve balances (intraday_liquidity.reserve).
      - Scan instruments tagged with HQLA levels (assets or non-liability tagged instruments).
      - Apply haircuts:
          Level 2A: 15% haircut (×0.85)
          Level 2B: 50% haircut (×0.50)
      - Enforce caps:
          Level 2B ≤ 15% of total HQLA (post-haircut)
          (Level 2A + Level 2B) ≤ 40% of total HQLA

    PARAMS:
      portfolio (dict): expects keys:
        - "intraday_liquidity": {"reserve": float}
        - "liquidity_profile": list of instruments with fields:
            "type", "hql_level" ("Level 1"|"Level 2A"|"Level 2B"), "notional" (USD)

    RETURNS:
      float: final HQLA amount (USD) after haircuts and caps.
    """
    # Initialize Level 1 with reserve balances; treat missing as 0.
    l1 = float(portfolio.get("intraday_liquidity", {}).get("reserve", 0.0))
    l2a = 0.0
    l2b = 0.0

    # Walk through instruments to accumulate HQLA contributions.
    for inst in portfolio.get("liquidity_profile", []):
        t = (inst.get("type") or "").lower()         # Instrument type (asset/liability/etc.)
        lvl = (inst.get("hql_level") or "").strip().lower()  # HQLA tag, normalized.
        notional = float(inst.get("notional", 0) or 0)

        # Assets count by default; liabilities should not count even if tagged.
        is_asset_like = t in ASSET_TYPES
        has_hqla_tag_not_liability = bool(lvl) and (t not in LIAB_TYPES)

        # Skip zero/negative notionals and non-eligible items early for performance.
        if notional <= 0:
            continue
        if not (is_asset_like or has_hqla_tag_not_liability):
            continue

        # Apply haircuts by level.
        if lvl == "level 1":
            l1 += notional
        elif lvl == "level 2a":
            l2a += notional * 0.85  # 15% haircut
        elif lvl == "level 2b":
            l2b += notional * 0.50  # 50% haircut

    # If nothing eligible, return 0 to avoid divide-by-zero downstream.
    total = l1 + l2a + l2b
    if total <= 0:
        return 0.0

    # Cap Level 2B at 15% of total HQLA.
    cap_2b = 0.15 * total
    if l2b > cap_2b:
        l2b = cap_2b

    # Recompute total and apply aggregate Level 2 cap (2A + 2B ≤ 40% of HQLA).
    total = l1 + l2a + l2b
    cap_l2 = 0.40 * total
    if (l2a + l2b) > cap_l2:
        # Scale 2A and 2B proportionally to fit within the 40% cap.
        scale = cap_l2 / (l2a + l2b + 1e-12)  # epsilon avoids division by zero
        l2a *= scale
        l2b *= scale

    # Return non-negative final HQLA (defensive).
    return max(0.0, l1 + l2a + l2b)


def portfolio_to_flows(portfolio):
    """
    Select the raw cashflow array from the portfolio.

    WHY:
      Some sources name the key "cashflows"; others "liquidity_profile_cashflows".
      This helper normalizes without copying data.

    PARAMS:
      portfolio (dict): container of either "cashflows" or "liquidity_profile_cashflows".

    RETURNS:
      list[dict]: raw cashflow events (each with instrument_id, type, currency, date, amount).
    """
    return portfolio.get("cashflows", []) or portfolio.get("liquidity_profile_cashflows", [])


def roll_daily_contractual_flows_detail(cashflows, as_of, horizon_days=180):
    """
    Aggregate raw contractual cashflows into daily buckets over a horizon.

    CONVENTIONS:
      - Positive amounts => inflows
      - Negative amounts => outflows
      - Dates strictly after `as_of` and ≤ `as_of + horizon_days` are included.

    PARAMS:
      cashflows (list[dict]): events with {"date": ISO, "amount": ±float}
      as_of (datetime | str): anchor date for the horizon
      horizon_days (int): number of days to roll (default 180)

    RETURNS:
      dict[date -> {'in': float, 'out': float}] with both components non-negative.
    """
    asof = _dt(as_of)
    end  = asof + timedelta(days=horizon_days)

    # Default each date bucket to zeroed in/out.
    daily = defaultdict(lambda: {'in': 0.0, 'out': 0.0})

    for cf in cashflows:
        d = _dt(cf["date"])
        if asof < d <= end:
            amt = float(cf["amount"])
            if amt >= 0:
                daily[d.date()]['in']  += amt
            else:
                daily[d.date()]['out'] += -amt  # store as positive outflow
    return daily


def compute_size_proxies(portfolio):
    """
    Compute size proxies for messaging and light analytics:
      - deposit_base: CDs + retail/SME/corporate deposits (USD)
      - wholesale_base: repo, CP, interbank, fed funds/discount window (USD)
      - ir_notionals: IR-linked instruments (IRS, bonds, futures, MBS) (USD)
      - fx_notionals: FX-linked derivatives (FX forward, CCS) (USD)

    WHY:
      The AI summaries (and some heuristics) need quick magnitude estimates
      without re-running heavy models.

    PARAMS:
      portfolio (dict): expects "liquidity_profile" instruments with "type" and "notional".

    RETURNS:
      dict with four float proxies (USD).
    """
    deposit_base = wholesale_base = ir_notionals = fx_notionals = 0.0
    for inst in portfolio.get("liquidity_profile", []):
        t = inst.get("type", "")
        notional = float(inst.get("notional", 0) or 0)

        # Deposits
        if t in ("certificate_of_deposit", "retail_deposits", "sme_deposits", "corporate_deposits"):
            deposit_base += notional

        # Wholesale funding
        if t in ("repo", "commercial_paper", "interbank_borrowing", "fed_funds", "fed_discount_window"):
            wholesale_base += notional

        # Interest-rate linked notionals (used for margin estimates)
        if t in ("interest_rate_swap", "bond", "futures", "mortgage_backed_security"):
            ir_notionals += notional

        # FX-linked notionals (used for margin estimates)
        if t in ("fx_forward", "cross_currency_swap"):
            fx_notionals += notional

    return {
        "deposit_base": deposit_base,
        "wholesale_base": wholesale_base,
        "ir_notionals": ir_notionals,
        "fx_notionals": fx_notionals,
    }


def scenario_to_behavior_params(macro, severity):
    """
    Map macro shocks + severity into behavior parameters.

    INPUTS:
      macro (dict): may include keys "vix", "credit_spreads_baa", "credit_spreads_hy", "us10y_yield".
      severity (str): "mild" | "base" | "severe" (affects scaling of behaviors).

    OUTPUT:
      dict:
        - deposit_runoff_30d_pct: fraction of adjusted deposit base expected to run off in 30d
        - notroll_prob_30d: 0–1 probability of wholesale non-roll within 30d
        - notroll_prob_90d: 0–1 probability of wholesale non-roll by 90d
        - margin_factor: scalar for margin-call sizing (higher under stress)

    RATIONALE:
      - VIX & credit spreads are common stress drivers; higher values intensify runoff/not-roll.
      - Severity multiplier enforces escalating stress profile across scenarios.
    """
    vix  = float(macro.get("vix", 18))
    baa  = float(macro.get("credit_spreads_baa", 180))
    hy   = float(macro.get("credit_spreads_hy", 400))
    teny = float(macro.get("us10y_yield", 4.2))

    # Severity scaling to push behaviors up/down coherently.
    sev_mult = {"mild": 0.6, "base": 1.0, "severe": 1.6}.get(severity, 1.0)

    # Deposit runoff (basis points per unit of stress) -> bounded to [0.3%, 8%] over 30d.
    base_runoff = 0.5 + 0.01*(vix - 18) + 0.002*(baa - 180)
    dep_runoff_30d = max(0.3, min(8.0, base_runoff * sev_mult))

    # Wholesale non-roll: driven more by VIX and HY spreads; then convert to 0–1.
    base_notroll_30d = 5 + 0.6*(vix - 18) + 0.05*(hy - 400)
    notroll_30d = max(5, min(60, base_notroll_30d * sev_mult)) / 100.0
    notroll_90d = min(90, (notroll_30d * 100.0) * 1.5) / 100.0  # monotonically ≥ 30d prob

    # Margin factor increases with vol and rates; bounded for stability then severity-scaled.
    margin_factor = max(0.5, min(3.0, 0.8 + 0.04*(vix - 18) + 0.1*(teny - 4.2)))
    margin_factor *= sev_mult

    return {
        "deposit_runoff_30d_pct": dep_runoff_30d / 100.0,
        "notroll_prob_30d": notroll_30d,
        "notroll_prob_90d": notroll_90d,
        "margin_factor": margin_factor
    }


def estimate_behavioral_series_detail(portfolio, as_of, behavior, horizon_days=180):
    """
    Create a behavioral inflow/outflow series (daily buckets) given behavior params.

    STEPS:
      1) Deposits: compute an adjusted deposit base (higher sensitivity for low SFF),
         then spread a 30-day runoff evenly across 30 days as outflows.
      2) Wholesale: allocate non-roll outflows over 0–30d and 31–90d buckets.
      3) Margin calls: scale by IR + FX notionals times margin_factor over the next 7 days.

    PARAMS:
      portfolio (dict): expects "liquidity_profile" with "type", "notional",
                        and optional "stable_funding_factor" (0–1).
      as_of (datetime | str): anchor date.
      behavior (dict): from scenario_to_behavior_params().
      horizon_days (int): default 180; used for date bounds consistency.

    RETURNS:
      dict[date -> {'in': float, 'out': float}]
    """
    asof = _dt(as_of)
    daily = defaultdict(lambda: {'in': 0.0, 'out': 0.0})

    # --- 1) Deposits runoff over next 30 days ---
    dep_total = 0.0
    for inst in portfolio.get("liquidity_profile", []):
        if inst.get("type") in {"certificate_of_deposit", "retail_deposits", "sme_deposits", "corporate_deposits"}:
            n = float(inst.get("notional", 0) or 0)
            sff = float(inst.get("stable_funding_factor", 0.6))  # default SFF=0.6 if not provided
            # Adjustment increases effective base for less-stable funding (SFF ↓).
            adj = 1.0 + 0.75 * (1.0 - max(0.0, min(1.0, sff)))
            dep_total += n * adj

    runoff_amt = dep_total * behavior["deposit_runoff_30d_pct"]
    if runoff_amt > 0:
        per_day = runoff_amt / 30.0
        for i in range(1, 31):
            daily[(asof + timedelta(days=i)).date()]['out'] += per_day

    # --- 2) Wholesale non-roll across 0–30d and 31–90d windows ---
    wholesale_base = 0.0
    for inst in portfolio.get("liquidity_profile", []):
        if inst.get("type") in {"repo", "commercial_paper", "interbank_borrowing", "fed_funds", "fed_discount_window"}:
            wholesale_base += float(inst.get("notional", 0) or 0)

    out_30 = wholesale_base * behavior["notroll_prob_30d"] * 0.5
    out_90 = wholesale_base * max(0.0, behavior["notroll_prob_90d"] - behavior["notroll_prob_30d"]) * 0.5

    # Evenly distribute over the respective windows.
    for i in range(1, 31):
        daily[(asof + timedelta(days=i)).date()]['out'] += out_30 / 30.0
    for i in range(31, 91):
        daily[(asof + timedelta(days=i)).date()]['out'] += out_90 / 60.0

    # --- 3) Margin calls (IR + FX notionals × factor) spread over next 7 days ---
    ir_notionals = fx_notionals = 0.0
    for inst in portfolio.get("liquidity_profile", []):
        t = inst.get("type")
        n = float(inst.get("notional", 0) or 0)
        if t in {"interest_rate_swap", "bond", "futures", "mortgage_backed_security"}:
            ir_notionals += n
        if t in {"fx_forward", "cross_currency_swap"}:
            fx_notionals += n

    margin_total = 0.001 * (ir_notionals + fx_notionals) * behavior["margin_factor"]
    if margin_total > 0:
        per_day = margin_total / 7.0
        for i in range(1, 8):
            daily[(asof + timedelta(days=i)).date()]['out'] += per_day

    return daily


def combine_daily_detail(*series):
    """
    Sum multiple daily series into a single {'in', 'out'} map.

    WHY:
      Contractual flows, behavior overlays, and other adjustments are produced
      separately. This function merges them additively per date.

    PARAMS:
      *series: any number of dict[date -> {'in', 'out'}] maps.

    RETURNS:
      dict[date -> {'in', 'out'}]: merged totals.
    """
    out = defaultdict(lambda: {'in': 0.0, 'out': 0.0})
    for s in series:
        for d, io in s.items():
            out[d]['in']  += io.get('in', 0.0)
            out[d]['out'] += io.get('out', 0.0)
    return out


def compute_kpis_from_daily_detail(detail, as_of, hqla_base: float, horizon_days=180):
    """
    Compute liquidity KPIs from daily cashflow detail.

    KPIs:
      - worst_30d_outflow: maximum net outflow over any 30-day window with 75% inflow cap.
      - lcr: HQLA / worst_30d_outflow.
      - survival_days: earliest day cumulative net outflow exceeds HQLA (else horizon).
      - peak_cumulative_outflow: max cumulative (out - in) over the horizon.

    PARAMS:
      detail (dict): date -> {'in': float, 'out': float}
      as_of (datetime | str): anchor date
      hqla_base (float): HQLA stock (USD) to test against
      horizon_days (int): default 180

    RETURNS:
      dict with keys: 'hqla', 'worst_30d_outflow', 'lcr', 'survival_days', 'peak_cumulative_outflow'
    """
    asof = _dt(as_of)
    days = [(asof + timedelta(days=i)).date() for i in range(1, horizon_days + 1)]

    # Build aligned inflow/outflow arrays (0 if no entry for a date).
    inflow  = [detail.get(d, {}).get('in', 0.0)  for d in days]
    outflow = [detail.get(d, {}).get('out', 0.0) for d in days]

    # --- LCR worst 30-day net outflow with 75% inflow cap ---
    worst_30 = 1.0  # Initialize to positive sentinel to avoid divide-by-zero if all 0.
    for i in range(0, len(days) - 29):
        win_in  = sum(inflow[i:i+30])
        win_out = sum(outflow[i:i+30])
        capped_in = min(win_in, 0.75 * win_out)  # 75% inflow cap
        net = max(0.0, win_out - capped_in)      # net outflow cannot be negative
        worst_30 = max(worst_30, net)
    lcr = hqla_base / worst_30

    # --- Survival and peak cumulative outflow ---
    cum = 0.0
    peak = 0.0
    survival = None
    for idx in range(len(days)):
        cum += (outflow[idx] - inflow[idx])  # net outflow adds to cumulative
        peak = max(peak, cum)
        if survival is None and cum > hqla_base:
            survival = idx + 1  # convert index to day count (1-based)

    return {
        "hqla": hqla_base,
        "worst_30d_outflow": worst_30,
        "lcr": lcr,
        "survival_days": survival if survival is not None else horizon_days,
        "peak_cumulative_outflow": peak
    }


def apply_instrument_impacts_to_cashflows(cashflows, portfolio, impacts, as_of):
    """
    Apply AI scenario *instrument impacts* to the cashflow schedule.

    RULES (directional effects):
      - Liabilities (CD, CP, repo, interbank, fed_funds, discount window, deposits):
          * prepay / not_rollover / terminate -> add Outflow on impact date,
            and add an offset Inflow on original maturity (to avoid double-counting).
          * extend_maturity -> cancel original maturity (offset Inflow), and push new Outflow.
      - Assets (bonds, MBS, loans):
          * prepay / terminate -> add Inflow on impact date,
            and add an offset Outflow on original maturity.
          * extend_maturity -> cancel original maturity inflow (offset Outflow), then add new Inflow.
      - margin_call -> Outflow on impact date.
      - exercise_option -> Outflow on impact date (conservative default).

    PARAMS:
      cashflows (list[dict]): original schedule (future and possibly past events)
      portfolio (dict): used to map impact IDs to instrument metadata (type, currency)
      impacts (list[dict]): AI-proposed actions:
          {"id", "action", "date", "amount", "new_maturity" (optional)}
      as_of (datetime | str): evaluation date; only future CFs are adjusted

    RETURNS:
      list[dict]: augmented cashflow list including impact-derived entries.

    CONTROL:
      - Conservative defaults for unknown/other types: treat as outflow on impact date.
      - Only modifies *future* cashflows relative to `as_of` for offset logic.
    """
    from datetime import datetime  # Local import keeps module namespace minimal.
    asof = datetime.fromisoformat(str(as_of))

    # Map instrument id -> instrument record for type/currency lookup.
    inst_map = {i["id"]: i for i in portfolio.get("liquidity_profile", [])}

    # Build an index of existing *future* cashflows per instrument to locate first upcoming CF.
    futures = {}
    for cf in cashflows:
        d = datetime.fromisoformat(str(cf["date"]))
        if d > asof:
            futures.setdefault(cf["instrument_id"], []).append(cf)
    for k in futures:
        futures[k].sort(key=lambda x: x["date"])  # earliest first

    def first_future_cf(inst_id):
        """Return the earliest future cashflow for an instrument, if any."""
        lst = futures.get(inst_id, [])
        return lst[0] if lst else None

    # Start from a shallow copy so we append in-place without mutating caller's list.
    new_cfs = list(cashflows)

    for imp in impacts or []:
        inst_id = imp.get("id")
        action  = imp.get("action")
        date_s  = imp.get("date")
        amt     = float(imp.get("amount", 0) or 0)
        new_mat = imp.get("new_maturity")

        inst = inst_map.get(inst_id)
        if not inst or not date_s or amt <= 0:
            # Skip incomplete or non-positive impacts (defensive).
            continue

        ccy = inst.get("currency", "USD")
        typ = inst.get("type", "other").lower()

        # First future cashflow acts as the "original" leg we offset when needed.
        old_cf = first_future_cf(inst_id)
        old_amt = float(old_cf["amount"]) if old_cf else 0.0
        old_date = old_cf["date"] if old_cf else None

        def add_cf(amount, date):
            """Append a normalized cashflow entry (positive=inflow, negative=outflow)."""
            new_cfs.append({
                "instrument_id": inst_id,
                "type": typ,
                "currency": ccy,
                "date": date,
                "amount": float(amount)
            })

        # --- Margin / Option (always outflows on impact date in this conservative mapping) ---
        if action == "margin_call" or action == "exercise_option":
            add_cf(-amt, date_s)
            continue

        # --- Liabilities mapping ---
        if typ in {
            "certificate_of_deposit", "commercial_paper", "repo", "interbank_borrowing",
            "fed_funds", "fed_discount_window", "corporate_deposits", "retail_deposits", "sme_deposits"
        }:
            if action in {"prepay", "not_rollover", "terminate"}:
                # Pay early / do not roll: negative on impact date; offset the original future outflow.
                add_cf(-amt, date_s)
                if old_date:
                    add_cf(+amt, old_date)
            elif action == "extend_maturity" and new_mat:
                # Cancel original maturity (offset) and push to new date.
                if old_date:
                    add_cf(+abs(old_amt), old_date)
                add_cf(-abs(amt if amt else old_amt), new_mat)

        # --- Assets mapping ---
        elif typ in {"bond", "mortgage_backed_security", "loan"}:
            if action in {"prepay", "terminate"}:
                # Early prepayment/termination: inflow now; offset the original future inflow.
                add_cf(+amt, date_s)
                if old_date:
                    add_cf(-amt, old_date)
            elif action == "extend_maturity" and new_mat:
                # Cancel original inflow and add new inflow at extension date.
                if old_date:
                    add_cf(-abs(old_amt), old_date)
                add_cf(+abs(amt if amt else old_amt), new_mat)

        # --- Default: unknown types treated as conservative outflow on impact date ---
        else:
            add_cf(-amt, date_s)

    return new_cfs
