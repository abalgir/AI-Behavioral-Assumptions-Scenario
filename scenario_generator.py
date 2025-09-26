# -*- coding: utf-8 -*-
"""
scenario_generator.py

AI Scenario Generator (LangChain + OpenAI) — Fully Commented Public Edition

Purpose
-------
Given a bank's portfolio snapshot and a macro input stub, this module prompts an
LLM to generate three liquidity stress scenarios (mild/base/severe) with:
  - Macro shocks (rates, credit, vol, FX)
  - Instrument-level impacts (e.g., not_rollover, extend_maturity, margin_call)
  - Rationale linking macro to actions

Why this matters
----------------
Traditional liquidity stress testing often relies on fixed, static rules and a
small set of pre-baked scenarios. This module illustrates how an AI agent can:
  1) Tailor scenarios to the observed macro state and actual instrument mix
  2) Propose concrete, near-term actions that materially change cashflows
  3) Explain *why* each instrument is impacted (auditability, governance)

Data expectations
-----------------
- `macro_data` is a dict of macro indicators (see the prompt fields).
- `portfolio` is a dict that contains a top-level list `liquidity_profile`
  where each instrument is a dict with at least:
    - "id": unique instrument identifier (string)
    - "type": product type (e.g., "deposit", "repo", "commercial_paper", "irs", "ccs")
    - Optional currency keys depending on product:
        * "currency" (simple single-currency instruments), OR
        * "notional_leg1" / "notional_leg2" with nested {"currency": "..."} for CCS,
        * "notional_buy" / "notional_sell" for FX options/forwards.

Security & Ops
--------------
- The OpenAI API key is loaded from environment variables via python-dotenv.
- Model and temperature are explicitly specified for reproducibility.
"""

import json      # JSON (de)serialization for prompt payloads and LLM output parsing.
import os        # Access environment variables for API keys.
import re        # Robust extraction of JSON from LLM responses (handles fenced code blocks).

# LangChain's OpenAI chat wrapper (thin client around OpenAI Chat Completions).
from langchain_openai import ChatOpenAI
# Prompt templating: keeps prompt text clean, maintains variable insertion safely.
from langchain.prompts import ChatPromptTemplate
# Load local .env into process environment for development / demos.
from dotenv import load_dotenv


# --- Environment bootstrap ---
load_dotenv()  # WHY: Allow developers to store OPENAI_API_KEY in a local .env for convenience.
if not os.getenv("OPENAI_API_KEY"):
    # Fail fast with a clear error: avoids confusing 401s later in the LLM call.
    raise EnvironmentError("Missing OPENAI_API_KEY. Please add it to your .env file.")

# Single LLM client instance reused across calls.
# Model choice: gpt-4o-mini balances cost/latency with strong structured output fidelity.
# Temperature 0.4: slight creativity to diversify scenarios while preserving determinism.
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.4)


def _extract_json(text: str) -> str:
    """
    Extract a JSON object from an LLM response string.

    WHY:
      Despite strict "JSON only" instructions, LLMs may still wrap output in code fences
      or include brief preambles. This function defensively extracts the first JSON object.

    STRATEGY:
      1) Prefer fenced blocks labeled ```json ... ```
      2) Fallback: grab from first '{' to last '}' if present
      3) Otherwise, return the original text (caller will handle parse errors)

    PARAMS:
      text (str): raw response text from the LLM.

    RETURNS:
      str: best-effort JSON substring suitable for json.loads().

    NOTE:
      - DOTALL flag lets '.' match newlines, so we can capture multi-line JSON.
      - This is intentionally minimal; a stricter JSON schema validation can be layered above.
    """
    s = text.strip()
    # 1) Look for fenced JSON block: ```json { ... } ```
    fenced = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", s, flags=re.DOTALL)
    if fenced:
        return fenced.group(1).strip()
    # 2) Fallback: take the first curly to the last curly
    if "{" in s and "}" in s:
        start = s.find("{")
        end = s.rfind("}")
        return s[start:end+1]
    # 3) As-is (caller will attempt json.loads and report errors)
    return s


def generate_ai_scenario(macro_data, portfolio):
    """
    Generate three AI-driven liquidity stress scenarios (mild/base/severe).

    PIPELINE:
      Step 1: Extract currency universe from portfolio to build an FX shock schema
              tailored to actual currencies in play (avoids hallucinated pairs).
      Step 2: Construct FX pair symbols. Prefer USD crosses if USD is present,
              otherwise create crosses against the first currency found.
      Step 3: Build a minimal instrument list (id, type, maturity) to reduce token use
              while still giving the model enough structure to make targeted impacts.
      Step 4: Create a strict, JSON-only prompt with rules (allowed actions, severity
              scaling, near-term impact requirement) to enforce material cashflow change.
      Step 5: Invoke the LLM via LangChain and capture the response.
      Step 6: Parse JSON robustly; include raw output on failure for audit/debug.

    PARAMS:
      macro_data (dict): macro indicators used by the prompt (rates/credit/vol/FX).
      portfolio (dict): must contain key "liquidity_profile" as a list of instruments.

    RETURNS:
      dict:
        - On success: {"scenarios": [ ... three scenario dicts ... ]}
        - On parse failure: {"error": "...", "raw_output": "<model text>"}

    RISK/CONTROL NOTES:
      - The prompt enforces "near-term impact" (≤ 30 days) to ensure scenarios
        actually move cashflows in the LCR window—this differentiates AI output
        from abstract narratives and makes it decision-useful.
      - Allowed actions are whitelisted to simplify downstream mapping logic.
    """
    # -----------------------------
    # Step 1: Extract currencies (for FX schema)
    # -----------------------------
    # WHY: We only include FX pairs the portfolio *needs*, improving precision and
    #      reducing prompt size. Handles single-currency instruments and two-legged FX/CCS.
    currencies = []
    for inst in portfolio["liquidity_profile"]:
        if "currency" in inst:
            currencies.append(inst["currency"])
        elif "notional_leg1" in inst:  # e.g., Cross-Currency Swap
            currencies.append(inst["notional_leg1"]["currency"])
            currencies.append(inst["notional_leg2"]["currency"])
        elif "notional_buy" in inst:   # e.g., FX forward/option notionals
            currencies.append(inst["notional_buy"]["currency"])
            currencies.append(inst["notional_sell"]["currency"])
    currencies = list(set(currencies))  # De-duplicate to avoid redundant FX keys.

    # -----------------------------
    # Step 2: Build FX pairs relative to USD if present
    # -----------------------------
    # WHY: Most liquidity stress frameworks anchor to USD crosses when USD exists.
    #      If USD is absent, we still construct meaningful crosses among present CCYs.
    fx_pairs = []
    if "USD" in currencies:
        for cur in currencies:
            if cur != "USD":
                # Convention note: example pair formats (lowercase) as expected by downstream engines.
                # Provide a special case to keep common ticker style for USDJPY if needed.
                fx_pairs.append("usdjpy" if cur == "JPY" else f"{cur.lower()}usd")
    elif currencies:
        # No USD present: build crosses against the first seen currency as a pragmatic default.
        base = currencies[0]
        for cur in currencies[1:]:
            fx_pairs.append(f"{base.lower()}{cur.lower()}")

    # Build a JSON field schema stub for the prompt (keys only; values filled by the model).
    # Example:  "eurusd": "...",\n  "usdjpy": "...",
    fx_schema = "\n".join([f'"{pair}": "...",' for pair in fx_pairs])

    # -----------------------------
    # Step 3: Build instrument list (minimal but sufficient)
    # -----------------------------
    # WHY: Provide IDs/types/maturities so the model proposes *targeted* actions (e.g., not_rollover
    #      a specific CP line maturing in 12 days). This keeps token count low for speed/cost.
    instrument_list = [
        {"id": inst["id"], "type": inst["type"], "maturity": inst.get("maturity")}
        for inst in portfolio["liquidity_profile"]
    ]

    # -----------------------------
    # Step 4: Prompt template (strict JSON with enforcement rules)
    # -----------------------------
    # WHY: The structure and rules below are designed to:
    #   - force scenario completeness (macro + impacts + rationale)
    #   - guarantee near-term cashflow changes (≤ 30d)
    #   - scale severity coherently across mild/base/severe
    #   - keep outputs machine-readable (no markdown fences)
    prompt_template = ChatPromptTemplate.from_template(
        """
You are a senior treasury risk manager.
Given the current macroeconomic data and the bank's instrument portfolio,
generate three behavioral stress scenarios: mild, base, and severe.

- Macro data:
  {macro_data}

- Portfolio instruments (ID, type, maturity):
  {instrument_list}

OUTPUT RULES:
- Return ONLY valid JSON, no markdown fences or commentary.
- JSON must have the following structure:

{{
  "scenarios": [
    {{
      "severity": "mild",
      "scenario_name": "...",
      "macro_shocks": {{
        "fed_funds_rate": "...",
        "us10y_yield": "...",
        "vix": "...",
        "credit_spreads_baa": "...",
        "credit_spreads_hy": "...",
        "fx": {{
          {fx_schema}
        }}
      }},
      "instrument_impacts": [
        {{
          "id": "...",
          "action": "...",
          "date": "...",
          "amount": ...,
          "new_maturity": "..."
        }}
      ],
      "rationale": {{
        "instrument_id": "Explain why this instrument is impacted"
      }}
    }},
    {{
      "severity": "base",
      ...
    }},
    {{
      "severity": "severe",
      ...
    }}
  ]
}}

STRICT RULES:
1) Allowed actions ONLY:
   - "prepay"
   - "extend_maturity"
   - "not_rollover"
   - "terminate"
   - "margin_call"
   - "exercise_option"

2) Sizing:
   - If principal is known, use 20–60% fractions.
   - Otherwise, set amount ≥ 250,000,000 and ≤ 25% of wholesale base.

3) **Near-term impact requirement:** Each scenario must include ≥1 impact within the next 30 days
   that is either a wholesale liability (repo, commercial_paper, interbank_borrowing, fed_funds)
   with action "not_rollover"/"prepay", or a "margin_call". Mandatory.

4) Severity must escalate:
   - Mild: 1–2 small impacts.
   - Base: 3–4 moderate impacts.
   - Severe: multiple instruments impacted, including margin calls and repo/FX stress.

5) Each scenario must materially change the CASHFLOW profile.

6) The "rationale" must explain, linking macro shocks to each impacted instrument.
"""
    )

    # -----------------------------
    # Step 5: Run LLM (format prompt + invoke)
    # -----------------------------
    # Format macro data and instruments as pretty JSON strings so the model
    # treats them as data, not prose. Insert the dynamic FX schema.
    messages = prompt_template.format_messages(
        macro_data=json.dumps(macro_data, indent=2),
        instrument_list=json.dumps(instrument_list, indent=2),
        fx_schema=fx_schema
    )
    # Single call to the chat model; returns an AIMessage with .content (string).
    response = llm.invoke(messages)

    # -----------------------------
    # Step 6: Parse JSON safely (defensive)
    # -----------------------------
    # WHY: Even with strict instructions, failures can occur. We extract best-effort JSON,
    # try to parse, and include raw text in the error payload to aid debugging.
    content = _extract_json(response.content)
    try:
        scenario = json.loads(content)
    except Exception as e:
        # Return a structured error object rather than raising, so callers can log
        # and continue (e.g., fall back to a deterministic scenario set).
        scenario = {"error": f"Invalid JSON returned: {e}", "raw_output": response.content}
    return scenario
