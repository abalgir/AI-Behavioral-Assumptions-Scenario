# ai_narrative.py
# -*- coding: utf-8 -*-
"""
AI Narratives for Treasury Liquidity Scenarios — Fully Commented Public Edition

Purpose
-------
This module produces *executive-grade narratives* around:
  1) The current macro environment (for ALCO packs), and
  2) Per-scenario liquidity results (headline + narrative + table notes).

Why it matters
--------------
Traditional stress-testing outputs are numeric and hard to digest quickly.
For decision-speed at ALCO / Treasury Committee, we pair KPIs with concise
narratives that:
  - Interpret policy (Fed funds), term structure (US10Y), volatility (VIX),
    and credit tone (IG/HY spreads), with explicit liquidity/funding angles.
  - Explain how macro shocks -> instrument actions -> LCR / HQLA / survival.

Design notes
------------
- Uses an LLM via LangChain for consistent tone and formatting.
- Enforces JSON-only returns (where applicable) for machine-readability.
- Strips markdown/code fences defensively, as some models still add them.
- Keeps temperature modest for reproducibility (0.4).

Security/Operations
-------------------
- Requires OPENAI_API_KEY provided via environment (.env supported).
- Fails fast if the key is missing to avoid confusing runtime errors.

Dependencies
------------
- python-dotenv: load_dotenv() to read local .env in development.
- langchain_openai.ChatOpenAI: thin wrapper around OpenAI Chat Completions.
- langchain.prompts.ChatPromptTemplate: clean prompt assembly.

"""

import os      # Environment variable access (API key).
import json    # JSON serialization for prompt inputs and parsing model outputs.
import re      # Robust removal of markdown/code fences from model responses.

from dotenv import load_dotenv                 # Developer-friendly .env support.
from langchain_openai import ChatOpenAI        # OpenAI chat model client via LangChain.
from langchain.prompts import ChatPromptTemplate  # Templated prompt construction.

# --- Environment bootstrap ---
load_dotenv()  # Allow local development with a .env file containing OPENAI_API_KEY.

# Fail fast with a clear error to prevent opaque 401s later.
if not os.getenv("OPENAI_API_KEY"):
    raise EnvironmentError("Missing OPENAI_API_KEY")

# Single client instance reused by helper functions.
# Model: gpt-4o-mini gives strong instruction-following at reasonable cost/latency.
# Temperature: 0.4 adds slight variability while keeping narratives stable.
_llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.4)


def _strip_fences(t: str) -> str:
    """
    Remove optional markdown fences from an LLM response.

    WHY:
      Even with "no markdown" instructions, some models return fenced blocks:
        ```json
        { ... }
        ```
      Downstream JSON parsing / string post-processing is simpler without these.

    PARAMS:
      t: Raw string returned by the LLM.

    RETURNS:
      String with leading "```json"/"```" and trailing "```" removed if present.
      If no fences are detected, returns the input trimmed.

    NOTES:
      - Case-insensitive match for "json" after opening backticks.
      - Does not attempt schema validation; that happens at the caller.
    """
    s = t.strip()
    if s.startswith("```"):
        # Drop the opening fence (with optional 'json' tag) and any leading whitespace.
        s = re.sub(r"^```(?:json)?", "", s, flags=re.I).strip()
        # Drop a trailing fence if present.
        if s.endswith("```"):
            s = s[:-3].strip()
    return s


def ai_macro_view(macro_data: dict) -> str:
    """
    Produce a 4–6 sentence ALCO cover paragraph interpreting the macro state
    with explicit liquidity/funding implications.

    INPUT:
      macro_data (dict): Expected fields used by the prompt include:
        - fed_funds_rate (policy stance)
        - us10y_yield (term structure anchor)
        - vix (market volatility proxy)
        - credit_spreads_baa / credit_spreads_hy (IG/HY tone)
        - Optional FX keys (function instructs model to include FX only if present)

    OUTPUT:
      str: A single paragraph (no bullets, no markdown, newline-collapsed) suitable
           for pasting into ALCO decks / management reports.

    RATIONALE:
      - Keeps executives focused on funding/liquidity angles (not just macro trivia).
      - Forces brevity and structure to reduce editing time in reporting packs.
    """
    # Prompt template keeps instructions tight and role-grounded (senior treasury officer).
    prompt = ChatPromptTemplate.from_template(
        """
You are a senior bank treasury officer writing an ALCO cover paragraph.
Write 4–6 sentences (no bullets, no markdown) interpreting policy stance (Fed funds),
term structure (US 10y), volatility (VIX), and credit tone (IG/HY spreads) with liquidity/funding implications.
Include FX only if present.
Macro JSON:
{macro}
"""
    )
    # Provide macro JSON as data (not prose) to reduce hallucinations and anchor numbers.
    msgs = prompt.format_messages(macro=json.dumps(macro_data))
    resp = _llm.invoke(msgs)
    # Normalize: remove any stray fences and collapse newlines into spaces for clean paragraph flow.
    return _strip_fences(resp.content).replace("\n", " ").strip()


def ai_explain_scenario(scenario: dict, kpis: dict) -> dict:
    """
    Generate an executive-friendly explanation for a single scenario.

    INPUTS:
      scenario (dict): Should include fields like 'severity', 'scenario_name',
                       'macro_shocks', 'behavior_params', 'instrument_impacts'.
      kpis (dict): Scenario KPI results (e.g., 'lcr', 'hqla', 'survival_days',
                   'worst_30d_outflow', etc.).

    OUTPUT (dict):
      {
        "headline":    str  # ≤ 20 words; the key liquidity takeaway.
        "narrative":   str  # 1–2 short paragraphs linking macro -> actions -> LCR/HQLA/survival.
        "table_notes": str  # 2–3 lines guiding how to read the KPI table (e.g., inflow caps, survival construct).
      }

    GUARANTEES & FALLBACKS:
      - The prompt instructs the model to return ONLY JSON with the three fields.
      - If parsing fails, a safe default object is returned to avoid breaking pipelines.

    RATIONALE:
      - Headline aids slide titling and takeaways.
      - Narrative ties macro, instrument actions, and KPI movement for auditability.
      - Table notes standardize interpretation (e.g., LCR inflow cap, survival definition).
    """
    prompt = ChatPromptTemplate.from_template(
        """
Return ONLY JSON with fields: headline, narrative, table_notes.
- headline: ≤ 20 words, key liquidity takeaway.
- narrative: 1–2 short paragraphs linking macro shocks -> actions -> LCR/HQLA/survival.
- table_notes: 2–3 lines on reading the metrics (inflow cap, survival construct).
Scenario:
{scenario}
KPIs:
{kpis}
"""
    )
    # Send both scenario and KPIs as JSON strings to keep the model grounded in actual numbers.
    msgs = prompt.format_messages(scenario=json.dumps(scenario), kpis=json.dumps(kpis))
    resp = _llm.invoke(msgs)

    # Clean any accidental markdown fences and attempt JSON parse.
    txt = _strip_fences(resp.content)
    try:
        obj = json.loads(txt)
    except Exception:
        # Safe fallback to ensure downstream rendering never breaks.
        obj = {
            "headline": "Scenario: key liquidity impacts",
            "narrative": (
                "Stress concentrates outflows, compressing LCR and survival. "
                "Actions reflect funding frictions."
            ),
            "table_notes": (
                "LCR uses worst 30d net outflow with 75% inflow cap. "
                "Survival = earliest day cumulative net outflow exceeds HQLA."
            )
        }

    # Final polish: normalize whitespace; keep outputs paste-ready for slides/docs.
    for k in ("headline", "narrative", "table_notes"):
        obj[k] = obj.get(k, "").replace("\n", " ").strip()
    return obj
