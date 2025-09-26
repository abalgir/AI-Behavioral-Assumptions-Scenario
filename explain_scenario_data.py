#!/usr/bin/env python3
"""
Explain your liquidity scenario JSON in plain, executive-ready language.
- No changes to your engine required.
- Reads:  out/scenario_run_YYYY-MM-DD.json
- Writes: out/ALCO_explainer_YYYY-MM-DD.md
"""

import os, json, sys
from pathlib import Path
from datetime import datetime
from dotenv import load_dotenv

# --- LLM client (LangChain OpenAI) ---
try:
    from langchain_openai import ChatOpenAI
except Exception as e:
    print("Missing dependency. Install with: pip install langchain-openai python-dotenv")
    raise

load_dotenv()
API_KEY = os.getenv("OPENAI_API_KEY")
if not API_KEY:
    print("ERROR: OPENAI_API_KEY not found. Put it in a .env file or env var.")
    sys.exit(1)

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.2)

# ---------- helpers ----------
def load_latest_json(out_dir="out"):
    p = Path(out_dir)
    if not p.exists():
        raise FileNotFoundError(f"Directory not found: {out_dir}")
    # Prefer a file named like scenario_run_YYYY-MM-DD.json; else last modified
    candidates = sorted(p.glob("scenario_run_*.json"), key=lambda x: x.stat().st_mtime, reverse=True)
    if not candidates:
        raise FileNotFoundError("No scenario_run_*.json files found in ./out")
    return candidates[0]

def format_currency(x):
    try:
        return f"${x:,.0f}"
    except Exception:
        return str(x)

def extract_facts(data: dict) -> dict:
    """Build a facts dictionary that the LLM will be forced to reference."""
    as_of = data.get("as_of", "")
    base = data.get("baseline_kpis", {})
    base_gaps = data.get("baseline_gaps_to_targets", {})
    scens = data.get("scenarios", [])

    def scen_row(s):
        name = s.get("scenario_name", s.get("severity","")).strip() or s.get("severity","").title()
        k = s.get("kpis", {})
        gaps = s.get("gap_to_targets", {})
        what = s.get("what_it_will_do", {})
        return {
            "name": name,
            "severity": s.get("severity",""),
            "lcr": k.get("lcr"),
            "survival_days": k.get("survival_days"),
            "hqla": k.get("hqla"),
            "worst_30d_outflow": k.get("worst_30d_outflow"),
            "peak_cumulative_outflow": k.get("peak_cumulative_outflow"),
            "binding_metric": gaps.get("binding_metric"),
            "binding_gap_usd": gaps.get("binding_gap_usd"),
            "deposit_runoff_30d_usd": what.get("30d_deposit_outflow_usd"),
            "wholesale_notroll_0_90d_usd": what.get("0_90d_wholesale_notroll_usd"),
            "margin_days_1_7_usd": what.get("expected_margin_calls_usd"),
        }

    scen_rows = [scen_row(s) for s in scens]

    return {
        "as_of": as_of,
        "baseline": {
            "hqla": base.get("hqla"),
            "worst_30d_outflow": base.get("worst_30d_outflow"),
            "lcr": base.get("lcr"),
            "survival_days": base.get("survival_days"),
            "peak_cumulative_outflow": base.get("peak_cumulative_outflow"),
            "binding_metric": base_gaps.get("binding_metric") if base_gaps else None,
            "binding_gap_usd": base_gaps.get("binding_gap_usd") if base_gaps else None,
        },
        "scenarios": scen_rows,
    }

def facts_to_markdown(f: dict) -> str:
    """Render a tight, non-debatable facts section the model must not change."""
    base = f["baseline"]
    lines = []
    lines.append(f"**As of:** {f['as_of']}")
    lines.append("")
    lines.append("**Baseline (contractual + HQLA):**")
    lines.append(f"- HQLA: {format_currency(base['hqla'])}")
    lines.append(f"- Worst 30d outflow: {format_currency(base['worst_30d_outflow'])}")
    lines.append(f"- LCR: {round(base['lcr']*100,1)}%")
    lines.append(f"- Survival days: {base['survival_days']}")
    lines.append(f"- Peak cumulative outflow (180d): {format_currency(base['peak_cumulative_outflow'])}")
    if base.get("binding_metric"):
        lines.append(f"- Binding: {base['binding_metric']} | Gap USD: {format_currency(base['binding_gap_usd'])}")
    lines.append("")
    lines.append("**Scenarios:**")
    lines.append("| Scenario | LCR | Survival | HQLA | Worst 30d | Peak Cum | Binding | Gap USD | 30d Deposits | 0–90d Wholesale | Margin 1–7d |")
    lines.append("|---|---:|---:|---:|---:|---:|---|---:|---:|---:|---:|")
    for s in f["scenarios"]:
        lines.append(
            f"| {s['name']} | "
            f"{round((s['lcr'] or 0)*100,1)}% | "
            f"{s['survival_days']} | "
            f"{format_currency(s['hqla'])} | "
            f"{format_currency(s['worst_30d_outflow'])} | "
            f"{format_currency(s['peak_cumulative_outflow'])} | "
            f"{s.get('binding_metric','')} | "
            f"{format_currency(s.get('binding_gap_usd',0) or 0)} | "
            f"{format_currency(s.get('deposit_runoff_30d_usd',0) or 0)} | "
            f"{format_currency(s.get('wholesale_notroll_0_90d_usd',0) or 0)} | "
            f"{format_currency(s.get('margin_days_1_7_usd',0) or 0)} |"
        )
    return "\n".join(lines)

SYSTEM_MSG = """You are a senior treasury communicator and editor.
Your job is to produce a crisp, accurate explanation of liquidity stress results for executives and engineers.
DO NOT invent numbers. ONLY use numbers given in the FACTS block provided by the user.
Be precise, concise, and structured in Markdown.
If you derive a claim (e.g., “>2x coverage”), show the simple math or reference the exact field it comes from.
Avoid jargon unless you define it in one line.
"""

USER_TEMPLATE = """Produce a clear, executive-ready explainer from the following FACTS.
Use this structure (Markdown):

# Executive summary
- 3 bullets: what changed across scenarios, which metric binds (if any), and the dollar gap to target (if any).

# What the model does (for non-treasurers)
- 4–6 lines: macro → behavior levers (deposit runoff, wholesale not-roll, margin) → dated cashflows → KPIs (HQLA, LCR, Survival).

# Readout (numbers from FACTS only)
- Baseline one-liner.
- A small paragraph per scenario: what increases, how KPIs move, and whether anything binds.

# TL;DR mapping (claims → fields)
- Show 3–5 lines mapping each headline claim to the exact field in FACTS (e.g., LCR %, Survival days, Gap USD).

# “So what?” (actions a CFO/treasurer can take)
- 3–5 bullets; keep generic (add Level-1 liquidity, extend tenors, stabilize deposits, margin readiness).
- Do not guess costs.

# FACTS (verbatim)
Include the FACTS block at the end for auditability.

FACTS:
{facts_md}
"""

def main():
    try:
        in_file = sys.argv[1] if len(sys.argv) > 1 else None
        if in_file:
            path = Path(in_file)
        else:
            path = load_latest_json("out")
        data = json.loads(Path(path).read_text())
    except Exception as e:
        print(f"Failed to load scenario JSON: {e}")
        sys.exit(1)

    facts = extract_facts(data)
    facts_md = facts_to_markdown(facts)

    messages = [
        {"role": "system", "content": SYSTEM_MSG},
        {"role": "user", "content": USER_TEMPLATE.format(facts_md=facts_md)}
    ]
    resp = llm.invoke(messages)
    content = resp.content.strip()

    as_of = data.get("as_of", datetime.today().strftime("%Y-%m-%d"))
    out_dir = Path("out"); out_dir.mkdir(exist_ok=True)
    out_path = out_dir / f"ALCO_explainer_{as_of}.md"
    out_path.write_text(content, encoding="utf-8")

    print(f"Wrote: {out_path.resolve()}")

if __name__ == "__main__":
    main()
