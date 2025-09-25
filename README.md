Behavioral Liquidity Scenario Engine
Overview
This project provides an AI-driven behavioral scenario analysis tool for bank treasury operations, designed to enhance liquidity risk management by integrating macroeconomic indicators (e.g., GDP, inflation, Federal Reserve rates) into dynamic stress scenarios. Unlike traditional static systems like Murex or Nasdaq Calypso, this model maps real-time economic events to behavioral parameters (e.g., deposit runoffs, wholesale non-rolls), generating actionable cashflow projections and key performance indicators (KPIs) such as Liquidity Coverage Ratio (LCR) and survival days. The tool outperforms conventional approaches by identifying risks earlier, as demonstrated by detecting a $6.5 billion liquidity gap in severe crisis scenarios.
Features

Dynamic Scenario Generation: Translates macroeconomic shocks (e.g., Fed funds rate to 6.0%, VIX to 35) into behavioral impacts like 1.5% deposit runoff or 40% wholesale non-roll in 30 days.
Portfolio Integration: Processes banking book positions (e.g., $80B retail deposits, $25B U.S. Treasuries) to compute contractual and behavioral cashflows.
KPI Computation: Generates LCR (e.g., 310.8% baseline to 208.6% severe), survival days (180 to 99), and gap-to-target metrics ($6.5B in severe scenarios).
Explainable AI: Provides plain-language narratives and rationales linking macro events to treasury actions, aiding C-suite decision-making.
Open-Source: Fully commented code for transparency and customization, hosted on GitHub under abalgir/behavioral_scenario.

Installation
Prerequisites

Python 3.10+
Required packages:pip install langchain_openai python-dotenv


OpenAI API key (stored in .env file as OPENAI_API_KEY).

Setup

Clone the repository:git clone https://github.com/abalgir/AI-Behavioral-Assumptions-Scenario.git
cd behavioral_scenario


Create a .env file with your OpenAI API key:echo "OPENAI_API_KEY=your-key-here" > .env


Install dependencies:pip install -r requirements.txt



Usage

Prepare Input Data:

Provide a sample_portfolio.json file with liquidity profile and cashflows (see example in repository).
Macro inputs are stubbed in fetch_macro_data(); replace with live feeds (e.g., FRED, Bloomberg) for production.


Run the Script:
python orchestrator_scenarios.py


Outputs a JSON file (out/scenario_run_YYYY-MM-DD.json) with scenario results, including KPIs and narratives.
Console displays human-readable JSON summary.


Key Files:

orchestrator_scenarios.py: Main script coordinating data load, AI scenario generation, and KPI computation.
scenario_generator.py: Generates AI-driven scenarios using LangChain and OpenAI's GPT-4o-mini.
ai_narratives.py: Produces executive-friendly narratives for macro context and scenario results.
portfolio_engine.py: Handles deterministic cashflow calculations, HQLA eligibility, and behavioral overlays.



How It Works
The tool follows a structured workflow:

Load Liquidity Profile: Imports portfolio data (e.g., $37.3B HQLA, $10B commercial paper).
Fetch Macro Inputs: Uses indicators like Fed funds rate (5.25%), VIX (18.2), or credit spreads (190 bps).
AI Scenario Generation: Proposes mild, base, and severe scenarios with macro shocks (e.g., VIX to 35) and instrument impacts (e.g., not rolling over repos).
Cashflow Generation: Converts portfolio into contractual daily cashflows using portfolio_to_flows.
Behavioral Overlays: Applies AI-derived parameters (e.g., 1.5% deposit runoff in severe case) via estimate_behavioral_series_detail.
KPI Calculation: Computes LCR, survival days, and gaps using compute_kpis_from_daily_detail.
Output: Saves results with narratives (e.g., "Severe crisis: LCR at 208.6%, $6.5B gap") to JSON and console.

Example Output
{
  "as_of": "2025-09-23",
  "baseline_kpis": {
    "hqla": 37300000000.0,
    "lcr": 3.1083333333333334,
    "survival_days": 180
  },
  "scenarios": [
    {
      "scenario_name": "Severe Financial Crisis",
      "kpis": {
        "lcr": 2.0860192306738705,
        "survival_days": 99,
        "peak_cumulative_outflow": 43769260000.0
      },
      "gap_to_targets": {
        "binding_gap_usd": 6469260000.0
      },
      "ai_note": {
        "headline": "Severe Financial Crisis: Liquidity Pressure and LCR Challenges"
      }
    }
  ]
}

Benefits

Enhanced Risk Detection: Identifies $6.5B liquidity gaps in severe scenarios, missed by static models.
Proactive Treasury Actions: Suggests strategies like adding Level-1 liquidity or extending wholesale tenors.
Regulatory Compliance: Aligns with Basel III stress testing and ECB AI guidelines.
Transparency: Open-source code with detailed comments ensures auditability.

Contributing
Contributions are welcome! Please submit pull requests or open issues for bugs, feature requests, or improvements. Ensure code follows PEP 8 style and includes comments for clarity.
License
MIT License. See LICENSE for details.
