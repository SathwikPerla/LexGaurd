"""
benchmarks.py — Hardcoded legal clause benchmarks for semantic comparison.

These represent known-good and known-predatory clause patterns across common
contract types. Agent 2 (Risk Analyzer) compares extracted clauses against
these benchmarks using Vertex AI embeddings + ChromaDB to classify whether
a clause is industry-standard or predatory.

Design decisions:
  - Hardcoded, not loaded from a database — zero runtime cost, reproducible
  - Each entry has is_predatory: bool so the risk analyzer gets a signal
  - clause_type matches ClauseType enum values exactly for consistent mapping
  - benchmark_id is stable — used as ChromaDB document ID

How to add new benchmarks:
  - Append to BENCHMARKS list below
  - Run `python -c "from core.benchmarks import BENCHMARKS; print(len(BENCHMARKS))"` to verify
  - ChromaDB will pick up new entries on next app start (ingests on init)
"""
from __future__ import annotations

from typing import List, TypedDict


class Benchmark(TypedDict):
    benchmark_id: str
    clause_type: str          # matches ClauseType enum values
    is_predatory: bool
    risk_level: str           # RED / YELLOW / GREEN
    severity_score: float     # 1.0 – 10.0
    text: str                 # representative clause text
    notes: str                # why this is standard or predatory


BENCHMARKS: List[Benchmark] = [
    # ── Non-Compete ─────────────────────────────────────────────────────────
    {
        "benchmark_id": "nc_standard_6mo",
        "clause_type": "non_compete",
        "is_predatory": False,
        "risk_level": "YELLOW",
        "severity_score": 3.5,
        "text": (
            "Employee agrees not to directly solicit Company's current clients "
            "for a period of 6 months following termination of employment."
        ),
        "notes": "6-month client non-solicitation is industry standard; courts routinely uphold this.",
    },
    {
        "benchmark_id": "nc_predatory_2yr_broad",
        "clause_type": "non_compete",
        "is_predatory": True,
        "risk_level": "RED",
        "severity_score": 9.0,
        "text": (
            "Employee agrees not to engage in any competing business activity "
            "within a 50-mile radius for 2 years following termination."
        ),
        "notes": "2-year geographic non-compete with undefined scope. Courts frequently void these.",
    },
    {
        "benchmark_id": "nc_predatory_broad_industry",
        "clause_type": "non_compete",
        "is_predatory": True,
        "risk_level": "RED",
        "severity_score": 8.5,
        "text": (
            "Employee shall not work for any company in the technology sector "
            "for 18 months following separation from employment."
        ),
        "notes": "Entire-industry restriction is almost never enforceable but creates chilling effect.",
    },
    # ── IP Transfer ──────────────────────────────────────────────────────────
    {
        "benchmark_id": "ip_standard_work_product",
        "clause_type": "ip_transfer",
        "is_predatory": False,
        "risk_level": "GREEN",
        "severity_score": 2.0,
        "text": (
            "All work product created by Employee during working hours using "
            "Company equipment and directly related to Company business is the "
            "exclusive property of Employer."
        ),
        "notes": "Standard IP assignment limited to company time and resources. Widely accepted.",
    },
    {
        "benchmark_id": "ip_predatory_all_inventions",
        "clause_type": "ip_transfer",
        "is_predatory": True,
        "risk_level": "RED",
        "severity_score": 9.5,
        "text": (
            "All inventions, discoveries, and creative works conceived by Employee "
            "during the term of employment, whether on company time or not, "
            "and whether or not related to Company business, are the sole and "
            "exclusive property of Employer."
        ),
        "notes": "Captures personal side projects on personal time. Extremely predatory. Many states limit this.",
    },
    # ── Arbitration ──────────────────────────────────────────────────────────
    {
        "benchmark_id": "arb_mutual_standard",
        "clause_type": "arbitration",
        "is_predatory": False,
        "risk_level": "YELLOW",
        "severity_score": 4.5,
        "text": (
            "Any disputes arising under this Agreement shall be resolved by binding "
            "arbitration administered by JAMS in accordance with its Employment "
            "Arbitration Rules. Costs shall be shared equally."
        ),
        "notes": "Mutual arbitration with shared costs is the industry standard. Acceptable.",
    },
    {
        "benchmark_id": "arb_predatory_one_sided",
        "clause_type": "arbitration",
        "is_predatory": True,
        "risk_level": "RED",
        "severity_score": 7.5,
        "text": (
            "Any disputes shall be resolved through binding arbitration. Employee "
            "waives all rights to class action or collective proceedings. All "
            "arbitration costs shall be borne solely by Employee."
        ),
        "notes": "Employee-pays-all plus class action waiver is one-sided and predatory.",
    },
    # ── Liability Limitation ─────────────────────────────────────────────────
    {
        "benchmark_id": "liability_standard_cap",
        "clause_type": "liability",
        "is_predatory": False,
        "risk_level": "YELLOW",
        "severity_score": 4.0,
        "text": (
            "In no event shall either party be liable for indirect, incidental, "
            "special, or consequential damages. Total liability shall not exceed "
            "the fees paid in the 12 months preceding the claim."
        ),
        "notes": "Mutual limitation with a fee-based cap is standard SaaS/vendor contract language.",
    },
    {
        "benchmark_id": "liability_predatory_total_waiver",
        "clause_type": "liability",
        "is_predatory": True,
        "risk_level": "RED",
        "severity_score": 8.0,
        "text": (
            "Company shall not be liable for any damages of any kind arising from "
            "use of the Service, including but not limited to direct, indirect, "
            "incidental, consequential, or punitive damages, even if advised of "
            "the possibility of such damages."
        ),
        "notes": "Total liability waiver with no carve-out for gross negligence is predatory.",
    },
    # ── Termination ──────────────────────────────────────────────────────────
    {
        "benchmark_id": "termination_standard_notice",
        "clause_type": "termination",
        "is_predatory": False,
        "risk_level": "GREEN",
        "severity_score": 2.0,
        "text": (
            "Either party may terminate this Agreement with 30 days written notice. "
            "Company may terminate immediately for cause."
        ),
        "notes": "30-day notice with for-cause carve-out is balanced and standard.",
    },
    {
        "benchmark_id": "termination_predatory_at_will",
        "clause_type": "termination",
        "is_predatory": True,
        "risk_level": "YELLOW",
        "severity_score": 6.0,
        "text": (
            "Employer may terminate Employee's employment at any time, with or "
            "without cause, and with or without notice, at Employer's sole discretion."
        ),
        "notes": "At-will termination is legal in most US states but combined with other clauses can be predatory.",
    },
    # ── Auto-Renewal ─────────────────────────────────────────────────────────
    {
        "benchmark_id": "autorenewal_standard",
        "clause_type": "auto_renewal",
        "is_predatory": False,
        "risk_level": "YELLOW",
        "severity_score": 3.0,
        "text": (
            "This Agreement shall automatically renew for successive one-year terms "
            "unless either party provides 30 days written notice of non-renewal "
            "prior to the end of the then-current term."
        ),
        "notes": "30-day auto-renewal with written notice is reasonable and widely accepted.",
    },
    {
        "benchmark_id": "autorenewal_predatory_hidden",
        "clause_type": "auto_renewal",
        "is_predatory": True,
        "risk_level": "RED",
        "severity_score": 7.0,
        "text": (
            "This subscription automatically renews annually. Customer must cancel "
            "at least 90 days before renewal to avoid charges. Early termination "
            "fee equals the full remaining subscription value."
        ),
        "notes": "90-day cancellation window plus full-value termination fee traps customers.",
    },
    # ── Privacy / Data Collection ─────────────────────────────────────────────
    {
        "benchmark_id": "privacy_standard_minimal",
        "clause_type": "data_collection",
        "is_predatory": False,
        "risk_level": "GREEN",
        "severity_score": 2.5,
        "text": (
            "Company collects only the personal data necessary to provide the Service. "
            "Data is not sold to third parties. Users may request deletion at any time."
        ),
        "notes": "Data minimization with deletion rights meets GDPR/CCPA baseline. Standard.",
    },
    {
        "benchmark_id": "privacy_predatory_broad_sharing",
        "clause_type": "data_collection",
        "is_predatory": True,
        "risk_level": "RED",
        "severity_score": 8.5,
        "text": (
            "By using the Service, you grant us an irrevocable, perpetual, worldwide "
            "license to collect, store, share, and monetize all data you provide, "
            "including communications, for any commercial purpose without additional notice."
        ),
        "notes": "Irrevocable perpetual monetization license is extremely predatory. Fails GDPR.",
    },
    # ── Indemnification ──────────────────────────────────────────────────────
    {
        "benchmark_id": "indemnification_mutual",
        "clause_type": "indemnification",
        "is_predatory": False,
        "risk_level": "YELLOW",
        "severity_score": 4.0,
        "text": (
            "Each party shall indemnify the other against claims arising from its "
            "own breach of this Agreement or negligence."
        ),
        "notes": "Mutual indemnification tied to breach or negligence is standard.",
    },
    {
        "benchmark_id": "indemnification_predatory_broad",
        "clause_type": "indemnification",
        "is_predatory": True,
        "risk_level": "RED",
        "severity_score": 8.0,
        "text": (
            "Customer shall defend, indemnify, and hold harmless Company from any "
            "and all claims, damages, costs, and expenses arising from Customer's "
            "use of the Service, regardless of cause."
        ),
        "notes": "One-sided indemnification regardless of cause. Customer bears all risk.",
    },
    # ── Governing Law ─────────────────────────────────────────────────────────
    {
        "benchmark_id": "governing_law_standard_delaware",
        "clause_type": "governing_law",
        "is_predatory": False,
        "risk_level": "GREEN",
        "severity_score": 2.0,
        "text": "This Agreement shall be governed by the laws of the State of Delaware.",
        "notes": "Delaware governing law is standard for US corporations. Low risk.",
    },
    {
        "benchmark_id": "governing_law_inconvenient_venue",
        "clause_type": "governing_law",
        "is_predatory": True,
        "risk_level": "YELLOW",
        "severity_score": 5.5,
        "text": (
            "This Agreement is governed by the laws of the Cayman Islands. Any "
            "disputes shall be resolved exclusively in the courts of the Cayman Islands."
        ),
        "notes": "Offshore jurisdiction chosen to limit consumer access to courts. Suspicious.",
    },
    # ── Confidentiality ──────────────────────────────────────────────────────
    {
        "benchmark_id": "confidentiality_standard",
        "clause_type": "confidentiality",
        "is_predatory": False,
        "risk_level": "GREEN",
        "severity_score": 2.5,
        "text": (
            "Each party agrees to keep confidential all non-public information "
            "received from the other party and not to disclose it to third parties "
            "without prior written consent."
        ),
        "notes": "Mutual NDA with standard carve-outs is ubiquitous and balanced.",
    },
    {
        "benchmark_id": "confidentiality_predatory_perpetual",
        "clause_type": "confidentiality",
        "is_predatory": True,
        "risk_level": "YELLOW",
        "severity_score": 6.0,
        "text": (
            "Employee's obligation of confidentiality shall survive termination of "
            "employment in perpetuity and extends to all information Employee "
            "encountered in any capacity during employment."
        ),
        "notes": "Perpetual confidentiality with unlimited scope is overbroad. Standard is 2-5 years.",
    },
]
