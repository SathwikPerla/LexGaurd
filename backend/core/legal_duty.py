"""
legal_duty.py — detect clauses that merely restate a duty the signer already owes by law.

Examples: "You must not alter evidence or influence witnesses."

These are not negotiable and not a business risk: the signer must obey them
whether or not they sign, and asking to remove them would mean asking to be
allowed to break the law. The model sometimes scores them YELLOW and recommends
"negotiate" because the prompts treat every non-green clause as negotiable, so
this deterministic guard overrides that.

Deliberately NARROW: it only matches clear criminal-conduct prohibitions, and
only in short clauses, so a long clause that mixes such a sentence with a real
burden (indemnity, penalties, ...) is NOT auto-accepted.
"""
from __future__ import annotations

import re

# A short clause: long ones can hide other terms next to a boilerplate sentence.
_MAX_CLAUSE_CHARS = 300

# The clause must forbid something (prohibition cue), not grant or demand it.
_PROHIBITION = re.compile(
    r"\b(must not|shall not|may not|will not|cannot|can not|agrees? not to|"
    r"not to|prohibited|forbidden|refrain|never|"
    r"neither\b[^.;]{0,30}\b(may|shall|will|must)|no (party|person|one)\b)\b",
    re.IGNORECASE,
)

_DUTY_PATTERNS: tuple[re.Pattern[str], ...] = tuple(
    re.compile(p, re.IGNORECASE)
    for p in (
        # tampering with / destroying evidence
        r"\b(alter\w*|destroy\w*|delet\w*|falsif\w*|tamper\w*|conceal\w*|fabricat\w*)\b[^.;]{0,60}\bevidence\b",
        # influencing / intimidating witnesses
        r"\b(influenc\w*|intimidat\w*|coerc\w*|tamper\w*|bribe\w*|threaten\w*)\b[^.;]{0,60}\bwitness\w*",
        # obstruction of justice
        r"\bobstruct\w*\b[^.;]{0,40}\b(justice|investigation|proceeding)",
        # bribery / corruption / money laundering / fraud as prohibited conduct
        r"\b(bribe\w*|brib\w+|kickback\w*|money laundering)\b",
        r"\b(commit\w*|engag\w+ in)\b[^.;]{0,30}\b(fraud|corruption|bribery)\b",
    )
)


# If a clause also mentions any of these, it carries a real commercial burden next
# to the legal-duty sentence, so it must go through normal risk analysis.
_RISK_TERMS = re.compile(
    r"\b(indemnif\w*|liab\w*|penalt\w*|damages|fines?|forfeit\w*|waive\w*|terminat\w*|"
    r"non-?compete|arbitrat\w*|assign\w*|exclusiv\w*|pay(ment|able|s)?|fees?|"
    r"reimburs\w*|surrender\w*|release\w*|confidential\w*|governing law|renew\w*)\b",
    re.IGNORECASE,
)


def is_legal_duty_clause(text: str | None) -> bool:
    """True if the clause is a short prohibition on clearly unlawful conduct."""
    if not text:
        return False
    stripped = text.strip()
    if not stripped or len(stripped) > _MAX_CLAUSE_CHARS:
        return False
    if not _PROHIBITION.search(stripped):
        return False
    if _RISK_TERMS.search(stripped):
        return False
    return any(p.search(stripped) for p in _DUTY_PATTERNS)
