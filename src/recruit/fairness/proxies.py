"""Demographic proxies from observable data.

WHY proxies, not real labels:
- The HF resume-job-description-fit corpus has no demographic annotations.
- The fairness audit therefore relies on documented PROXIES — best-effort
  inferences that approximate sensitive attributes, with their limitations
  named.

VIVA: "Aren't proxies themselves biased?"
- Yes. The name→gender mapper inherits Western-name bias; many names are
  ambiguous or out-of-vocabulary. The thesis discusses this explicitly:
  audit findings are *signals*, not certainty. The infrastructure
  generalises trivially to real demographic data when available.
"""

from __future__ import annotations

import re

# Two compact name→gender lists. Not exhaustive — calibrated for *signal*,
# not coverage. Defensible because the audit reports coverage explicitly
# ("inferred for N of 477 résumés").
_MALE_NAMES: frozenset[str] = frozenset(n.lower() for n in {
    "James", "John", "Robert", "Michael", "William", "David", "Richard",
    "Joseph", "Thomas", "Charles", "Christopher", "Daniel", "Matthew",
    "Anthony", "Mark", "Donald", "Steven", "Paul", "Andrew", "Joshua",
    "Kenneth", "Kevin", "Brian", "George", "Edward", "Ronald", "Timothy",
    "Jason", "Jeffrey", "Ryan", "Jacob", "Gary", "Nicholas", "Eric",
    "Jonathan", "Stephen", "Larry", "Justin", "Scott", "Brandon", "Frank",
    "Benjamin", "Gregory", "Samuel", "Raymond", "Patrick", "Alexander",
    "Jack", "Dennis", "Jerry", "Tyler", "Aaron", "Henry", "Douglas",
    "Peter", "Jose", "Adam", "Nathan", "Zachary", "Walter", "Harold",
    "Kyle", "Carl", "Arthur", "Gerald", "Roger", "Keith", "Jeremy",
    "Lawrence", "Sean", "Christian", "Ethan", "Albert", "Vikram", "Karthik",
    "Rohit", "Suresh", "Pradeep", "Kiran", "Ankit", "Rahul", "Amit",
})

_FEMALE_NAMES: frozenset[str] = frozenset(n.lower() for n in {
    "Mary", "Patricia", "Jennifer", "Linda", "Elizabeth", "Barbara",
    "Susan", "Jessica", "Sarah", "Karen", "Lisa", "Nancy", "Betty",
    "Margaret", "Sandra", "Ashley", "Kimberly", "Emily", "Donna",
    "Michelle", "Carol", "Amanda", "Melissa", "Deborah", "Stephanie",
    "Rebecca", "Laura", "Sharon", "Cynthia", "Kathleen", "Amy", "Shirley",
    "Angela", "Helen", "Anna", "Brenda", "Pamela", "Nicole", "Samantha",
    "Katherine", "Christine", "Catherine", "Debra", "Rachel", "Carolyn",
    "Janet", "Maria", "Heather", "Diane", "Julie", "Joyce", "Victoria",
    "Kelly", "Christina", "Joan", "Evelyn", "Lauren", "Judith", "Olivia",
    "Frances", "Martha", "Cheryl", "Megan", "Andrea", "Hannah", "Jacqueline",
    "Ann", "Gloria", "Jean", "Kathryn", "Alice", "Teresa", "Sara", "Janice",
    "Doris", "Madison", "Julia", "Grace", "Judy", "Theresa", "Beverly",
    "Priya", "Anjali", "Meera", "Harshitha", "Asha", "Kavya", "Lakshmi",
    "Pooja", "Sneha", "Ananya", "Riya",
})


_FIRST_NAME_PATTERN = re.compile(r"\b([A-Z][a-z]{2,})\b")


def infer_gender_from_text(text: str) -> str | None:
    """Look for a likely first-name in the first ~200 chars of the résumé.

    Returns 'm' / 'f' / None.
    Limitations (documented in thesis): Western-name bias, miss on
    ambiguous names, fail entirely on transliterated names not in the lists.
    """
    head = text[:200]
    for token in _FIRST_NAME_PATTERN.findall(head):
        low = token.lower()
        if low in _MALE_NAMES:
            return "m"
        if low in _FEMALE_NAMES:
            return "f"
    return None


_COUNTRY_NEEDLES: list[tuple[str, str]] = [
    ("india", "IN"), ("bengaluru", "IN"), ("bangalore", "IN"),
    ("hyderabad", "IN"), ("mumbai", "IN"), ("delhi", "IN"),
    ("united states", "US"), ("u.s.", "US"), ("california", "US"),
    ("new york", "US"), ("texas", "US"), ("florida", "US"),
    ("united kingdom", "UK"), ("london", "UK"),
    ("germany", "DE"), ("singapore", "SG"),
]


def infer_country_from_text(text: str) -> str:
    """Best-effort country code; defaults to 'other' when nothing matches."""
    low = text.lower()
    for needle, code in _COUNTRY_NEEDLES:
        if needle in low:
            return code
    return "other"
