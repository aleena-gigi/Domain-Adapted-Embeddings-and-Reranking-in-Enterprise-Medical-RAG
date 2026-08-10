"""Clinical vocabulary for the anaphoric filter (Open Item #2, resolved).

**Decision: MeSH descriptor terms, not scispacy.** The spec says to choose on
install friction rather than marginal accuracy. MeSH's tree file is a single
2.7 MB public-domain download from NLM with no registration, no license
click-through, and no dependency at all. scispacy would pin an older spaCy and
a model wheel from a third-party bucket. MeSH also pays a second dividend: the
tree numbers give the corpus specialty/body-system distribution
(`eda_specialty.png`) for free, from the same file.

**The blocklist is the important part.** MeSH contains descriptors like
"Patients", "Pain", "Surgery", "Treatment Outcome" and "Patient Discharge".
Counting those as clinical entities would defeat the filter's second clause
entirely, because they are precisely the words anaphoric queries are built
from ("What was the outcome of the patient's treatment?"). GENERIC_MESH_TERMS
below is derived from the most frequent content words among queries carrying a
definite patient reference, and is frozen here so the filter is a fixed
function of its input rather than something refit per run.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from pathlib import Path

log = logging.getLogger(__name__)

MESH_YEAR = 2025
# The ASCII descriptor file (~30 MB) rather than the tree file (~3 MB): it
# carries ENTRY / PRINT ENTRY synonyms *and* MN tree numbers, so it is a single
# source of truth. Synonyms matter concretely — "Quadriparesis" and
# "Chemotherapy" are entry terms whose preferred headings are "Quadriplegia"
# and "Drug Therapy", and without them the filter drops answerable queries like
# "What was the cause of the patient's symmetrical quadriparesis?"
MESH_URL = f"https://nlmpubs.nlm.nih.gov/projects/mesh/{MESH_YEAR}/asciimesh/d{MESH_YEAR}.bin"
MESH_FILENAME = f"d{MESH_YEAR}.bin"

# MeSH top-level tree categories, used for the specialty/body-system figure.
MESH_CATEGORIES = {
    "A": "Anatomy",
    "B": "Organisms",
    "C": "Diseases",
    "D": "Chemicals & Drugs",
    "E": "Analytical/Diagnostic/Therapeutic Techniques",
    "F": "Psychiatry & Psychology",
    "G": "Phenomena & Processes",
    "H": "Disciplines & Occupations",
    "I": "Anthropology/Education/Sociology",
    "J": "Technology & Food",
    "K": "Humanities",
    "L": "Information Science",
    "M": "Named Groups",
    "N": "Health Care",
    "V": "Publication Characteristics",
    "Z": "Geographicals",
}

# Body-system view (C = Diseases, A = Anatomy) at the second tree level; used
# to describe corpus coverage breadth in paper Section 2.
DISEASE_SUBCATEGORIES = {
    "C01": "Infections",
    "C04": "Neoplasms",
    "C05": "Musculoskeletal",
    "C06": "Digestive",
    "C07": "Stomatognathic",
    "C08": "Respiratory",
    "C09": "Otorhinolaryngologic",
    "C10": "Nervous System",
    "C11": "Eye",
    "C12": "Urogenital",
    "C14": "Cardiovascular",
    "C15": "Hemic & Lymphatic",
    "C16": "Congenital/Neonatal",
    "C17": "Skin & Connective Tissue",
    "C18": "Nutritional & Metabolic",
    "C19": "Endocrine",
    "C20": "Immune System",
    "C21": "Environmentally Induced",
    "C22": "Animal Diseases",
    "C23": "Pathological Conditions",
    "C25": "Chemically-Induced",
    "C26": "Wounds & Injuries",
}

# Generic MeSH descriptors that must NOT count as clinical entities.
# Derived empirically: the highest-frequency content words across the 8,700+
# queries containing a definite patient reference. Every one of these is a real
# MeSH descriptor, and every one appears in queries that have no unique answer.
GENERIC_MESH_TERMS = frozenset(
    {
        # care process / episode
        "patients", "patient care", "patient discharge", "patient admission",
        "hospitalization", "hospitals", "hospital units", "aftercare",
        "length of stay", "ambulatory care", "critical care", "nursing care",
        "postoperative care", "preoperative care", "perioperative care",
        "intraoperative care", "terminal care", "primary health care",
        "delivery of health care", "continuity of patient care",
        "patient readmission", "patient transfer", "referral and consultation",
        "follow-up studies", "aftercare", "convalescence", "recovery of function",
        # outcome / status
        "treatment outcome", "outcome assessment, health care", "prognosis",
        "disease progression", "remission induction", "recurrence",
        "health status", "disease management", "symptom assessment",
        "signs and symptoms", "diagnosis", "diagnosis, differential",
        "physical examination", "medical history taking", "vital signs",
        "clinical deterioration", "disease attributes",
        # generic intervention
        "therapeutics", "surgical procedures, operative", "general surgery",
        "drug therapy", "combined modality therapy", "medication therapy management",
        "pharmaceutical preparations", "prescriptions", "self care",
        "patient education as topic", "counseling",
        # generic single words that are also MeSH descriptors
        "pain", "fever", "surgery", "treatment", "care", "therapy", "diagnosis",
        "symptoms", "outcome", "recovery", "admission", "discharge",
        "complications", "procedure", "medication", "medications", "examination",
        "findings", "history", "management", "plan", "condition", "status",
        "birth", "death", "life", "time", "hours", "months", "years", "day",
        "weight", "growth", "sleep", "diet", "exercise", "family", "home",
        "work", "risk", "safety", "quality of life",
        # people / roles
        "physicians", "nurses", "caregivers", "family", "parents", "mothers",
        "fathers", "child", "infant", "adult", "aged", "adolescent", "male",
        "female", "humans", "men", "women",
        # Second pass: MeSH descriptors observed leaking through the save
        # clause on the full corpus. Each is a real descriptor but carries no
        # entity information in a query ("the patient's disease", "role of X
        # in the patient's treatment"). Blocking a unigram does not block
        # phrases containing it — "back pain" and "blood pressure" still match.
        "role", "affect", "address", "health", "disease", "pressure", "back",
        "attention", "control", "reference values", "records", "reports",
        "physical functional performance", "activities of daily living",
    }
)

# Acronyms and initialisms — TEVAR, NIHSS, EBV, ANBP, WBC, MRI, TEE, ECMO.
# These carry the clinical signal MeSH's preferred terms miss, and are exactly
# the vocabulary the fine-tune is meant to learn (spec v2 §1).
ACRONYM_RE = re.compile(r"\b[A-Z][A-Z0-9]{1,7}\b")

# Eponymous scales and gradings: Clavien-Dindo, Glasgow-Blatchford.
EPONYM_RE = re.compile(r"\b[A-Z][a-z]{2,}-[A-Z][a-z]{2,}\b")

# Eponym + instrument noun: "Nurick scale", "Cobb angle", "Clavien-Dindo grade",
# "Glasgow score". Named instruments are the vocabulary the domain adaptation
# is supposed to learn (spec v2 §1) and MeSH indexes almost none of them, so
# without this pattern the filter drops answerable queries that name one.
EPONYM_MEASURE_RE = re.compile(
    r"\b[A-Z][A-Za-z]{2,}(?:-[A-Z][A-Za-z]{2,})?\s+"
    r"(?:scale|score|angle|grade|grading|index|classification|criteria|"
    r"stage|staging|sign|test|maneuver|manoeuvre|ratio|scoring)s?\b"
)

# Capitalized words that are not eponyms — a sentence-initial question word or
# a generic qualifier followed by "test"/"score" is not a named instrument.
EPONYM_LEAD_STOPLIST = frozenset(
    {
        "what", "which", "who", "when", "where", "how", "why", "the", "these",
        "those", "this", "that", "any", "other", "another", "diagnostic",
        "imaging", "lab", "laboratory", "blood", "urine", "clinical", "scale",
        "score", "follow", "screening", "routine", "initial", "final", "first",
        "second", "third", "further", "additional", "special", "standard",
    }
)

# Acronym-shaped tokens that are not clinical entities.
ACRONYM_STOPLIST = frozenset(
    {
        "AND", "THE", "FOR", "WAS", "WERE", "WHAT", "HOW", "WHY", "WHEN",
        "ICU", "ER", "OR",  # care settings, not conditions — generic like the above
        "PT",  # ambiguous: patient vs. physical therapy vs. prothrombin time
        "US", "UK", "USA", "I", "II", "III", "IV", "V", "VI",  # numerals/geography
        "NO", "YES", "OK", "TV", "PC", "ID", "AM", "PM",
    }
)

_TOKEN_RE = re.compile(r"[a-z0-9]+(?:[-'][a-z0-9]+)*")


@dataclass
class ClinicalVocab:
    """MeSH-derived clinical term lookup.

    `phrases` holds multi-word descriptors matched as n-grams; `unigrams` holds
    single-word descriptors. Both exclude GENERIC_MESH_TERMS.
    """

    phrases: set[str] = field(default_factory=set)
    unigrams: set[str] = field(default_factory=set)
    tree: dict[str, list[str]] = field(default_factory=dict)
    max_phrase_len: int = 5
    source: str = ""

    def __len__(self) -> int:
        return len(self.phrases) + len(self.unigrams)

    def find(self, text: str) -> list[str]:
        """Clinical terms present in `text`. Empty list means the query names
        no clinical entity — the filter's second clause."""
        hits: list[str] = []

        for match in ACRONYM_RE.findall(text):
            if match not in ACRONYM_STOPLIST and not match.isdigit():
                hits.append(match)
        hits.extend(EPONYM_RE.findall(text))
        for m in EPONYM_MEASURE_RE.findall(text):
            m = m.strip()
            if m.split()[0].lower().rstrip("s") not in EPONYM_LEAD_STOPLIST:
                hits.append(m)

        tokens = _TOKEN_RE.findall(text.lower())
        for n in range(min(self.max_phrase_len, len(tokens)), 1, -1):
            for i in range(len(tokens) - n + 1):
                gram = " ".join(tokens[i : i + n])
                if gram in self.phrases:
                    hits.append(gram)
        for tok in tokens:
            if tok in self.unigrams:
                hits.append(tok)

        seen: set[str] = set()
        return [h for h in hits if not (h.lower() in seen or seen.add(h.lower()))]

    def categories(self, text: str) -> set[str]:
        """MeSH top-level categories for terms found in `text` (specialty figure)."""
        out: set[str] = set()
        for term in self.find(text):
            for num in self.tree.get(term.lower(), ()):
                out.add(num[0])
        return out

    def disease_subcategories(self, text: str) -> set[str]:
        out: set[str] = set()
        for term in self.find(text):
            for num in self.tree.get(term.lower(), ()):
                if num[0] in ("C", "A") and len(num) >= 3:
                    out.add(num[:3])
        return out


def download_mesh(dest: Path, url: str = MESH_URL) -> Path:
    """Fetch the MeSH descriptor file if absent. Public domain, no registration."""
    dest = Path(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() and dest.stat().st_size > 10_000_000:
        return dest

    import urllib.request

    log.info("downloading MeSH descriptors from %s", url)
    urllib.request.urlretrieve(url, dest)
    if dest.stat().st_size < 10_000_000:
        raise RuntimeError(f"MeSH download looks wrong ({dest.stat().st_size} bytes)")
    return dest


@dataclass
class MeshDescriptor:
    preferred: str
    tree_numbers: list[str] = field(default_factory=list)
    entry_terms: list[str] = field(default_factory=list)


def parse_mesh_descriptors(path: Path) -> list[MeshDescriptor]:
    """Parse the ASCII MeSH descriptor file into records.

    Record shape:
        *NEWRECORD
        MH = Quadriplegia
        PRINT ENTRY = Quadriparesis|T047|NON|NRW|...
        ENTRY = Tetraplegia|...
        MN = C10.597.622.447

    Entry-term lines carry pipe-delimited metadata after the term itself; only
    the leading field is the synonym.
    """
    records: list[MeshDescriptor] = []
    current: MeshDescriptor | None = None

    with Path(path).open("r", encoding="utf-8", errors="replace") as fh:
        for line in fh:
            line = line.rstrip("\n")
            if line.startswith("*NEWRECORD"):
                if current and current.preferred:
                    records.append(current)
                current = MeshDescriptor(preferred="")
                continue
            if current is None or " = " not in line:
                continue
            key, _, value = line.partition(" = ")
            value = value.split("|", 1)[0].strip()
            if not value:
                continue
            if key == "MH":
                current.preferred = value
            elif key == "MN":
                current.tree_numbers.append(value)
            elif key in ("ENTRY", "PRINT ENTRY"):
                current.entry_terms.append(value)

    if current and current.preferred:
        records.append(current)
    return records


def _variants(term: str) -> set[str]:
    """A term plus its un-inverted reading ("Arthritis, Rheumatoid")."""
    term = term.strip().lower()
    out = {term}
    if "," in term:
        parts = [p.strip() for p in term.split(",") if p.strip()]
        if len(parts) > 1:
            out.add(" ".join(reversed(parts)))
    return {t for t in out if t}


def build_clinical_vocab(
    mesh_path: Path, min_term_len: int = 3, blocklist: frozenset[str] = GENERIC_MESH_TERMS
) -> ClinicalVocab:
    """Build the vocabulary the anaphoric filter consults.

    Includes MeSH entry terms (synonyms), not just preferred headings. A
    descriptor whose *preferred* form is blocklisted contributes none of its
    synonyms either — otherwise blocking "Drug Therapy" would leak straight
    back in as its entry term "Pharmacotherapy".
    """
    records = parse_mesh_descriptors(mesh_path)

    phrases: set[str] = set()
    unigrams: set[str] = set()
    tree: dict[str, list[str]] = {}
    n_entry = 0

    for rec in records:
        pref_variants = _variants(rec.preferred)
        if pref_variants & blocklist:
            continue

        surface_terms = set(pref_variants)
        for entry in rec.entry_terms:
            surface_terms |= _variants(entry)
            n_entry += 1

        for term in surface_terms:
            if term in blocklist or len(term) < min_term_len:
                continue
            if rec.tree_numbers:
                tree.setdefault(term, []).extend(rec.tree_numbers)
            (phrases if " " in term else unigrams).add(term)

    vocab = ClinicalVocab(
        phrases=phrases, unigrams=unigrams, tree=tree, source=str(mesh_path)
    )
    log.info(
        "clinical vocab: %s phrases, %s unigrams from %s descriptors "
        "(%s entry terms, %s blocklisted)",
        f"{len(phrases):,}",
        f"{len(unigrams):,}",
        f"{len(records):,}",
        f"{n_entry:,}",
        len(blocklist),
    )
    return vocab
