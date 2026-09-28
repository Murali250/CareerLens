import os
import re
import json
import io
import fitz
import numpy as np

from copy import deepcopy
from pydantic import BaseModel, Field
from groq import Groq
from docx import Document
from fastembed import TextEmbedding


# =========================================================
# CONFIG
# =========================================================

MAX_RESUME_CHARS = 12000
MAX_JD_CHARS = 8000
EMBED_MODEL = "BAAI/bge-small-en-v1.5"

WEIGHTS = {
    "required": .25,
    "preferred": .08,
    "experience": .15,
    "responsibilities": .15,
    "semantic": .10,
    "keywords": .08,
    "title": .05,
    "education": .04,
    "parseability": .10,
}


# =========================================================
# MODELS
# =========================================================

class Keyword(BaseModel):
    term: str
    importance: str = "preferred"


class Requirement(BaseModel):
    text: str
    importance: str = "required"


class JDProfile(BaseModel):
    title: str = ""
    required: list[Keyword] = Field(default_factory=list)
    preferred: list[Keyword] = Field(default_factory=list)
    responsibilities: list[str] = Field(default_factory=list)
    education: list[str] = Field(default_factory=list)
    experience_years: float = 0


class ResumeChange(BaseModel):
    id: str
    original: str
    replacement: str
    reason: str


class WeakBullet(BaseModel):
    original: str
    reason: str


class Analysis(BaseModel):
    target_role: str
    summary: str
    strengths: list[str] = Field(default_factory=list)
    changes: list[ResumeChange] = Field(default_factory=list)
    weak_bullets: list[WeakBullet] = Field(default_factory=list)


class RejectionReview(BaseModel):
    valid_reason: bool
    explanation: str
    alternative: str = ""


# =========================================================
# TEXT / DOCUMENT HELPERS
# =========================================================

def clean_text(text):
    return re.sub(r"\s+", " ", text or "").strip()


def pdf_bytes(uploaded):
    return uploaded.getvalue() if hasattr(uploaded, "getvalue") else uploaded


def extract_pdf_text(data):
    doc = fitz.open(stream=data, filetype="pdf")
    text = "\n".join(page.get_text() for page in doc[:10])
    doc.close()
    return clean_text(text)


def docx_bytes(uploaded):
    return uploaded.getvalue() if hasattr(uploaded, "getvalue") else uploaded


def _iter_paragraphs(parent):
    """
    Iterate through normal paragraphs and paragraphs
    inside tables.
    """
    for p in getattr(parent, "paragraphs", []):
        yield p

    for table in getattr(parent, "tables", []):
        for row in table.rows:
            for cell in row.cells:
                yield from _iter_paragraphs(cell)


def iter_docx_paragraphs(doc):
    yield from _iter_paragraphs(doc)

    for section in doc.sections:
        yield from _iter_paragraphs(section.header)
        yield from _iter_paragraphs(section.footer)


def extract_docx_text(data):
    doc = Document(io.BytesIO(data))
    parts = []

    for p in iter_docx_paragraphs(doc):
        text = clean_text(p.text)
        if text:
            parts.append(text)

    return "\n".join(parts)


# =========================================================
# GROQ
# =========================================================

def _model():
    return os.getenv("GROQ_MODEL", "openai/gpt-oss-120b")


def _client():
    key = os.getenv("GROQ_API_KEY")
    if not key:
        raise ValueError("GROQ_API_KEY is missing.")
    return Groq(api_key=key)


def _chat(system, user, temperature=0):
    response = _client().chat.completions.create(
        model=_model(),
        temperature=temperature,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
    )
    return response.choices[0].message.content


def friendly_error(e):
    msg = str(e)

    if "authentication" in msg.lower():
        return "Invalid Groq API key."

    if "rate" in msg.lower():
        return "Groq rate limit reached. Please try again."

    if "connection" in msg.lower():
        return "Could not connect to Groq."

    return msg


def _parse_json(text):
    text = text.strip()

    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?", "", text)
        text = re.sub(r"```$", "", text).strip()

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        start = text.find("{")
        end = text.rfind("}")

        if start >= 0 and end > start:
            return json.loads(text[start:end + 1])

        raise ValueError("AI response could not be parsed as JSON.")


# =========================================================
# JD INTELLIGENCE
# =========================================================

JD_SYSTEM = """
You are an expert ATS and recruitment analyst.

Analyze the job description and extract structured information.

Return ONLY valid JSON:

{
  "title": "",
  "required": [
    {"term": "", "importance": "required"}
  ],
  "preferred": [
    {"term": "", "importance": "preferred"}
  ],
  "responsibilities": [],
  "education": [],
  "experience_years": 0
}

Rules:
- Required means explicitly required or strongly mandatory.
- Preferred means nice-to-have or preferred.
- Extract concrete technologies, skills, tools, concepts and qualifications.
- Do not invent requirements.
- Keep terms concise.
"""


def extract_jd_profile(jd):
    result = _chat(
        JD_SYSTEM,
        jd[:MAX_JD_CHARS],
        temperature=0
    )

    data = _parse_json(result)
    return JDProfile(**data)


# =========================================================
# NORMALIZATION
# =========================================================

ALIASES = {
    "javascript": {"js"},
    "typescript": {"ts"},
    "python": {"py"},
    "machine learning": {"ml"},
    "deep learning": {"dl"},
    "artificial intelligence": {"ai"},
    "natural language processing": {"nlp"},
    "large language model": {"llm"},
    "large language models": {"llm"},
    "react.js": {"react"},
    "node.js": {"node", "nodejs"},
    "postgresql": {"postgres", "postgresql"},
    "mongodb": {"mongo"},
    "scikit-learn": {"sklearn"},
    "rest api": {"rest", "restful api"},
}


def normalize(text):
    text = text.lower()
    text = re.sub(r"[^a-z0-9+#.\- ]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def present(term, text):
    term = normalize(term)
    text = normalize(text)

    if term in text:
        return True

    for alias in ALIASES.get(term, set()):
        if alias in text:
            return True

    return False


# =========================================================
# RESUME SECTIONS / EVIDENCE
# =========================================================

SECTION_NAMES = {
    "experience": ["experience", "work experience", "professional experience"],
    "projects": ["projects", "personal projects", "academic projects"],
    "skills": ["skills", "technical skills", "technologies"],
    "education": ["education", "academic background"],
    "certifications": ["certifications", "certificates"],
}


def extract_sections(text):
    lines = [x.strip() for x in text.splitlines() if x.strip()]
    sections = {}
    current = "other"

    for line in lines:
        n = normalize(line)

        found = False

        for section, names in SECTION_NAMES.items():
            if n in names:
                current = section
                sections.setdefault(current, [])
                found = True
                break

        if not found:
            sections.setdefault(current, []).append(line)

    return {
        k: "\n".join(v)
        for k, v in sections.items()
    }


def evidence_strength(term, sections):
    if present(term, sections.get("experience", "")):
        return 1.0

    if present(term, sections.get("projects", "")):
        return 0.85

    if present(term, sections.get("skills", "")):
        return 0.65

    if present(term, sections.get("certifications", "")):
        return 0.50

    return 0


# =========================================================
# EMBEDDINGS
# =========================================================

_embedder = None


def _get_embedder():
    global _embedder

    if _embedder is None:
        _embedder = TextEmbedding(model_name=EMBED_MODEL)

    return _embedder


def _chunks(text, size=700):
    words = text.split()
    return [
        " ".join(words[i:i + size])
        for i in range(0, len(words), size)
    ]


def _embeddings(texts):
    return np.array(list(_get_embedder().embed(texts)))


def semantic_similarity(a, b):
    if not a or not b:
        return 0

    vectors = _embeddings([a, b])

    a_vec, b_vec = vectors

    denom = np.linalg.norm(a_vec) * np.linalg.norm(b_vec)

    if denom == 0:
        return 0

    return float(np.dot(a_vec, b_vec) / denom)


def best_semantic_match(query, text):
    chunks = _chunks(text)

    if not chunks:
        return 0

    q = _embeddings([query])[0]
    vectors = _embeddings(chunks)

    denom = np.linalg.norm(vectors, axis=1) * np.linalg.norm(q)

    scores = np.divide(
        vectors @ q,
        denom,
        out=np.zeros(len(vectors)),
        where=denom != 0
    )

    return float(np.max(scores))


# =========================================================
# ATS MATCHING
# =========================================================

def skill_score(keywords, resume, sections):
    if not keywords:
        return 1.0

    scores = []

    for k in keywords:
        if present(k.term, resume):
            strength = evidence_strength(k.term, sections)
            scores.append(0.6 + 0.4 * strength)
        else:
            scores.append(0)

    return float(np.mean(scores))


def responsibility_score(responsibilities, resume):
    if not responsibilities:
        return 1.0

    scores = [
        best_semantic_match(r, resume)
        for r in responsibilities
    ]

    return float(np.mean(scores))


def title_score(jd_title, resume):
    if not jd_title:
        return 1.0

    lines = resume.splitlines()[:30]

    if any(present(jd_title, x) for x in lines):
        return 1.0

    return min(
        1,
        best_semantic_match(jd_title, "\n".join(lines)) + .2
    )


def experience_score(required_years, resume):
    if not required_years:
        return 1.0

    matches = re.findall(
        r"(\d+(?:\.\d+)?)\+?\s*(?:years?|yrs?)",
        resume.lower()
    )

    if not matches:
        return 0.5

    years = max(float(x) for x in matches)

    return min(1, years / required_years)


def education_score(education, resume):
    if not education:
        return 1.0

    matched = sum(
        present(x, resume)
        for x in education
    )

    return matched / len(education)


def keyword_match(keywords, resume):
    if not keywords:
        return 1.0

    matched = sum(
        present(k.term, resume)
        for k in keywords
    )

    return matched / len(keywords)


# =========================================================
# PARSEABILITY
# =========================================================

def parseability_score(resume):
    issues = []

    if len(resume.strip()) < 500:
        issues.append("Very little extractable text.")

    if resume.count("\n") < 8:
        issues.append("Very low text structure.")

    if re.search(r"[^\x00-\x7F]{20,}", resume):
        issues.append("Potential encoding issues.")

    score = max(
        0,
        1 - len(issues) * 0.15
    )

    return score, issues


# =========================================================
# ATS ENGINE
# =========================================================

def ats_score(resume, jd):
    profile = extract_jd_profile(jd)
    sections = extract_sections(resume)

    required = skill_score(
        profile.required,
        resume,
        sections
    )

    preferred = skill_score(
        profile.preferred,
        resume,
        sections
    )

    experience = experience_score(
        profile.experience_years,
        resume
    )

    responsibilities = responsibility_score(
        profile.responsibilities,
        resume
    )

    semantic = best_semantic_match(
        jd[:4000],
        resume
    )

    keywords = keyword_match(
        profile.required + profile.preferred,
        resume
    )

    title = title_score(
        profile.title,
        resume
    )

    education = education_score(
        profile.education,
        resume
    )

    parseability, issues = parseability_score(resume)

    semantic_norm = np.clip(
        (semantic - .45) / .40,
        0,
        1
    )

    score = (
        required * WEIGHTS["required"] +
        preferred * WEIGHTS["preferred"] +
        experience * WEIGHTS["experience"] +
        responsibilities * WEIGHTS["responsibilities"] +
        semantic_norm * WEIGHTS["semantic"] +
        keywords * WEIGHTS["keywords"] +
        title * WEIGHTS["title"] +
        education * WEIGHTS["education"] +
        parseability * WEIGHTS["parseability"]
    )

    # Required-skill coverage guardrail
    if required < .25:
        score = min(score, .50)
    elif required < .40:
        score = min(score, .60)
    elif required < .55:
        score = min(score, .70)

    required_terms = [
        k.term for k in profile.required
    ]

    preferred_terms = [
        k.term for k in profile.preferred
    ]

    return {
        "score": round(score * 100),
        "components": {
            "required": round(required * 100),
            "preferred": round(preferred * 100),
            "experience": round(experience * 100),
            "responsibilities": round(responsibilities * 100),
            "semantic": round(semantic_norm * 100),
            "keywords": round(keywords * 100),
            "title": round(title * 100),
            "education": round(education * 100),
            "parseability": round(parseability * 100),
        },
        "required_coverage": round(required * 100),
        "candidate_years": experience,
        "matched_required": [
            x for x in required_terms
            if present(x, resume)
        ],
        "missing_required": [
            x for x in required_terms
            if not present(x, resume)
        ],
        "missing_preferred": [
            x for x in preferred_terms
            if not present(x, resume)
        ],
        "responsibilities": profile.responsibilities,
        "parseability_issues": issues,
        "profile": profile.model_dump(),
    }


# =========================================================
# AI RESUME ANALYSIS
# =========================================================

ANALYSIS_SYSTEM = """
You are an expert resume optimization assistant.

Analyze the resume against the job description.

Return ONLY valid JSON:

{
  "target_role": "",
  "summary": "",
  "strengths": [],
  "changes": [
    {
      "id": "C1",
      "original": "",
      "replacement": "",
      "reason": ""
    }
  ],
  "weak_bullets": [
    {
      "original": "",
      "reason": ""
    }
  ]
}

Rules:
1. NEVER invent experience, skills, projects, numbers or achievements.
2. Every proposed change must be supported by the existing resume.
3. Improve wording, relevance, clarity and ATS alignment.
4. Keep replacements reasonably close to the original length.
5. Only propose changes to text that actually appears in the resume.
6. Prefer changing existing bullets instead of creating new claims.
7. Return at most 8 changes.
"""


def analyze_with_llm(resume, jd):
    prompt = f"""
JOB DESCRIPTION:

{jd[:MAX_JD_CHARS]}

RESUME:

{resume[:MAX_RESUME_CHARS]}
"""

    result = _chat(
        ANALYSIS_SYSTEM,
        prompt,
        temperature=0
    )

    return Analysis(**_parse_json(result))


# =========================================================
# CHANGE VERIFICATION
# =========================================================

def _normalized_resume(resume):
    return normalize(resume)


def verify_changes(analysis, resume):
    text = _normalized_resume(resume)

    verified = []

    for change in analysis.changes:
        original = normalize(change.original)

        if original and original in text:
            verified.append(change)

    return verified


def verify_bullets(analysis, resume):
    text = _normalized_resume(resume)

    return [
        b for b in analysis.weak_bullets
        if normalize(b.original) in text
    ]


# =========================================================
# HITL REJECTION REVIEW
# =========================================================

REJECTION_SYSTEM = """
You are reviewing why a user rejected a resume optimization.

Determine whether the rejection means:

1. The proposed claim is inaccurate / not genuinely supported.
2. The wording is poor and can be improved.
3. The user simply prefers the original.

Return ONLY:

{
  "valid_reason": true,
  "explanation": "",
  "alternative": ""
}

Never invent experience or skills.

If the rejection is about wording, provide a concise alternative.
If the proposed change is inaccurate, leave alternative empty.
"""


def review_rejection(change, reason):
    prompt = f"""
ORIGINAL:
{change.original}

PROPOSED:
{change.replacement}

USER REASON:
{reason}
"""

    return RejectionReview(
        **_parse_json(
            _chat(
                REJECTION_SYSTEM,
                prompt,
                temperature=0
            )
        )
    )


# =========================================================
# DOCX EDITING
# =========================================================

def _replace_in_paragraph(paragraph, original, replacement):
    """
    Replace text across Word runs while preserving the
    formatting of surrounding content.

    If original spans multiple runs:
    - replacement is placed into the first affected run
    - text from remaining affected runs is removed
    - all unrelated formatting remains untouched
    """

    full_text = "".join(
        run.text or ""
        for run in paragraph.runs
    )

    if not full_text:
        return False

    start = full_text.find(original)

    if start < 0:
        # Case-insensitive fallback
        match = re.search(
            re.escape(original),
            full_text,
            flags=re.IGNORECASE
        )

        if not match:
            return False

        start = match.start()
        end = match.end()

    else:
        end = start + len(original)

    run_ranges = []
    cursor = 0

    for i, run in enumerate(paragraph.runs):
        text = run.text or ""
        run_ranges.append(
            (i, cursor, cursor + len(text))
        )
        cursor += len(text)

    affected = [
        (i, a, b)
        for i, a, b in run_ranges
        if b > start and a < end
    ]

    if not affected:
        return False

    first_i = affected[0][0]

    # Put replacement into first affected run.
    first_run = paragraph.runs[first_i]
    first_text = first_run.text or ""

    first_start = start - run_ranges[first_i][1]
    first_end = min(
        len(first_text),
        end - run_ranges[first_i][1]
    )

    first_run.text = (
        first_text[:first_start]
        + replacement
        + first_text[first_end:]
    )

    # Remove affected text from remaining runs.
    for i, a, b in reversed(affected[1:]):
        run = paragraph.runs[i]
        text = run.text or ""

        local_start = max(
            0,
            start - a
        )

        local_end = min(
            len(text),
            end - a
        )

        run.text = (
            text[:local_start]
            + text[local_end:]
        )

    return True


def _replace_text(paragraph, original, replacement):
    """
    Handles exact and whitespace-normalized matching.
    """

    # Exact match
    if original in paragraph.text:
        return _replace_in_paragraph(
            paragraph,
            original,
            replacement
        )

    # Whitespace-normalized matching
    clean_original = re.sub(
        r"\s+",
        " ",
        original
    ).strip()

    full = paragraph.text

    clean_full = re.sub(
        r"\s+",
        " ",
        full
    ).strip()

    if clean_original not in clean_full:
        return False

    # Try individual words / shorter normalized chunks.
    words = clean_original.split()

    for size in range(
        len(words),
        max(3, len(words) // 2),
        -1
    ):
        candidate = " ".join(words[:size])

        if candidate.lower() in clean_full.lower():
            # Search exact text first.
            for run in paragraph.runs:
                if candidate.lower() in (run.text or "").lower():
                    actual = re.search(
                        re.escape(candidate),
                        run.text or "",
                        re.IGNORECASE
                    )

                    if actual:
                        return _replace_in_paragraph(
                            paragraph,
                            actual.group(0),
                            replacement
                        )

    return False


def apply_approved_changes(docx_data, approved_changes):
    """
    Modify ONLY approved text.

    Returns:
        optimized_docx,
        applied_changes,
        skipped_changes
    """

    doc = Document(io.BytesIO(docx_data))

    applied = []
    skipped = []

    paragraphs = list(
        iter_docx_paragraphs(doc)
    )

    for change in approved_changes:

        # Prevent absurdly large replacements.
        if (
            len(change.replacement)
            > max(80, len(change.original) * 1.5)
        ):
            skipped.append({
                "id": change.id,
                "reason": "Replacement is too long."
            })
            continue

        found = False

        for paragraph in paragraphs:

            if not paragraph.text.strip():
                continue

            if _replace_text(
                paragraph,
                change.original,
                change.replacement
            ):
                applied.append(change.id)
                found = True
                break

        if not found:
            skipped.append({
                "id": change.id,
                "reason": "Original text not found in DOCX."
            })

    output = io.BytesIO()
    doc.save(output)

    return (
        output.getvalue(),
        applied,
        skipped
    )


# =========================================================
# COVER LETTER
# =========================================================

COVER_SYSTEM = """
Write a concise professional cover letter using ONLY
information supported by the resume and job description.

Do not invent experience.

Use plain text.
"""


def stream_cover_letter(resume, jd):
    prompt = f"""
RESUME:
{resume[:MAX_RESUME_CHARS]}

JOB DESCRIPTION:
{jd[:MAX_JD_CHARS]}
"""

    response = _client().chat.completions.create(
        model=_model(),
        temperature=.3,
        stream=True,
        messages=[
            {"role": "system", "content": COVER_SYSTEM},
            {"role": "user", "content": prompt},
        ],
    )

    for chunk in response:
        content = chunk.choices[0].delta.content

        if content:
            yield content