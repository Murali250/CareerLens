# CareerLens AI

**Analyze → Propose → Human approval → Apply → Verify.**

CareerLens AI is a human-in-the-loop resume optimization tool. Upload a resume PDF and a target job description. The app analyzes the match, proposes evidence-grounded content changes, asks the user to approve or reject each change, and only then creates a new PDF containing approved changes.

## What makes it different

- **HITL by design:** the AI never silently modifies the resume.
- **Evidence-grounded changes:** proposed edits must be supported by information already present in the resume.
- **Rejection loop:** if a user rejects a change, CareerLens asks why. Genuine accuracy/meaning objections are respected; wording/optimization objections can receive one alternative recommendation.
- **Design lock:** the original PDF is the source of truth. The app does not intentionally change page size, page count, margins, colors, or section structure. If an approved replacement cannot safely fit its original text region, it is skipped instead of shrinking the design.
- **Verification:** the final PDF is checked for page count and page geometry before download.
- **Existing analysis:** keyword coverage and semantic similarity remain part of the job-match score.
- **Cover letter:** optional, streamed and based only on resume facts.

## HITL flow

```text
Resume PDF + Job Description
            |
            v
      AI analysis
            |
            v
      Proposed changes
            |
            v
       HUMAN REVIEW
        /        \
    Approve      Reject
      |             |
      |          Ask why?
      |          /      \
      |     Genuine   Optimization issue
      |       |             |
      |     Keep         Recommend alternative
      |                     |
      +---------<-----------+
                |
                v
      Final user confirmation
                |
                v
       Apply approved changes
                |
                v
          Layout verification
                |
                v
       Download optimized PDF
```

## Important PDF behavior

The app edits the original PDF rather than rebuilding the entire resume from a blank template. It searches for approved original text and replaces it inside the original text region. The original page geometry remains the source of truth.

PDFs are not universally editable. Complex/scanned PDFs or replacements that do not fit safely may be rejected by the PDF layer. **The app will skip unsafe replacements rather than changing font size or rearranging the resume.**

The current PDF renderer uses PyMuPDF and preserves the original page geometry. Exact font preservation cannot be guaranteed for every PDF because PDF fonts can be embedded or subsetted in ways that prevent direct reuse. The verification layer therefore reports geometry preservation and the user should inspect the final PDF before submitting it.

## Run locally

```bash
python -m venv myenv
myenv\\Scripts\\activate        # Windows
pip install -r requirements.txt
```

Create `.env`:

```text
GROQ_API_KEY=your_groq_api_key
GROQ_MODEL=openai/gpt-oss-120b
```

Run:

```bash
streamlit run app.py
```

## Deploy on Streamlit Community Cloud

1. Push `app.py`, `analyzer.py`, `requirements.txt` and `README.md` to GitHub.
2. Create a Streamlit Community Cloud app using `app.py`.
3. Add `GROQ_API_KEY` under Streamlit Secrets.

No database or authentication is required for the MVP.

## Project structure

```text
app.py             Streamlit UI + HITL workflow
analyzer.py        LLM analysis, scoring, PDF extraction and PDF modification
requirements.txt   Runtime dependencies
README.md          Project documentation
```

## Safety / trust rules

CareerLens must not invent a skill, metric, employer, project, qualification or responsibility. A JD keyword is not sufficient evidence to add it to a resume. Only user-approved changes can reach the PDF modification function.

## Limitations

- Scanned/image-only PDFs are not supported by the current text extraction path.
- Very long resumes/JDs are truncated before LLM analysis.
- Exact font reproduction is not guaranteed for arbitrary PDF files.
- An external ATS score is intentionally not hard-coded into the MVP because ATS vendors and free API limits change. The internal job-match score remains deterministic and transparent.
