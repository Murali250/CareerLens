import os
import streamlit as st

from dotenv import load_dotenv
import analyzer as az

load_dotenv()


# =========================================================
# PAGE
# =========================================================

st.set_page_config(
    page_title="CareerLens AI",
    page_icon="🎯",
    layout="wide"
)

st.title("🎯 CareerLens AI")
st.caption(
    "Your resume. Your control. AI optimization."
)


# =========================================================
# SESSION
# =========================================================

defaults = {
    "analysis": None,
    "resume_text": "",
    "jd_text": "",
    "docx_data": None,
    "ats": None,
    "approved": [],
    "rejected": {},
    "optimized_docx": None,
    "applied": [],
    "skipped": [],
    "cover_letter": "",
}

for key, value in defaults.items():
    if key not in st.session_state:
        st.session_state[key] = value


# =========================================================
# INPUT
# =========================================================

st.subheader("1. Upload Resume")

resume_file = st.file_uploader(
    "Upload your resume",
    type=["docx", "pdf"],
    help=(
        "DOCX is recommended because CareerLens can "
        "modify Word documents while preserving formatting."
    )
)

st.subheader("2. Target Job Description")

jd = st.text_area(
    "Paste the job description",
    height=260,
    placeholder="Paste the complete job description here..."
)


# =========================================================
# ANALYZE
# =========================================================

if st.button(
    "🔍 Analyze Resume",
    type="primary",
    use_container_width=True
):

    if not resume_file:
        st.warning("Please upload your resume.")

    elif not jd.strip():
        st.warning("Please paste the job description.")

    else:

        try:
            data = resume_file.getvalue()

            if resume_file.name.lower().endswith(".docx"):

                resume_text = az.extract_docx_text(data)
                docx_data = data

            else:

                resume_text = az.extract_pdf_text(data)
                docx_data = None

            if not resume_text.strip():
                st.error(
                    "Could not extract readable text from the resume."
                )
                st.stop()

            resume_text = resume_text[:az.MAX_RESUME_CHARS]

            with st.spinner(
                "Analyzing resume against the job description..."
            ):

                analysis = az.analyze_with_llm(
                    resume_text,
                    jd
                )

                ats = az.ats_score(
                    resume_text,
                    jd
                )

            st.session_state.analysis = analysis
            st.session_state.resume_text = resume_text
            st.session_state.jd_text = jd
            st.session_state.docx_data = docx_data
            st.session_state.ats = ats

            st.session_state.approved = []
            st.session_state.rejected = {}
            st.session_state.optimized_docx = None
            st.session_state.applied = []
            st.session_state.skipped = []

            st.success("Analysis complete.")

        except Exception as e:
            st.error(
                f"Analysis failed: {az.friendly_error(e)}"
            )


# =========================================================
# RESULTS
# =========================================================

analysis = st.session_state.analysis
ats = st.session_state.ats

if analysis and ats:

    st.divider()

    st.subheader("📊 Resume Analysis")

    c1, c2, c3, c4 = st.columns(4)

    c1.metric(
        "Job Match",
        f"{min(100, round(ats['score']))}%"
    )

    c2.metric(
        "ATS Compatibility",
        f"{ats['score']}%"
    )

    c3.metric(
        "Required Coverage",
        f"{ats['required_coverage']}%"
    )

    c4.metric(
        "Experience",
        f"{ats['components']['experience']}%"
    )


    # =====================================================
    # SCORE BREAKDOWN
    # =====================================================

    with st.expander(
        "📈 ATS Score Breakdown",
        expanded=True
    ):

        components = ats["components"]

        for name, value in components.items():

            label = name.replace(
                "_",
                " "
            ).title()

            st.write(
                f"**{label}: {value}%**"
            )

            st.progress(
                min(100, value) / 100
            )


    # =====================================================
    # ROLE
    # =====================================================

    st.subheader(
        f"🎯 Target Role: {analysis.target_role}"
    )

    st.write(analysis.summary)


    # =====================================================
    # SKILLS
    # =====================================================

    col1, col2 = st.columns(2)

    with col1:

        st.markdown("### ✅ Matched Required Skills")

        if ats["matched_required"]:
            for skill in ats["matched_required"]:
                st.write(f"• {skill}")
        else:
            st.write("No required skills matched.")


    with col2:

        st.markdown("### ❌ Missing Required Skills")

        if ats["missing_required"]:
            for skill in ats["missing_required"]:
                st.write(f"• {skill}")
        else:
            st.write("No major required skills missing.")


    # =====================================================
    # PREFERRED
    # =====================================================

    st.markdown("### ⭐ Missing Preferred Skills")

    if ats["missing_preferred"]:

        st.write(
            ", ".join(ats["missing_preferred"])
        )

        st.caption(
            "These are signals from the JD only. "
            "CareerLens will not add unsupported skills."
        )

    else:
        st.write("None.")


    # =====================================================
    # RESPONSIBILITIES
    # =====================================================

    with st.expander("📋 Job Responsibilities"):

        for responsibility in ats["responsibilities"]:
            st.write(f"• {responsibility}")


    # =====================================================
    # PARSEABILITY
    # =====================================================

    with st.expander("🤖 ATS Parseability"):

        score = ats["components"]["parseability"]

        st.write(
            f"Parseability score: **{score}%**"
        )

        if ats["parseability_issues"]:

            for issue in ats["parseability_issues"]:
                st.warning(issue)

        else:
            st.success(
                "No major parseability issues detected."
            )


    # =====================================================
    # STRENGTHS
    # =====================================================

    st.subheader("💪 Strengths")

    for strength in analysis.strengths:
        st.write(f"• {strength}")


    # =====================================================
    # HITL
    # =====================================================

    st.divider()

    st.subheader(
        "🧠 Human-in-the-Loop Optimization"
    )

    st.info(
        "CareerLens will never modify your resume automatically. "
        "Approve only the changes you want."
    )


    verified_changes = az.verify_changes(
        analysis,
        st.session_state.resume_text
    )

    if not verified_changes:

        st.success(
            "No safe resume changes were identified."
        )

    else:

        for change in verified_changes:

            st.markdown(
                f"### {change.id}"
            )

            col1, col2 = st.columns(2)

            with col1:

                st.markdown("**Original**")

                st.info(
                    change.original
                )

            with col2:

                st.markdown("**AI Suggestion**")

                st.success(
                    change.replacement
                )

            st.caption(
                f"Why: {change.reason}"
            )


            approve_key = f"approve_{change.id}"
            reject_key = f"reject_{change.id}"


            col_a, col_b = st.columns(2)

            with col_a:

                if st.button(
                    "✅ Approve",
                    key=approve_key,
                    use_container_width=True
                ):

                    if change.id not in [
                        x.id
                        for x in st.session_state.approved
                    ]:
                        st.session_state.approved.append(
                            change
                        )

                    st.success(
                        f"{change.id} approved."
                    )


            with col_b:

                if st.button(
                    "❌ Reject",
                    key=reject_key,
                    use_container_width=True
                ):

                    st.session_state.rejected[
                        change.id
                    ] = True


            if change.id in st.session_state.rejected:

                reason = st.text_input(
                    f"Why don't you want {change.id}?",
                    key=f"reason_{change.id}"
                )

                if st.button(
                    "Review Rejection",
                    key=f"review_{change.id}"
                ):

                    if not reason.strip():

                        st.warning(
                            "Please explain your reason."
                        )

                    else:

                        try:

                            review = az.review_rejection(
                                change,
                                reason
                            )

                            st.markdown(
                                "### 🔎 Rejection Review"
                            )

                            st.write(
                                review.explanation
                            )

                            if review.alternative:

                                st.markdown(
                                    "**Alternative:**"
                                )

                                st.success(
                                    review.alternative
                                )

                                if st.button(
                                    "✅ Approve Alternative",
                                    key=f"alt_{change.id}"
                                ):

                                    alternative = az.ResumeChange(
                                        id=change.id,
                                        original=change.original,
                                        replacement=review.alternative,
                                        reason="User-approved alternative"
                                    )

                                    st.session_state.approved = [
                                        x for x in
                                        st.session_state.approved
                                        if x.id != change.id
                                    ]

                                    st.session_state.approved.append(
                                        alternative
                                    )

                                    st.success(
                                        f"{change.id} alternative approved."
                                    )

                        except Exception as e:

                            st.error(
                                az.friendly_error(e)
                            )


    # =====================================================
    # APPLY CHANGES
    # =====================================================

    st.divider()

    st.subheader(
        "📄 Generate Optimized Resume"
    )

    if not st.session_state.docx_data:

        st.warning(
            "PDF resumes can be analyzed, but they cannot "
            "be safely edited while preserving the original "
            "design. Upload the original DOCX to enable editing."
        )

    else:

        approved = st.session_state.approved

        st.write(
            f"Approved changes: **{len(approved)}**"
        )

        if approved:

            if st.button(
                "🚀 Apply Approved Changes",
                type="primary",
                use_container_width=True
            ):

                try:

                    optimized, applied, skipped = (
                        az.apply_approved_changes(
                            st.session_state.docx_data,
                            approved
                        )
                    )

                    st.session_state.optimized_docx = optimized
                    st.session_state.applied = applied
                    st.session_state.skipped = skipped

                except Exception as e:

                    st.error(
                        f"Could not create optimized DOCX: {e}"
                    )


    # =====================================================
    # RESULT
    # =====================================================

    if st.session_state.optimized_docx:

        st.success(
            f"Applied {len(st.session_state.applied)} approved changes."
        )

        if st.session_state.skipped:

            st.warning("Some changes were skipped:")

            for item in st.session_state.skipped:

                st.write(
                    f"• {item['id']}: {item['reason']}"
                )

        st.download_button(
            "⬇️ Download Optimized Resume",
            data=st.session_state.optimized_docx,
            file_name="CareerLens_Optimized_Resume.docx",
            mime=(
                "application/vnd.openxmlformats-officedocument."
                "wordprocessingml.document"
            ),
            use_container_width=True
        )


    # =====================================================
    # COVER LETTER
    # =====================================================

    st.divider()

    st.subheader(
        "✉️ Generate Cover Letter"
    )

    if st.button(
        "Generate Cover Letter",
        use_container_width=True
    ):

        try:

            st.session_state.cover_letter = ""

            placeholder = st.empty()

            for chunk in az.stream_cover_letter(
                st.session_state.resume_text,
                st.session_state.jd_text
            ):

                st.session_state.cover_letter += chunk

                placeholder.markdown(
                    st.session_state.cover_letter
                )

        except Exception as e:

            st.error(
                az.friendly_error(e)
            )


    if st.session_state.cover_letter:

        st.download_button(
            "⬇️ Download Cover Letter",
            data=st.session_state.cover_letter,
            file_name="CareerLens_Cover_Letter.txt",
            mime="text/plain",
            use_container_width=True
        )