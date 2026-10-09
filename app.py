
from pathlib import Path
import streamlit as st
import joblib

st.set_page_config(page_title="Fake Job Posting Detector", page_icon="🛡️", layout="centered")
MODEL_PATH = Path(__file__).parent / "outputs" / "fake_job_pipeline.joblib"

@st.cache_resource
def load_model():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model file missing: {MODEL_PATH}. Run the notebook first.")
    return joblib.load(MODEL_PATH)

st.title("🛡️ Fake Job Posting Detector")
st.write("Enter a job advertisement to estimate whether it may be fraudulent.")
st.warning("Educational screening aid only. A prediction is not proof that a job is fake or genuine.")

with st.form("job_form"):
    title = st.text_input("Job title")
    company_profile = st.text_area("Company profile")
    description = st.text_area("Job description")
    requirements = st.text_area("Requirements")
    benefits = st.text_area("Benefits")
    location = st.text_input("Location")
    department = st.text_input("Department")
    employment_type = st.selectbox("Employment type", ["", "Full-time", "Part-time", "Contract", "Temporary", "Internship", "Other"])
    required_experience = st.text_input("Required experience")
    required_education = st.text_input("Required education")
    industry = st.text_input("Industry")
    function = st.text_input("Job function")
    submitted = st.form_submit_button("Analyze job posting")

if submitted:
    if not title.strip() and not description.strip():
        st.error("Enter at least a job title or job description.")
    else:
        job = {
            "title": title, "company_profile": company_profile, "description": description,
            "requirements": requirements, "benefits": benefits, "location": location,
            "department": department, "employment_type": employment_type,
            "required_experience": required_experience, "required_education": required_education,
            "industry": industry, "function": function
        }
        ordered_cols = [
            "title", "company_profile", "description", "requirements", "benefits",
            "location", "department", "employment_type", "required_experience",
            "required_education", "industry", "function"
        ]
        text = " ".join(str(job.get(c, "") or "") for c in ordered_cols).strip()
        try:
            model = load_model()
            pred = int(model.predict([text])[0])
            score = float(model.predict_proba([text])[0, 1])
            st.metric("Estimated fraud score", f"{score:.1%}")
            st.progress(max(0.0, min(1.0, score)))
            if pred == 1:
                st.error("Potentially fraudulent — verify carefully.")
            else:
                st.success("Likely legitimate according to this model — still verify independently.")
            st.caption("Never pay recruitment fees. Verify the employer using official contact details.")
        except Exception as e:
            st.error(f"Could not analyze the posting: {e}")

st.divider()
st.caption("Trained on historical job posting data. Scams change over time and model errors are possible.")
