import streamlit as st
import joblib
import re
import nltk
import numpy as np
from scipy.sparse import hstack, csr_matrix

nltk.download('stopwords', quiet=True)
nltk.download('wordnet',   quiet=True)
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

st.set_page_config(page_title="Fake Job Detector", layout="centered")

#style
st.markdown("""
<link rel="stylesheet"
      href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.5.0/css/all.min.css"
      crossorigin="anonymous" referrerpolicy="no-referrer" />

<style>
  .stApp { background: linear-gradient(135deg, #0d1117 0%, #161b22 100%); }

  .title-card {
    background: rgba(255,255,255,0.05);
    border: 1px solid rgba(255,255,255,0.1);
    border-radius: 16px; padding: 28px;
    text-align: center; margin-bottom: 24px;
  }

  .result-fake {
    background: linear-gradient(135deg, #7f0000, #c62828);
    border-radius: 14px; padding: 24px;
    text-align: center; color: white;
    font-size: 1.4rem; font-weight: bold; margin-top: 16px;
  }
  .result-real {
    background: linear-gradient(135deg, #1b5e20, #2e7d32);
    border-radius: 14px; padding: 24px;
    text-align: center; color: white;
    font-size: 1.4rem; font-weight: bold; margin-top: 16px;
  }

  .input-card {
    background: rgba(255,255,255,0.04);
    border: 1px solid rgba(255,255,255,0.08);
    border-radius: 12px; padding: 18px; margin-bottom: 14px;
  }

  h1, h2, h3, label, p { color: white !important; }

  .field-label {
    color: #c9d1d9;
    font-size: 0.88rem;
    font-weight: 500;
    margin-bottom: 4px;
    display: flex;
    align-items: center;
    gap: 7px;
  }
  .field-label i {
    color: #58a6ff;
    font-size: 0.95rem;
    width: 16px;
    text-align: center;
  }

  .section-heading {
    color: #e6edf3;
    font-size: 1.05rem;
    font-weight: 600;
    margin: 12px 0 16px 0;
    display: flex;
    align-items: center;
    gap: 8px;
  }
  .section-heading i { color: #58a6ff; font-size: 1rem; }

  .footer-text {
    color: #6e7681;
    font-size: 0.8rem;
    text-align: center;
    display: flex;
    align-items: center;
    justify-content: center;
    gap: 6px;
  }
  .footer-text i { color: #58a6ff; }
</style>
""", unsafe_allow_html=True)

#Header
st.markdown("""
<div class="title-card">
  <h1 style="margin-bottom:8px;">
    <i class="fa-solid fa-shield-halved" style="color:#58a6ff;margin-right:10px;"></i>
    Fake Job Posting Detector
  </h1>
  <p style="color:#8b949e; margin:0;">
    Enter job posting details to detect if it is fraudulent or legitimate
    using Machine Learning
  </p>
</div>
""", unsafe_allow_html=True)

#Model Load
@st.cache_resource
def load_artifacts():
    import os
    if not os.path.exists('Models/best_model.pkl'):
        return None, None, None, None
    model    = joblib.load('Models/best_model.pkl')
    tfidf    = joblib.load('Models/tfidf_vectorizer.pkl')
    num_cols = joblib.load('Models/numeric_cols.pkl')
    mod_name = joblib.load('Models/best_model_name.pkl')
    return model, tfidf, num_cols, mod_name

model, tfidf, num_cols, mod_name = load_artifacts()

if model is None:
    st.error("Models not found! Run Section 8 of the Jupyter notebook first to save the model.")
    st.stop()

#Text cleaner
STOP_WORDS = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

def clean_text(text):
    if not text: return ''
    text = str(text).lower()
    text = re.sub(r'<[^>]+>', ' ', text)
    text = re.sub(r'http\S+|www\S+', ' ', text)
    text = re.sub(r'[^a-z\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    tokens = [lemmatizer.lemmatize(w) for w in text.split()
              if w not in STOP_WORDS and len(w) > 2]
    return ' '.join(tokens)

#Input form
st.markdown("""
<div class="section-heading">
  <i class="fa-solid fa-clipboard-list"></i>
  Enter Job Posting Details
</div>
""", unsafe_allow_html=True)

col1, col2 = st.columns(2)

with col1:
    st.markdown('<div class="input-card">', unsafe_allow_html=True)

    st.markdown('<div class="field-label"><i class="fa-solid fa-briefcase"></i> Job Title</div>',
                unsafe_allow_html=True)
    title = st.text_input("Job Title", placeholder="e.g. Software Engineer",
                          label_visibility="collapsed")

    st.markdown('<div class="field-label"><i class="fa-solid fa-building"></i> Company Profile</div>',
                unsafe_allow_html=True)
    company = st.text_input("Company Profile", placeholder="Brief company description",
                             label_visibility="collapsed")

    st.markdown('<div class="field-label"><i class="fa-solid fa-location-dot"></i> Location</div>',
                unsafe_allow_html=True)
    location = st.text_input("Location", placeholder="e.g. US, CA, San Francisco",
                              label_visibility="collapsed")

    st.markdown('<div class="field-label"><i class="fa-solid fa-industry"></i> Industry</div>',
                unsafe_allow_html=True)
    industry = st.text_input("Industry", placeholder="e.g. Information Technology",
                              label_visibility="collapsed")

    st.markdown('<div class="field-label"><i class="fa-solid fa-id-badge"></i> Employment Type</div>',
                unsafe_allow_html=True)
    emp_type = st.selectbox("Employment Type",
                             ["", "Full-time", "Part-time", "Contract", "Temporary", "Other"],
                             label_visibility="collapsed")

    st.markdown('<div class="field-label"><i class="fa-solid fa-star"></i> Required Experience</div>',
                unsafe_allow_html=True)
    experience = st.selectbox("Required Experience",
                               ["", "Entry level", "Mid-Senior level", "Associate",
                                "Director", "Executive", "Not Applicable"],
                               label_visibility="collapsed")

    st.markdown('</div>', unsafe_allow_html=True)

with col2:
    st.markdown('<div class="input-card">', unsafe_allow_html=True)

    st.markdown('<div class="field-label"><i class="fa-solid fa-file-lines"></i> Job Description</div>',
                unsafe_allow_html=True)
    description = st.text_area("Job Description",
                                placeholder="Describe the role and responsibilities...",
                                height=120, label_visibility="collapsed")

    st.markdown('<div class="field-label"><i class="fa-solid fa-list-check"></i> Requirements</div>',
                unsafe_allow_html=True)
    requirements = st.text_area("Requirements",
                                 placeholder="List required skills and qualifications...",
                                 height=80, label_visibility="collapsed")

    st.markdown('<div class="field-label"><i class="fa-solid fa-gift"></i> Benefits</div>',
                unsafe_allow_html=True)
    benefits = st.text_input("Benefits", placeholder="e.g. Health insurance, 401k",
                              label_visibility="collapsed")

    st.markdown('<div class="field-label" style="margin-top:12px;margin-bottom:8px;">'
                '<i class="fa-solid fa-sliders"></i> Additional Options</div>',
                unsafe_allow_html=True)

    telecommute = st.checkbox("Remote / Telecommuting position")
    has_logo    = st.checkbox("Company logo provided")
    has_ques    = st.checkbox("Screening questions included")

    st.markdown('</div>', unsafe_allow_html=True)

#Analyse button
st.markdown("")
analyse = st.button("Analyse Job Posting", use_container_width=True)

#Prediction steps
if analyse:
    #Assemble and clean text
    full_text = ' '.join([title, company, description, requirements,
                          benefits, emp_type, experience, industry])
    cleaned   = clean_text(full_text)

    #TF-IDF transform
    X_tfidf = tfidf.transform([cleaned])

    #Numeric features
    num_vals = [[
        int(telecommute), int(has_logo), int(has_ques),
        len(description), len(title), len(requirements),
        int(len(description) > 0),
        int(len(requirements) > 0),
        int(len(company) > 0),
    ]]
    X_num      = csr_matrix(np.array(num_vals, dtype=float))
    X_combined = hstack([X_tfidf, X_num])

    #Predict
    prediction  = model.predict(X_combined)[0]
    probability = model.predict_proba(X_combined)[0]
    fake_prob   = probability[1] * 100
    real_prob   = probability[0] * 100

    #Display result
    if prediction == 1:
        st.markdown(f"""
        <div class="result-fake">
          <i class="fa-solid fa-triangle-exclamation" style="margin-right:10px;"></i>
          FAKE JOB POSTING DETECTED<br>
          <span style="font-size:1rem;font-weight:normal;">
            Fraud Probability: {fake_prob:.1f}%
          </span>
        </div>
        """, unsafe_allow_html=True)
        st.warning("This posting shows signs of fraud. Do not provide personal information or pay any fees.")
    else:
        st.markdown(f"""
        <div class="result-real">
          <i class="fa-solid fa-circle-check" style="margin-right:10px;"></i>
          LEGITIMATE JOB POSTING<br>
          <span style="font-size:1rem;font-weight:normal;">
            Real Probability: {real_prob:.1f}%
          </span>
        </div>
        """, unsafe_allow_html=True)
        st.success("This posting appears to be legitimate.")

    #Confidence breakdown
    st.markdown("""
    <div class="section-heading" style="margin-top:20px;">
      <i class="fa-solid fa-chart-bar"></i> Confidence Breakdown
    </div>
    """, unsafe_allow_html=True)

    c1, c2 = st.columns(2)
    c1.metric("Fraud Probability",      f"{fake_prob:.1f}%")
    c2.metric("Legitimate Probability", f"{real_prob:.1f}%")
    st.progress(int(fake_prob))

    st.markdown(f"""
    <p style="color:#8b949e;font-size:0.82rem;text-align:center;margin-top:10px;">
      <i class="fa-solid fa-microchip" style="color:#58a6ff;margin-right:5px;"></i>
      Prediction by: <strong style="color:#c9d1d9;">{mod_name}</strong>
    </p>
    """, unsafe_allow_html=True)

#Footer
st.markdown("---")
st.markdown("""
<div class="footer-text">
  <i class="fa-solid fa-shield-halved"></i>
  Fake Job Posting Detection by Shareefdeen
</div>
""", unsafe_allow_html=True)