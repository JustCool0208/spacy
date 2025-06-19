import pdfplumber
import docx2txt
import re
import spacy
import json
import streamlit as st
import tempfile
import os

# Try loading the spaCy model with fallback
try:
    nlp = spacy.load("en_core_web_sm")
except OSError:
    st.error("SpaCy model 'en_core_web_sm' not found. Please install it using: `python -m spacy download en_core_web_sm`")
    st.stop()

# Skills database
SKILL_DB = [
    'python', 'java', 'c++', 'c', 'javascript', 'html', 'css',
    'machine learning', 'deep learning', 'nlp', 'data science', 'sql',
    'mongodb', 'mysql', 'tensorflow', 'pytorch', 'keras',
    'pandas', 'numpy', 'scikit-learn', 'git', 'docker', 'linux',
    'react', 'node.js', 'flask', 'fastapi', 'ml', 'dl', 'ai', 'dsa'
]

# ---------- TEXT EXTRACTION ----------
def extract_text(file_path):
    if file_path.endswith('.pdf'):
        with pdfplumber.open(file_path) as pdf:
            return " ".join(page.extract_text() for page in pdf.pages if page.extract_text())
    elif file_path.endswith('.docx'):
        return docx2txt.process(file_path)
    elif file_path.endswith('.txt'):
        with open(file_path, 'r', encoding='utf-8') as f:
            return f.read()
    else:
        raise ValueError("Unsupported file format")

# ---------- CLEANING ----------
def preprocess(text):
    return re.sub(r'\s+', ' ', text).strip()

# ---------- REGEX BASED EXTRACTIONS ----------
def extract_email(text):
    return re.findall(r"[\w\.-]+@[\w\.-]+", text)

def extract_phone(text):
    return re.findall(r"\+?\d[\d\s\-]{8,}\d", text)

def extract_linkedin(text):
    return re.findall(r"https?://(?:www\.)?linkedin\.com/in/[\w\-]+", text)

def extract_github(text):
    return re.findall(r"https?://(?:www\.)?github\.com/[\w\-]+", text)

def extract_skills(text):
    return [skill for skill in SKILL_DB if skill.lower() in text.lower()]

# ---------- ENTITY EXTRACTION ----------
def extract_entities(text):
    doc = nlp(text)
    entities = {
        "PERSON": set(),
        "ORG": set(),
        "DATE": set(),
        "LOCATION": set(),
        "EDUCATION": set(),
        "PROJECTS": set()
    }

    # SpaCy Named Entities
    for ent in doc.ents:
        if ent.label_ in ["PERSON", "ORG", "DATE"]:
            entities[ent.label_].add(ent.text)
        elif ent.label_ in ["GPE", "LOC"]:
            entities["LOCATION"].add(ent.text)

    # Keyword-based EDUCATION
    edu_keywords = ["b.tech", "m.tech", "bachelor", "master", "phd", "degree", "university", "college", "cgpa", "gpa"]
    for line in text.lower().splitlines():
        if any(keyword in line for keyword in edu_keywords):
            entities["EDUCATION"].add(line.strip())

    # Simple PROJECTS
    for line in text.splitlines():
        if any(kw in line.lower() for kw in ["project", "built", "developed", "created", "designed"]):
            if 5 < len(line) < 250:
                entities["PROJECTS"].add(line.strip())

    return {k: list(v) for k, v in entities.items()}

# ---------- MAIN FUNCTION ----------
def parse_resume(file_path):
    text = extract_text(file_path)
    text = preprocess(text)

    return {
        "email": extract_email(text),
        "phone": extract_phone(text),
        "linkedin": extract_linkedin(text),
        "github": extract_github(text),
        "skills": extract_skills(text),
        "entities": extract_entities(text)
    }

# ---------- STREAMLIT UI ----------
st.set_page_config(page_title="AI Resume Parser", layout="centered")
st.title("📄 AI Resume Parser")
st.markdown("Upload your resume (**PDF**, **DOCX**, or **TXT**) to extract structured information.")

uploaded_file = st.file_uploader("📤 Choose a resume file", type=["pdf", "docx", "txt"])

if uploaded_file is not None:
    # Get the correct extension from the uploaded file name
    extension = os.path.splitext(uploaded_file.name)[1]

    # Save to a temp file with the same extension
    with tempfile.NamedTemporaryFile(delete=False, suffix=extension) as tmp_file:
        tmp_file.write(uploaded_file.read())
        tmp_path = tmp_file.name

    try:
        result = parse_resume(tmp_path)
        st.success("✅ Resume parsed successfully!")
        st.subheader("🔍 Extracted Information")
        st.json(result)

        st.download_button(
            label="📥 Download as JSON",
            data=json.dumps(result, indent=2),
            file_name="parsed_resume.json",
            mime="application/json"
        )

    except Exception as e:
        st.error(f"❌ Error parsing resume: {str(e)}")
    finally:
        os.remove(tmp_path)
