import streamlit as st
import pdfplumber
import spacy
import os
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Load NLP Model
nlp = spacy.load("en_core_web_sm")

# Function to extract text from PDFs
def extract_text_from_pdf(pdf_file):
    text = ""
    with pdfplumber.open(pdf_file) as pdf:
        for page in pdf.pages:
            text += page.extract_text() + "\n"
    return text.strip()

# Function to preprocess text
def preprocess_text(text):
    doc = nlp(text.lower())
    return " ".join([token.lemma_ for token in doc if not token.is_stop and token.is_alpha])

# Function to rank resumes
def rank_resumes(resumes, job_desc):
    job_desc = preprocess_text(job_desc)
    texts = [job_desc] + [preprocess_text(resume) for resume in resumes]
    
    vectorizer = TfidfVectorizer()
    vectors = vectorizer.fit_transform(texts)
    similarities = cosine_similarity(vectors[0], vectors[1:])[0]

    ranked_indices = np.argsort(similarities)[::-1]
    ranked_resumes = [(resumes[i], similarities[i]) for i in ranked_indices]
    
    return ranked_resumes

# Streamlit UI
st.title("AI-Powered Resume Shortlisting")
st.write("Upload resumes and enter a job description to rank candidates.")

# Job Description Input
job_desc = st.text_area("Enter Job Description", height=200)

# Upload Resumes
uploaded_files = st.file_uploader("Upload Resumes (PDF)", type=["pdf"], accept_multiple_files=True)

if st.button("Shortlist Candidates"):
    if job_desc and uploaded_files:
        resume_texts = [extract_text_from_pdf(file) for file in uploaded_files]
        ranked_resumes = rank_resumes(resume_texts, job_desc)

        st.subheader("Ranked Candidates:")
        for i, (resume, score) in enumerate(ranked_resumes):
            st.write(f"**Rank {i+1}** - {uploaded_files[i].name} - Score: {score:.2f}")

    else:
        st.warning("Please enter a job description and upload resumes.")
