
import streamlit as st
import pandas as pd
import requests

# Set Page Config
st.set_page_config(page_title="AI Clinical Decision Support", layout="wide")

st.title("🩺 Clinical Decision Support System")
st.markdown("---")

# Sidebar for Patient Demographics (Added Context)
with st.sidebar:
    st.header("Patient Context")
    age = st.number_input("Age", 0, 120, 25)
    gender = st.selectbox("Gender", ["Male", "Female", "Other"])
    st.info("Basic medical history integration ready.")

# Main Layout
col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("Input Symptoms")
    user_input = st.text_area("Describe the patient's symptoms (e.g., 'persistent cough and fever')", height=150)

    if st.button("Generate Diagnosis"):
        if user_input:
            with st.spinner("Analyzing Clinical Knowledge Base..."):
                # 1. Logic Call (Connecting to your Phase 2/3 code)
                # For this example, we assume the logic is imported or called via API
                results = semantic_diagnostic_engine(user_input) 

                # Store results in session state
                st.session_state['results'] = results
        else:
            st.warning("Please enter symptoms first.")

with col2:
    st.subheader("Diagnostic Insights")
    if 'results' in st.session_state:
        primary = st.session_state['results'][0]

        # Confidence Meter
        st.metric(label="Primary Diagnosis", value=primary['Disease'])
        st.progress(primary['Match_Score'])
        st.caption(f"Confidence Score: {int(primary['Match_Score']*100)}%")

        # Severity Flag (Using your Severity CSV logic)
        st.error("Urgency Level: HIGH") # Example logic

        # Precautions (From your Dataset #2)
        with st.expander("Recommended Precautions"):
            for p in primary['Precautions']:
                st.write(f"• {p}")

# Final Report (The Groq Output)
st.markdown("---")
if 'results' in st.session_state:
    st.subheader("Generated Clinical Summary (LLM)")
    # This would be the response.json()['clinical_summary'] from Step 3
    st.info("Patient presents with indicators of " + primary['Disease'] + ". Further diagnostics required.")
