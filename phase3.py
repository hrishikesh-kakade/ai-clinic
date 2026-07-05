# The template we will send to Groq
PROMPT_TEMPLATE ="""
System: You are a Senior Clinical Decision Support Assistant. 
Context: A patient has reported the following symptoms: {user_input}
The AI model has identified the most likely condition as: {disease_name}
Medical Description: {description}
Precautions: {precautions}
Severity Score: {severity_score}/100

Task: Generate a professional, concise summary for a healthcare provider. 
- Highlight why the symptoms match the description.
- Note the urgency based on the severity score.
- Structure the advice into "Immediate Actions" and "Further Diagnostics".
- Tone: Clinical, objective, and non-alarmist.
"""



from fastapi import FastAPI
from pydantic import BaseModel
from groq import Groq
import os
from phase2 import semantic_diagnostic_engine

# Initialize FastAPI and Groq Client
# app = FastAPI(title="AI Clinical Support System")
client = Groq(api_key="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") # Use your free Groq key

class SymptomRequest(BaseModel):
    symptoms: str

# @app.post("/diagnose")
async def get_diagnosis(request: SymptomRequest):
    # 1. Run Semantic Search (from Phase 2)
    top_matches = semantic_diagnostic_engine(request.symptoms)
    primary_match = top_matches[0]
    
    # 2. Calculate Severity (Logic from Phase 1)
    # severity_score = calculate_total_severity(request.symptoms)
    
    # 3. Call Groq for Clinical Summary
    chat_completion = client.chat.completions.create(
        messages=[
            {"role": "system", "content": "You are a clinical assistant."},
            {"role": "user", "content": PROMPT_TEMPLATE.format(
                user_input=request.symptoms,
                disease_name=primary_match['Disease'],
                description=primary_match['Description'],
                precautions=", ".join(primary_match['Precautions']),
                severity_score=75 # Placeholder for logic
            )}
        ],
        model="llama-3.3-70b-versatile", # High speed, low latency
    )
    
    # 4. Final Response Object
    return {
        "diagnosis": primary_match['Disease'],
        "confidence": primary_match['Match_Score'],
        "clinical_summary": chat_completion.choices[0].message.content,
        "raw_data": {
            "description": primary_match['Description'],
            "precautions": primary_match['Precautions']
        }
    }




