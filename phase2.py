from sentence_transformers import SentenceTransformer, util
import torch
import pandas as pd
from phase1 import clinical_kb


# 1. Load the Model (Lightweight & Open Source)
model = SentenceTransformer('all-MiniLM-L6-v2')

# 2. Prepare the Knowledge Base
# clinical_kb contains your 41 diseases and descriptions
descriptions = clinical_kb['Description'].tolist()
disease_names = clinical_kb['Disease'].tolist()

# 3. Pre-compute Embeddings for the 41 Descriptions
# We do this once so the search is lightning fast later
description_embeddings = model.encode(descriptions, convert_to_tensor=True)

def semantic_diagnostic_engine(user_input_text, top_k=3):
    """
    Compares user text to disease descriptions using Cosine Similarity.
    """
    # Convert user input (e.g., "itching and red skin") into a vector
    user_embedding = model.encode(user_input_text, convert_to_tensor=True)
    
    # Calculate Cosine Similarity against all 41 diseases
    cosine_scores = util.cos_sim(user_embedding, description_embeddings)[0]
    
    # Get the top matches
    top_results = torch.topk(cosine_scores, k=top_k)
    
    predictions = []
    for score, idx in zip(top_results.values, top_results.indices):
        disease = disease_names[idx]
        
        # Fetch the original data for the report
        row = clinical_kb[clinical_kb['Disease'] == disease].iloc[0]
        
        predictions.append({
            "Disease": disease,
            "Match_Score": round(float(score), 4),
            "Description": row['Description'],
            "Precautions": [row['Precaution_1'], row['Precaution_2'], row['Precaution_3'], row['Precaution_4']]
        })
        
    return predictions

