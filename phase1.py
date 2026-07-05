import pandas as pd
import numpy as np

desc_df = pd.read_csv(r"C:\Users\pratik p kakade\Desktop\ai clinic\DATASETS\symptom_Description.csv")
prec_df = pd.read_csv(r"C:\Users\pratik p kakade\Desktop\ai clinic\DATASETS\symptom_precaution.csv")
severity_df = pd.read_csv(r"C:\Users\pratik p kakade\Desktop\ai clinic\DATASETS\Symptom-severity.csv")

severity_df["Symptom"]=severity_df["Symptom"].str.replace("_"," ").str.strip()

clinical_kb=pd.merge(desc_df,prec_df,on="Disease",how="inner")

severity_dict=dict(zip(severity_df["Symptom"],severity_df["weight"]))

available_symptoms = severity_df['Symptom'].unique().tolist()

print(f"✅ Phase 1 Reset Complete.")
print(f"Total Verified Diseases: {len(clinical_kb)}")
print(f"Total Tracked Symptoms: {len(available_symptoms)}")