# ---------------------------------------------
# Data Preprocessing for Webis-TLDR-17 Dataset
# Project: Reddit Online Misogyny – MensRights
# ---------------------------------------------

import pandas as pd
import regex as re
import os

# ---------------------------------------------
# Load Raw Dataset
# ---------------------------------------------
df = pd.read_csv("../data/raw/data_raw.csv")

# ---------------------------------------------
# Select the text column for NLP
# Webis dataset uses 'content' as full post text.
# ---------------------------------------------
# if "content" in df.columns:
#     df["text"] = df["content"].fillna("").astype(str)
# else:
#     raise KeyError(
#         "The dataset does not contain a 'content' column. "
#         "Available columns: " + str(df.columns.tolist())
#     )
df["text"] = df["content"].fillna("").astype(str)
df['sum'] = df["summary"].fillna("").astype(str)
# ---------------------------------------------
# Remove empty or whitespace-only posts
# ---------------------------------------------
df = df[df["text"].str.strip() != ""]
df = df[df["sum"].str.strip() != ""]

# ---------------------------------------------
# Remove punctuation-only entries
# ---------------------------------------------
df = df[~df["text"].str.replace(r"[^\w\s]", "", regex=True).str.strip().eq("")]
df = df[~df["sum"].str.replace(r"[^\w\s]", "", regex=True).str.strip().eq("")]


# ---------------------------------------------
# Lowercase clean text
# ---------------------------------------------
df["text_clean"] = df["text"].str.lower()
df["summary"] = df["sum"].str.lower()

# ---------------------------------------------
# Drop duplicates based on cleaned text
# ---------------------------------------------
df = df.drop_duplicates(subset=["text_clean"]).reset_index(drop=True)

print("After cleaning:", len(df), "rows remain")

# ---------------------------------------------
# Save cleaned file
# ---------------------------------------------
os.makedirs("../data/processed", exist_ok=True)
df.to_csv("../data/processed/data_clean.csv", index=False)