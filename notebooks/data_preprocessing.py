# Data Preprocessing for Webis-TLDR-17 Dataset saved in data/raw/ from data_collection.ipynb
import pandas as pd
import regex as re
import os

# Loading Raw Dataset
try: 
    df = pd.read_csv("../data/raw/data_raw.csv")
except Exception as e:
    print(f"Could not load data_raw due to {e}")
    print("\nMake sure the .zip file was unpacked and .csv file moved to data/raw/ folder")

# Or use: 
# import zipfile
# with zipfile.ZipFile("../data/raw/data_raw.zip", "r") as z:
#         with z.open("data_raw.csv") as f:
#             df = pd.read_csv(f)

# Webis dataset uses 'content' as full post text.
df["text"] = df["content"].fillna("").astype(str)
df["summary_cleaned"] = df["summary"].fillna("").astype(str)

# removing empty or whitespace-only posts
df = df[df["text"].str.strip() != ""]
df = df[df["summary_cleaned"].str.strip() != ""]

# Remove punctuation-only entries
df = df[~df["text"].str.replace(r"[^\w\s]", "", regex=True).str.strip().eq("")]
df = df[~df["summary_cleaned"].str.replace(r"[^\w\s]", "", regex=True).str.strip().eq("")]

# Lowercase clean text and summaries 
df["text_clean"] = df["text"].str.lower()
df["summary_cleaned"] = df["summary_cleaned"].str.lower()

# Drop duplicates based on cleaned text
len_old = len(df)
df = df.drop_duplicates(subset=["text_clean"]).reset_index(drop=True)
print(f"After cleaning: {len(df)} rows remain from original dataset with {len_old} rows.")

df = df[['author', 'subreddit', 'subreddit_id', 'id', 'content', 'summary_cleaned', 'text_clean']]

# Save cleaned file
df.to_csv("../data/processed/data_clean.csv", index=False)