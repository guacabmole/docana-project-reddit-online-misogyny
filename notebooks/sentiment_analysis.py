# 03_sentiment_analysis.py
# ---------------------------------------------
# Sentiment analysis for Reddit misogyny project
# ---------------------------------------------
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import warnings

warnings.filterwarnings("ignore")

# ------------------------------------------------
# 1. Load data with topics
# ------------------------------------------------
df = pd.read_csv("../data/processed/data_with_topics.csv")
df.rename(columns={"Topic": "TopicID","CustomName": "Topic"}, inplace=True)
df
# ------------------------------------------------
# 2. Run VADER sentiment analysis
# ------------------------------------------------
analyzer = SentimentIntensityAnalyzer()

def get_vader_score(text: str) -> float:
    if not isinstance(text, str):
        text = "" if pd.isna(text) else str(text)
    scores = analyzer.polarity_scores(text)
    return scores["compound"]

def label_vader(score: float) -> str:
    # Standard VADER thresholds
    if score >= 0.05:
        return "positive"
    elif score <= -0.05:
        return "negative"
    else:
        return "neutral"

print("Computing VADER sentiment scores...")
df["vader_score"] = df["text_clean"].apply(get_vader_score)
df["vader_sentiment"] = df["vader_score"].apply(label_vader)

# ------------------------------------------------
# 3. Per-topic sentiment summary
# ------------------------------------------------
sentiment_counts_per_topic = (
    df.groupby(["Topic", "vader_sentiment"])
      .size()
      .unstack(fill_value=0)
      .reset_index()
)

# Also compute ratios (percentages) per topic
sentiment_ratios_per_topic = sentiment_counts_per_topic.set_index("Topic")
sentiment_ratios_per_topic = sentiment_ratios_per_topic.div(
    sentiment_ratios_per_topic.sum(axis=1), axis=0
)

print("\nSentiment counts per topic (head):")
print(sentiment_counts_per_topic.head())

print("\nSentiment ratios per topic (head):")
print(sentiment_ratios_per_topic.head())


mean_sentiment_per_topic_vader = df.groupby('Topic')['vader_score'].mean().reset_index()
mean_sentiment_per_topic_vader["vader_sentiment"] = mean_sentiment_per_topic_vader["vader_score"].apply(label_vader)
color_map = {'positive': 'green', 'neutral': 'blue', 'negative': 'red'}
mean_sentiment_per_topic_vader['Color'] = mean_sentiment_per_topic_vader['vader_sentiment'].map(color_map)


# ------------------------------------------------
# 4. Simple plots
# ------------------------------------------------
plt.figure(figsize=(8, 5))
sns.countplot(data=df, x="vader_sentiment",
              order=["negative", "neutral", "positive"])
plt.title("Overall sentiment distribution (VADER)")
plt.xlabel("Sentiment")
plt.ylabel("Number of posts")
plt.tight_layout()
plt.show()

# Top N topics by number of documents
top_topics = df["Topic"].value_counts().head(10).index
df_top = df[df["Topic"].isin(top_topics)]

plt.figure(figsize=(12, 6))
sns.countplot(
    data=df_top,
    x="Topic",
    hue="vader_sentiment",
    order=top_topics,
    hue_order=["negative", "neutral", "positive"],
)
plt.title("Sentiment distribution per topic (top 10 topics)")
plt.xlabel("Topic")
plt.ylabel("Number of posts")
plt.legend(title="VADER sentiment")
plt.tight_layout()
plt.show()

# Visualizing the mean sentiment per topic with colors based on sentiment
import numpy as np

df_plot = mean_sentiment_per_topic_vader.copy()

def label_from_color(c):
    if c == "green":
        return "Positive"
    elif c == "red":
        return "Negative"
    else:
        return "Neutral"

df_plot["SentimentLabel"] = df_plot["Color"].apply(label_from_color)

import plotly.express as px

fig = px.bar(
    df_plot,
    x="Topic",
    y="vader_score",
    color="SentimentLabel",  # legend groups
    color_discrete_map={
        "Positive": "green",
        "Neutral": "blue",
        "Negative": "red",
    },
    labels={
        "Topic": "Topic",
        "vader_score": "Mean VADER Sentiment",
        "SentimentLabel": "Sentiment"
    },
    title="Mean VADER Sentiment per Topic",
)

# Rotate x-tick labels
fig.update_layout(
    xaxis_tickangle=-45,               # 45° rotation
    xaxis_tickfont=dict(size=10),
    bargap=0.2,
)

fig.write_html("../outputs/vader.html")
fig.show()

# ------------------------------------------------
# 5. Save combined dataset for later use
# ------------------------------------------------
df.to_csv("../data/processed/data_topics_sentiment.csv", index=False)
df.to_csv("../dashboard/data/data_topics_sentiment.csv", index=False)

print(f"\nSaved topics + sentiment to data/processed")
