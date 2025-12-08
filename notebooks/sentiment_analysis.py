# Sentiment analysis for Reddit Online Misogyny project
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
import seaborn as sns
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import warnings
warnings.filterwarnings("ignore")

# Loading data with topics
try: 
    df = pd.read_csv("../data/processed/data_with_topics.csv")
except Exception as e:
    print(f"Could not load data due to {e}")
    print("\nMake sure the .zip file was unpacked and .csv file moved to data/processed/ folder")

# Or use: 
import zipfile
with zipfile.ZipFile("../data/processed/data_with_topics.zip", "r") as z:
        with z.open("data_with_topics.csv") as f:
            df = pd.read_csv(f)

df.rename(columns={"Topic": "TopicID","CustomName": "Topic"}, inplace=True)
df = df[df['Topic'] != 'Outlier Topic'] # fitering for Outliers

# Running VADER sentiment analysis
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

df["vader_score"] = df["text_clean"].apply(get_vader_score)
df["vader_sentiment"] = df["vader_score"].apply(label_vader)

# Per-topic sentiment summary
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
print(sentiment_counts_per_topic)

print("\nSentiment ratios per topic (head):")
print(sentiment_ratios_per_topic)


mean_sentiment_per_topic_vader = df.groupby('Topic')['vader_score'].mean().reset_index()
mean_sentiment_per_topic_vader["vader_sentiment"] = mean_sentiment_per_topic_vader["vader_score"].apply(label_vader)
color_map = {'positive': 'green', 'neutral': 'blue', 'negative': 'red'}
mean_sentiment_per_topic_vader['Color'] = mean_sentiment_per_topic_vader['vader_sentiment'].map(color_map)

# Creating a pie chart of sentiment distribution per subreddit
sentiment_counts_per_subreddit = (
    df.groupby(["subreddit", "vader_sentiment"])
    .size()
    .unstack(fill_value=0)
)

subreddits = (df["subreddit"].value_counts().index)

for subreddit in subreddits:
    counts = sentiment_counts_per_subreddit.loc[subreddit]
    counts = counts.reindex(["positive", "neutral", "negative"], fill_value=0)
    labels = counts.index
    sizes = counts.values
    colors = [color_map[label] for label in labels]

    plt.figure(figsize=(6, 6))
    plt.pie(
        sizes,
        labels=labels,
        autopct="%1.1f%%",
        colors=colors,
        startangle=100,
    )
    plt.title(f"Sentiment Distribution in r/{subreddit} (VADER)")
    plt.axis("equal")
    plt.show()

# Visualizing the mean sentiment per topic with colors based on sentiment
df_plot = mean_sentiment_per_topic_vader.copy()

def label_from_color(c):
    if c == "green":
        return "Positive"
    elif c == "red":
        return "Negative"
    else:
        return "Neutral"

df_plot["SentimentLabel"] = df_plot["Color"].apply(label_from_color)

fig = px.bar(
    df_plot,
    x="Topic",
    y="vader_score",
    color="SentimentLabel",
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

fig.update_layout(
    xaxis_tickangle=-45,
    xaxis_tickfont=dict(size=10),
    bargap=0.2,
)

fig.show()

# Save combined dataset for later use
df.to_csv("../data/processed/data_topics_sentiment.csv", index=False)
df.to_csv("../dashboard/data/data_topics_sentiment.csv", index=False)
