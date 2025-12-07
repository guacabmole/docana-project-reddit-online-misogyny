import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import networkx as nx
from matplotlib.lines import Line2D

# LOADING combined DATASET of topic and sentiment analysis results on chosen subreddits
df = pd.read_csv("../data/processed/data_topics_sentiment.csv")

# filtering empty or nan or uninformative content
df = df.dropna()
df = df[~df['vader_sentiment'].str.contains(r'\d+', regex=True)]
df = df[df['Topic'] != 'Outlier Topic']
df_filtered = df[df['author'] != '[deleted]']

# filtering out duplicates (same content by same user)
df_filtered.drop_duplicates(subset=['author', 'content'], keep=False, inplace=True)

# selecting a random sample from the complete dataset for visualization purposes
df_filtered = df_filtered.sample(1000, random_state=42)

# NETWORK OF USER CLUSTERS DEPENDING ON TOPIC
fig, ax = plt.subplots(figsize=(15, 10))

# selecting subreddit topics
top_topics = df_filtered['Topic'].value_counts()
top_topics = pd.DataFrame({'topic': top_topics.index, 'count': top_topics.values})
topics_list = top_topics['topic'].unique()
reduced_df = df_filtered[df_filtered['Topic'].isin(topics_list)]

# creating network graph
vader_graph = nx.Graph()

# get counts as a Series
author_counts_series = reduced_df['author'].value_counts()

# keep only the x most active authors
x = len(author_counts_series) # keeping all the authors
top_authors = author_counts_series.head(x).index
reduced_df = reduced_df[reduced_df['author'].isin(top_authors)]
authors = reduced_df['author'].unique()

# dict of counts for node sizes
author_counts = reduced_df['author'].value_counts().to_dict()

sentiment_colormap = {
    'positive': 'yellowgreen',
    'neutral': 'darkgrey',
    'negative': 'deeppink'
}

# calculating average sentiment of posts for each user/author
sentiment_mapping = {'negative': 0, 'neutral': 1, 'positive': 2}

def average_to_sentiment(avg):
    if avg < 0.5:
        return 'negative'
    elif avg < 1.5:
        return 'neutral'
    else:
        return 'positive'

avg_sentiment = (
    reduced_df
    .replace({'vader_sentiment': sentiment_mapping})
    .groupby('author')['vader_sentiment']
    .mean()
    .apply(average_to_sentiment)
)
avg_sentimentdf = pd.DataFrame({
    'author': avg_sentiment.index,
    'vader_sentiment': avg_sentiment.values
})

# creating nodes
for author in authors:
    curr_sentiment = avg_sentimentdf.loc[
        avg_sentimentdf['author'] == author, 'vader_sentiment'
    ].iloc[0]
    vader_graph.add_node(
        author,
        size=author_counts[author],
        sentiment=sentiment_colormap[curr_sentiment]
    )

# Adding edges with topic as an attribute (still full cliques, but on top 150 authors only)
topic_colors = sns.color_palette("pastel", len(topics_list))

for idx, topic in enumerate(topics_list):
    curr_topic_color = topic_colors[idx]
    authors_in_topic = reduced_df[reduced_df['Topic'] == topic]['author'].unique()
    for i, author1 in enumerate(authors_in_topic):
        temp_authors2 = np.delete(authors_in_topic, i)
        for idx2 in range(len(temp_authors2)):
            author2 = temp_authors2[idx2]
            vader_graph.add_edge(
                author1, author2,
                weight=0.3,
                color=curr_topic_color,
                width=0.01
            )

# node position
pos = nx.kamada_kawai_layout(vader_graph)

# drawing nodes
sizes = [vader_graph.nodes[node]['size'] * 20 for node in vader_graph.nodes]
colors = [vader_graph.nodes[node]['sentiment'] for node in vader_graph.nodes]
nx.draw_networkx_nodes(
    vader_graph, pos,
    node_size=sizes,
    node_color=colors,
    edgecolors='black',
    ax=ax
)

# drawing edges
edges = vader_graph.edges()
edge_colors = [vader_graph[u][v]['color'] for u, v in edges]
nx.draw_networkx_edges(
    vader_graph, pos,
    edge_color=edge_colors,
    alpha=0.35,
    ax=ax
)

ax.set_title(
    'Sample Network of Authors according to Topics \n(with VADER Sentiment Analyser)',
    fontsize=20
)
ax.axis('off')

# Legend

# activity distribution for legend text
author_activity = reduced_df['author'].value_counts()

# sentiment legend handles
sentiment_handles = [
    Line2D([0], [0], marker='o', linestyle='',
           markerfacecolor='yellowgreen', markeredgecolor='black',
           markersize=6, label='Average Positive User'),
    Line2D([0], [0], marker='o', linestyle='',
           markerfacecolor='deeppink', markeredgecolor='black',
           markersize=6, label='Average Negative User'),
    Line2D([0], [0], marker='o', linestyle='',
           markerfacecolor='darkgrey', markeredgecolor='black',
           markersize=6, label='Average Neutral User')
]

# activity legend handles
activity_handles = [
    Line2D([0], [0], marker='o', linestyle='',
           markerfacecolor='black', markeredgecolor='black',
           markersize=9,
           label=f'Most active user (in sample): {author_activity.iloc[0]} posts'),
    Line2D([0], [0], marker='o', linestyle='',
           markerfacecolor='black', markeredgecolor='black',
           markersize=3,
           label='Least active user (in sample): 1 post')
]

# topic handles
topic_handles = []
for i, topic in enumerate(topics_list):
    topic_handles.append(
        Line2D([0], [0],
               color=topic_colors[i],
               linestyle='solid',
               label=topic)
    )

all_handles = sentiment_handles + activity_handles + topic_handles

# place legend inside same figure (e.g., to the right)
ax.legend(
    handles=all_handles,
    loc='center left',
    bbox_to_anchor=(1, 0.5),
    frameon=False
)

plt.tight_layout()
plt.savefig('../outputs/AuthorNetworkGraph.jpg') #saving legend
plt.savefig('../dashboard/AuthorNetworkGraph.jpg') #saving legend
plt.show()
