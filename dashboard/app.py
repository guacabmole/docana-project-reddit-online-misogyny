import streamlit as st
import pandas as pd
import plotly.express as px
import streamlit.components.v1 as components
import zipfile

st.set_page_config(layout = "wide")
st.markdown("""
    <style>
        /* Remove excessive top padding from the main app area */
        section[data-testid="stHeader"] {visibility: hidden;}
        div.block-container {
            padding-top: 1rem !important;   /* reduce from ~6rem default */
        }
        .stMultiSelect [data-baseweb="tag"] {
            background-color: #4a4a4a !important;  /* gray tags */
            color: #ffffff !important;
            border-radius: 999px !important;
        }
        .stMultiSelect div[data-baseweb="select"] > div {
            max-height: 60px !important;   
            overflow-y: auto !important;   
        }
    </style>
""", unsafe_allow_html=True)

# Load data
@st.cache_data
def load_data():
    zip_path = "dashboard/data/data_topics_sentiment.zip"      
    csv_name = "data_topics_sentiment.csv"         

    with zipfile.ZipFile(zip_path, "r") as z:
        with z.open(csv_name) as f:
            df = pd.read_csv(f)

    return df

df = load_data()

st.title("Reddit Online Misogyny - Topic Explorer")
st.caption("Based on BERTopic + VADER sentiment")

# Segmented control for view choice (in main area)
view_choice = st.radio(
    "Select view mode",
    ["Sentiment analysis", "Network analysis", "BERTopic outputs"],
    index=2,  # 0-based → 2 == "BERTopic outputs"
    horizontal=True,  # puts the options side by side
)

# SENTIMENT ANALYSIS VIEW
if view_choice == "Sentiment analysis":
    st.subheader("Sentiment analysis")

    subreddits = sorted(df["subreddit"].dropna().unique()) if "subreddit" in df.columns else []
    topics = sorted(df["Topic"].dropna().unique()) if "Topic" in df.columns else []

    filtered_df = df.copy()

    if subreddits:
        selected_subs = st.sidebar.multiselect(
            "Subreddit",
            subreddits,
            default=subreddits
        )
        filtered_df = filtered_df[filtered_df["subreddit"].isin(selected_subs)]

    if topics:
        selected_topics = st.sidebar.multiselect(
            "Topic",
            topics,
            default=topics
        )
        filtered_df = filtered_df[filtered_df["Topic"].isin(selected_topics)]

    # Sentiment ratios per topic (stacked bar, sorted by negative ratio)
    st.subheader("Sentiment ratios per topic (VADER)")


    sentiment_counts = (
        filtered_df
        .groupby(["Topic", "vader_sentiment"])
        .size()
        .unstack(fill_value=0)
    )

    for col in ["positive", "negative", "neutral"]:
        if col not in sentiment_counts.columns:
            sentiment_counts[col] = 0

    sentiment_ratios = sentiment_counts.div(
        sentiment_counts.sum(axis=1), axis=0
    ).fillna(0)

    sentiment_ratios = sentiment_ratios.sort_values(
        by="negative",
        ascending=False
    )

    sentiment_ratios_pct = (sentiment_ratios * 100).round(2)

    # Long-format for Plotly
    ratios_long = (
        sentiment_ratios_pct
        .reset_index()
        .melt(
            id_vars="Topic",
            value_vars=["positive", "negative", "neutral"],
            var_name="Sentiment",
            value_name="Percentage"
        )
    )

    # Stacked bar chart
    fig_ratios = px.bar(
        ratios_long,
        x="Topic",
        y="Percentage",
        color="Sentiment",
        title="Normalized VADER sentiment ratios per topic (sorted by negative ratio)",
        barmode="stack",
        category_orders={"Topic": sentiment_ratios_pct.index.tolist()},
        color_discrete_map={
            "positive": "green",
            "negative": "red",
            "neutral": "blue",
        },
        text="Percentage"
    )

    fig_ratios.update_layout(
        xaxis_tickangle=-45,
        legend_title_text="Sentiment"
    )
    fig_ratios.update_traces(texttemplate="%{text:.1f}%", textposition="inside")

    st.plotly_chart(fig_ratios, use_container_width=True)

    # Example posts 
    st.subheader("Example posts")
    if len(filtered_df) > 0:
        n = st.slider("Number of posts to show", 5, 50, 10)
        cols_to_show = [
            c for c in ["subreddit", "Topic", "TopicName", "summary", "text_clean"]
            if c in filtered_df.columns
        ]
        st.dataframe(
            filtered_df[cols_to_show].sample(
                min(n, len(filtered_df)),
                random_state=42
            )
        )
    else:
        st.info("No posts to display with current filters.")

# NETWORK ANALYSIS VIEW
elif view_choice == "Network analysis":
    st.subheader("Network analysis")

    st.markdown(
        "This view shows a precomputed network visualization "
        "of most active authors according to topics"
    )

    image_path = "dashboard/network_output.png"

    try:
        st.image(
            image_path,
            caption="Network of topics and communities",
            width=1200
        )
    except Exception as e:
        st.error(f"Could not load network image: {e}")
        st.info("Make sure the image is in the 'data/' folder and the filename is correct.")

# BERTOPIC OUTPUTS VIEW
elif view_choice == "BERTopic outputs":
    st.subheader("Interactive BERTopic visualizations")

    st.markdown(
        "This view displays interactive HTML visualizations generated by BERTopic. "
        "Export them from your BERTopic model using methods like "
        "`topic_model.visualize_topics()` and save as HTML files."
    )

    viz_options = {
        "Topic distribution": "dashboard/data/topic_distribution.html",
        "Topic hierarchy": "dashboard/data/topics_hierarchy.html",
        "Topics' documents": "dashboard/data/topics_visualize_documents.html",
    }

    choice = st.selectbox(
        "Choose a BERTopic visualization",
        list(viz_options.keys())
    )

    html_path = viz_options[choice]

    try:
        with open(html_path, "r", encoding="utf-8") as f:
            html_content = f.read()
        components.html(html_content, height=800, scrolling=True)
    except FileNotFoundError:
        st.error(f"HTML file not found: {html_path}")
        st.info(
            "Make sure you exported the BERTopic visualization to this path. For example:\n\n"
            "```python\n"
            "fig = topic_model.visualize_topics()\n"
            "fig.write_html('dashboard/data/bertopic_topics_overview.html')\n"
            "```"
        )
    except Exception as e:
        st.error(f"Error loading HTML: {e}")
