# Exploring Misogyny Dynamics in Reddit Online Manosphere

Author: *Samuel Givanelli*  
Course: *Document Analysis (Summer semester 2025, M.Sc. Social and Economic Data Science, University of Konstanz)*  
Dashboard: [Link to the Dashboard](https://reddit-online-misogyny.streamlit.app/)
---

## Abstract

This project analyzes discourse in the “MensRights” and related subreddits to understand how the community constructs narratives related to gender, feminism, and men’s issues. Using topic modeling, sentiment analysis, and lexical analysis grounded in the framework proposed in *Mapping the Ideological Landscape of Extreme Misogyny* (Perliger, Stevens & Leidig, 2023), the project identifies dominant themes, emotional characteristics, and rhetorical patterns within the subreddit. Results are additionally contextualized through optional network visualizations to highlight user-topic interactions.

---

## Introduction

The Men’s Rights Movement (MRM), as discussed in Perliger et al. (2023), forms a core component of the broader “manosphere,” characterised by grievances around gender equality, perceived discrimination against men, and hostility toward feminist institutions. While scholars have extensively examined highly extreme communities such as incels, fewer computational studies have analyzed the more moderate yet influential Men’s Rights spaces.

This project uses computational linguistic tools to map discursive tendencies in the “MensRights” and "TheRedPill" subreddits. The analysis is informed by four dimensions identified in *Mapping the Ideological Landscape of Extreme Misogyny*:

1. **Legitimacy for Violence**  
2. **Adversarial Framing**  
3. **Deterministic vs. Dynamic Ideological Focus**  
4. **Emotional Characteristics**

The project aims to uncover how these dimensions manifest in everyday discourse in a large, active online Reddit community.

## Dataset

All posts and top-level comments were collected from the [Webis-TLDR-17 Hugging Face dataset](https://huggingface.co/datasets/webis/tldr-17). The dataset consists of 3,848,330 preprocessed Reddit posts (submissions & comments) containing the "TL;DR" mention from 2006 to 2016. Multiple subreddits are included, with an average length of 270 words for content, and 28 words for the summary. The variables in the dataset relevant to this study include:

- Post bodies  
- Cleaned post bodies
- Subreddit (name and id)
- Author names (for network analysis)  
- Cleaned summary (extracted from TL;DR mentions)

Using the r/MensRigths subreddit as a starting point, we conducted an Exploratory Data Analysis to pick up subreddits containing posts with similar dynamics. In line with literature, we considered subreddits r/MensRoghts and r/TheRedPill to train our BERTopic model. 

Since our purpose is to analyse how misogyny is expressed in online communities, we only considered the other subreddits as a counter-point of our analysis. 

After preprocessing (removing nulls, duplicates, and empty content), a filtered dataset was used across all analyses: BERTopic Modeling, Sentiment Analysis, and Network Analysis.

## Methods

#### Topic Modeling Setup

- Used environment: for the BERToppic modeling, we used Kaggle notebooks with Python Version 3 and accelerator GPU P100.

### Experiments

#### Initial Preprocessing Steps
- Exploratory Data Analysis: we conducted EDA to depict potential "manosphere" communities, using r/MensRights as a starting point, in line with literature.
- Subreddit Selection: We reduced our dataset to only include posts from the r/AskMen, r/MensRights, r/TheRedPill subreddits.
- Identified and removed rows that were empty, null, or contained only punctuation.
- Identified and removed exact duplicate texts.

#### Summary of Topic Modeling Workflow
We follow Schofield & Mimno (2016) and do not apply stemming or lemmatization prior to topic modeling, since these treatments can reduce interpretability and offer limited gains in English corpora. Instead, we interpret near-synonymous surface forms (e.g., feminist vs. feminists) as belonging to the same conceptual category in our qualitative analysis.

- ##### Pre-calculating Embeddings
First, embeddings for the content were pre-calculated using the SentenceTransformer model "all-MiniLM-L6-v2". This step ensures efficient handling of text data in subsequent processes.

- ##### Preventing Stochastic Behavior
To ensure reproducibility in topic modeling results, the UMAP dimensionality reduction algorithm was configured with a fixed random state. This setup mitigates the stochastic nature of UMAP, providing consistent results across multiple runs.

- ##### Controlling Number of Topics
The number of topics was controlled using HDBSCAN's `min_cluster_size` parameter. Adjusting this parameter influences the number of clusters formed: a higher value results in fewer topics, while a lower value generates more topics.

- ##### Vectorization
A `CountVectorizer` was employed to transform the text data into a numerical format, removing English stop words and considering both unigrams and bigrams.

- ##### Topic Representation Models
Various models inspired by KeyBER and Maximal Marginal Relevance (MMR) were used to diversify and enhance the topic words. These models were combined into a representation model for BERTopic.

- ##### BERTopic Configuration
The BERTopic model was configured with the pre-calculated embedding model, UMAP model, HDBSCAN model, vectorizer model, and the combined representation models. The model was set to identify the top 10 words per topic and was run in verbose mode to provide detailed outputs.

- ##### Topic Modeling Execution
The BERTopic model was fitted and transformed on the content data, using the pre-calculated embeddings to generate topics and their probabilities.

#### Summary of Sentiment Analysis Workflow

In our sentiment analysis experiments, we explored the VADER model. VADER operates using pre-defined sentiment lexicons rather than requiring training data. 

VADER is known for handling informal language, emojis, nuanced sentiments, and sarcasm interpretation. This capability makes VADER particularly well-suited for analyzing social media content and informal text data. Despite not having access to human-labeled data for direct accuracy comparisons during our experiments, VADER consistently showed superior performance in these areas.

Without access to human-labeled data for direct accuracy comparisons, it is challenging to definitively assess which model performed more accurately.

#### Summary of Network Analysis Workflow

- ##### Exploration and Preprocessing of Data
The data used in the first step of the Network Analysis part of the project consisted initially of the complete provided Reddit dataset. This data then underwent the same general filtering and preprocessing criteria enlisted at the beginning of this section, and a sample of 10000 posts was then used for further analysis.

- ##### Network Analysis on selected Subreddits
This part of the analysis was conducted on the resulting data of the previous two analysis steps (Topic+Sentiment Analysiss) on the subreddit data from preprocessing steps. This data underwent the same general filtering and preprocessing criteria enlisted at the beginning of this section.

We plotted the Network Graph, similarly to the previous plot, in which nodes represent different users and node size represents the number of posts per user. Here, edge colors represent different topics and nodes were colored on the basis of the average sentiment of each user's overall posts.

This part of the analysis relied on a random sample (N=1000 posts) drawn from the dataset for visualization purposes.These analyses can be found in the following file: network_analysis.py


## Setup and Installation 

### 1. Clone the Repository 

```bash
git clone https://github.com/yourusername/docana-project-reddit-online-misogyny.git
cd docana-project-reddit-online-misogyny
 ```

### 2. Create and Activate the Environment 
Use the provided Conda environment file:
```bash
conda env create -f environment.yml
conda activate docana-reddit-misogyny
 ```

### 3. (Optional) Run the Dashboard (Streamlit App) locally

```bash
cd dashboard
streamlit run app.py
 ```

## References

- Alexandra Schofield and David Mimno. 2016. Comparing Apples to Apple: The Effects of Stemmers on Topic Models. Transactions of the Association for Computational Linguistics, 4:287–300.

- Buntain, C.; Golbeck, J. Identifying social roles in reddit using network structure. In Proceedings of the 23rd International Conference on World Wide Web, Seoul, Republic of Korea, 7–11 April 2014; pp. 615–620.

- Grootendorst, M., 2022. BERTopic: Neural topic modeling with a class-based TF-IDF procedure. arXiv preprint arXiv:2203.05794.

- Grootendorst, M., 2021. Interactive Topic Modeling with BERTopic. https://towardsdatascience.com/interactive-topic-modeling-with-bertopic-1ea55e7d73d8
  
- Hutto, C., & Gilbert, E. (2014). VADER: A Parsimonious Rule-Based Model for Sentiment Analysis of Social Media Text. Proceedings of the International AAAI Conference on Web and Social Media, 8(1), 216-225.

- Perliger, A., Stevens, C., & Leidig, E. (2023). Mapping the Ideological Landscape of Extreme Misogyny. ICCT Research Paper.

- Roopam Srivastava, Prof. (Dr.) P.K. Bharti, Dr. Parul Verma, (2022). Comparative Analysis of Lexicon and Machine Learning Approach for Sentiment Analysis. (IJACSA) International Journal of Advanced Computer Science and Applications

- Samridh Prasad, reddit-analysis, (2019), GitHub repository, https://github.com/samridhprasad/reddit-analysis
