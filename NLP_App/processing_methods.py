import spacy
from spacy import displacy
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from scipy import spatial
import gradio as gr
import pandas as pd
import numpy as np
from nltk.sentiment.vader import SentimentIntensityAnalyzer

cosine_similarity = lambda x, y: 1 - spatial.distance.cosine(x, y)

nlp = spacy.load('en_core_web_lg')

sid = SentimentIntensityAnalyzer() # VADER analyzer for estimating sentiment

def get_topic(topic_top_words: list[str]) -> str:
    """
    Estimates generic words for the topic.\n
    Parameters:\n
    - topic_top_words: List of the most popular words for a particular topic.\n
    Returns the string of up to 10 most similar words to the topic.
    """
    topic_words_string = ' '.join(topic_top_words)
    doc = nlp(topic_words_string)
    computed_similarities = []
    for word in nlp.vocab:
        if word.has_vector and word.is_lower and word.is_alpha and not word.is_stop:
            similarity = cosine_similarity(doc.vector, word.vector)
            computed_similarities.append((word, similarity))
    computed_similarities = sorted(computed_similarities, key=lambda item: -item[1])
    similar_topic_words = [similarity[0].text for similarity in computed_similarities][:10]
    similar_topic_words_lemmas = list(set([nlp(tw)[0].lemma_ for tw in similar_topic_words]))
    return ', '.join(similar_topic_words_lemmas)

def analyze_topics(dataframe: pd.DataFrame, text_column_name: str, n_topics: int) -> tuple[np.ndarray, list[list[str]]]:
    """
    Estimates the topics across the corpus of the documents.\n
    Parameters:\n
    - dataframe: Dataframe which contains the corpus.
    - text_column_name: Name of Dataframe column, which contains the corpus.
    - n_topics: Number of wanted topics.\n
    Returns the tuple of topic estimates for each word and a list of lists of 15 top words for each topic.
    """
    texts = dataframe[text_column_name]
    cv = CountVectorizer(max_df=0.95, min_df=2, stop_words='english')
    lda = LatentDirichletAllocation(n_components=n_topics)
    dtm = cv.fit_transform(texts)
    lda.fit(dtm)
    topics_words = []
    for topic in lda.components_:
        topic_top_words = [cv.get_feature_names_out()[i] for i in topic.argsort()[-15:]]
        topics_words.append(topic_top_words)
    topic_results = lda.transform(dtm)
    return (topic_results, topics_words)