import spacy
from spacy import displacy
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from scipy import spatial
import pandas as pd
import numpy as np
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import re

cosine_similarity = lambda x, y: 1 - spatial.distance.cosine(x, y)  # for similarity calculation between word vectors

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
            computed_similarities.append((word, similarity))  # will contain similarity with every word in vocab
    computed_similarities = sorted(computed_similarities, key=lambda item: -item[1])  # sort similarities by descending
    similar_topic_words = [similarity[0].text for similarity in computed_similarities][:10]
    similar_topic_words_lemmas = list(set([nlp(tw)[0].lemma_ for tw in similar_topic_words]))  # extract unique words per topic
    return ', '.join(similar_topic_words_lemmas)

def analyze_topics(dataframe: pd.DataFrame, text_column_name: str, n_topics: int) -> tuple[np.ndarray, list[list[str]]]:
    """
    Estimates the topics across the corpus of the documents.\n
    Parameters:\n
    - dataframe: Dataframe which contains the corpus;
    - text_column_name: Name of Dataframe column, which contains the corpus;
    - n_topics: Number of wanted topics.\n
    Returns the tuple of topic estimates for each word and a list of lists of 15 top words for each topic.
    """
    texts = dataframe[text_column_name]
    cv = CountVectorizer(max_df=0.95, min_df=2, stop_words='english')
    lda = LatentDirichletAllocation(n_components=n_topics)
    dtm = cv.fit_transform(texts)
    lda.fit(dtm)  # needed to fit the size of the dataframe
    topics_words = []
    for topic in lda.components_:
        topic_top_words = [cv.get_feature_names_out()[i] for i in topic.argsort()[-15:]]
        topics_words.append(topic_top_words)
    topic_results = lda.transform(dtm)
    return (topic_results, topics_words)

def vader_processing(text: str, mode: str) -> dict:
    """
    Processes the text according to the mode and gives sentiment estimates based on VADER analysis.\n
    Parameters:
    - text: Source text to process;
    - mode: Mode of analysis (whole text, paragraphs, sentences, sentences (with comma separation))\n
    Returns a dictionary with sentiment estimates for parts of text according to the mode.
    """
    ncc = 0.25  # neutral compound cap; the cap which distinguishes Neutral and Positive/Negative in compound score
    vader_dict = {}
    get_sentiment = lambda d: 'Negative' if d['compound'] < -ncc else 'Positive' if d['compound'] > ncc else 'Neutral'
    texts = None
    if mode.lower() == 'whole text':
        scores = sid.polarity_scores(text)
        overall_score = get_sentiment(scores)
        vader_dict[text] = overall_score
    elif mode.lower() == 'paragraphs':
        texts = text.split('\n\n')
    elif mode.lower() == 'sentences':
        doc = nlp(text)
        texts = [sent.text for sent in doc.sents]
    elif mode.lower() == 'sentences (with comma separation)':
        doc = nlp(text)
        texts = [sent.text.split(';') for sent in doc.sents] # get list[list[str]], where list[str] - parts of sentence, separated by ';'
        texts = sum(texts, []) # convert it to list[str]
        texts = [t.split(',') for t in texts] # get list[list[str]], where list[str] - parts of sentence, separated by ','
        texts = [t.strip() for t in texts]
        texts = sum(texts, [])
    if texts:
        scores = [sid.polarity_scores(t) for t in texts]
        overall_scores = [get_sentiment(s) for s in scores]
        for i, text_data in enumerate(texts):
            vader_dict[text_data] = overall_scores[i]
    return vader_dict