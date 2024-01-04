import spacy
from spacy import displacy
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from scipy import spatial
import gradio as gr
import pandas as pd
from nltk.sentiment.vader import SentimentIntensityAnalyzer

cosine_similarity = lambda x, y: 1 - spatial.distance.cosine(x, y)

nlp = spacy.load('en_core_web_lg')

sid = SentimentIntensityAnalyzer() # VADER analyzer for estimating sentiment

data = None # Dataframe with the data to work with

def get_topic(topic_top_words):
    """
    Estimates generic words for the topic.\n
    Parameters:\n
    - topic_top_words: List[str];
        List of the most popular words for a particular topic.\n
    Returns the string of 5 most similar words to the topic.
    """
    topic_words_string = ' '.join(topic_top_words)
    doc = nlp(topic_words_string)
    computed_similarities = []
    for word in nlp.vocab:
        if word.has_vector:
            if word.is_lower:
                if word.is_alpha:
                    if not word.is_stop:
                        similarity = cosine_similarity(doc.vector, word.vector)
                        computed_similarities.append((word, similarity))
    computed_similarities = sorted(computed_similarities, key=lambda item: -item[1])
    similar_topic_words = [similarity[0].text for similarity in computed_similarities]
    return ', '.join(similar_topic_words[:5])

def analyze_topics(dataframe, text_column_name, n_topics):
    """
    Estimates the topics across the corpus of the documents.\n
    Parameters:\n
    - dataframe: pandas.Dataframe;
        Dataframe which contains the corpus.
    - text_column_name: str;
        Name of Dataframe column, which contains the corpus.
    - n_topics: int;
        Number of wanted topics.\n
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

def model_topics(text_column_name, n_topics):
    """
    Assigns a topic to each record in dataframe data.\n
    Parameters:\n
    - text_column_name: str;
        Name of Dataframe column, which contains the corpus.
    - n_topics: int;
        Number of wanted topics.\n
    Returns the updated dataframe for output.
    """
    topics_number = int(n_topics)
    tr, tw = analyze_topics(data, text_column_name, topics_number)
    topics = [get_topic(topic_bag_of_words) for topic_bag_of_words in tw]
    individual_topics = [topics[index] for index in tr.argmax(axis=1)]
    df = pd.DataFrame(data)
    df['Topic'] = individual_topics
    return df

def extract_dataframe(file_obj):
    """
    Reads dataframe to predefined variable.\n
    Parameters:\n
    - file_obj: file object;
        Opened file object.\n
    Returns the read dataframe.
    """
    global data
    data = pd.read_csv(file_obj)
    return data

def pos_ner_analysis(text):
    """
    Renders POS, dependencies and NER for given text.\n
    Parameters:\n
    - text: str;
        Source text to analyze.\n
    Returns a list of tokens, html render with NER and html render with dependencies.
    """
    doc = nlp(text)
    dep_html = displacy.render(doc, style="dep", page=True)
    dep_html = (
        "<div style='max-width:100%; max-height:360px; overflow:auto'>"
        + dep_html
        + "</div>"
    )
    ent_html = displacy.render(doc, style="ent", page=True)
    ent_html = (
        "<div style='max-width:100%; max-height:360px; overflow:auto'>"
        + ent_html
        + "</div>"
    )
    pos_tokens = []
    for token in doc:
        pos_tokens.extend([(token.text, token.pos_), (" ", None)])
    return pos_tokens, ent_html, dep_html

def vader_analysis(text):
    """
    Estimates sentiment of the text via VADER.\n
    Parameters:\n
    - text: str;
        Source text to analyze.\n
    Returns a string of overall sentiment estimate based on compound score.
    """
    scores = sid.polarity_scores(text)
    overall_score = "Neutral"
    if scores['compound'] < -0.25:
        overall_score = "Negative"
    elif scores['compound'] > 0.25:
        overall_score = "Positive"
    return overall_score

def vader_dataframe_analysis(text_column_name):
    """
    Assigns a sentiment estimate to each record in dataframe data.\n
    Parameters:\n
    - text_column_name: str;
        Name of Dataframe column, which contains the corpus.\n
    Returns the updated dataframe for output.
    """
    #scores = pd.Series([sid.polarity_scores(data[text_column_name][index]) for index in data.index])
    #scores = scores.apply(lambda item: 'Positive' if item['compound'] > 0.25 else 'Negative' if item['compound'] < -0.25 else 'Neutral')
    df = pd.DataFrame(data)
    df['Sentiment'] = data[text_column_name].apply(lambda text: vader_analysis(text))
    return df

def launch_gui():
    """
    Models and launches Gradio GUI.
    """
    with gr.Blocks() as gui:
        with gr.Tab("POS and NER analysis"):
            gr.Interface(fn=pos_ner_analysis, 
                         inputs=gr.Textbox(label='Enter your text'), 
                         outputs=["highlight", "html", "html"],
                         examples=[["What a beautiful morning for a walk!"],
                                   ["It was the best of times, it was the worst of times."]])
        with gr.Tab("VADER analysis"):
            gr.Interface(fn=vader_analysis, 
                         inputs=gr.Textbox(label='Enter your text'), 
                         outputs=gr.Textbox(label='Sentiment estimate'),
                         examples=[["What a beautiful morning for a walk!"],
                                   ["It was the best of times, it was the worst of times."]])
        with gr.Tab("Extraction"):
            gr.Interface(fn=extract_dataframe, inputs=gr.File(), outputs=gr.DataFrame())
        with gr.Tab("Topic Modeling"):
            gr.Interface(fn=model_topics, 
                         inputs=[gr.Textbox(label='Column with texts'), gr.Number(label='Number of desired topics')], 
                         outputs=gr.DataFrame())
        with gr.Tab("VADER Dataframe analysis"):
            gr.Interface(fn=vader_dataframe_analysis, inputs=gr.Textbox(label='Column with texts'), outputs=gr.DataFrame())
    gui.launch()

launch_gui()