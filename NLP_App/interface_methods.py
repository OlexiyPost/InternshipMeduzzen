import gradio as gr
import pandas as pd

from processing_methods import *
from html_converters import vader_to_html, lang_detect_to_html, multilang_analysis_to_html, doc_analysis_display

def model_topics(text_column_name: str, n_topics: int, file_obj: gr.File) -> pd.DataFrame:
    """
    Assigns a topic to each record in dataframe data.\n
    Parameters:\n
    - text_column_name: Name of Dataframe column, which contains the corpus.
    - n_topics: Number of wanted topics.
    - file_obj: File, which contains the dataframe.\n
    Returns the updated dataframe for output.
    """
    df = pd.read_csv(file_obj)
    tr, tw = analyze_topics(df, text_column_name, int(n_topics))
    topics = [get_topic(topic_bag_of_words) for topic_bag_of_words in tw]
    individual_topics = [topics[index] for index in tr.argmax(axis=1)]
    df['Topic'] = individual_topics
    return df

def extract_dataframe(file_obj: gr.File) -> pd.DataFrame:
    """
    Reads dataframe to predefined variable.\n
    Parameters:\n
    - file_obj: file object.\n
    Returns the read dataframe.
    """
    data = pd.read_csv(file_obj)
    return data

def vader_analysis(text: str, mode: str) -> gr.HTML:
    """
    Estimates sentiment of the text via VADER.\n
    Parameters:\n
    - text: Source text to analyze;
    - mode: Mode of analysis (whole text, paragraphs, sentences, sentences (with comma separation)).\n
    Returns a dataframe of sentiment estimate based on compound score.
    """
    vd = vader_processing(text, mode)
    return vader_to_html(vd)

def vader_dataframe_analysis(text_column_name: str, file_obj: gr.File) -> pd.DataFrame:
    """
    Assigns a sentiment estimate to each record in dataframe data.\n
    Parameters:\n
    - text_column_name: Name of Dataframe column, which contains the corpus.\n
    Returns the updated dataframe for output.
    """
    df = pd.read_csv(file_obj)
    df['Sentiment'] = df[text_column_name].apply(lambda text: list(vader_processing(text, mode='whole text').values()))
    return df

def multilang_analysis(text: str) -> gr.HTML:
    """
    """
    ld = multilang_processing(text)
    detected_langs_render = lang_detect_to_html(text, ld)
    highlighted_pos, ents_render, deps_render = multilang_analysis_to_html(text)
    return (detected_langs_render, highlighted_pos, ents_render, deps_render)

def launch_gui():
    """
    Models and launches Gradio GUI.
    """
    with gr.Blocks() as gui:
        with gr.Tab("Multilingual analysis"):
            gr.Interface(fn=multilang_analysis,
                         inputs=gr.Textbox(label='Enter your text'), 
                         outputs=[gr.HTML(label='Detected languages'),
                                  gr.HighlightedText(label='POS analysis'), 
                                  gr.HTML(label='NER analysis'), 
                                  gr.HTML(label='Dependencies')])
        with gr.Tab("VADER analysis"):
            gr.Interface(fn=vader_analysis, 
                         inputs=[gr.Textbox(label='Enter your text'),
                                 gr.Dropdown(label='Analysis mode',
                                             choices=['Whole Text', 
                                                      'Paragraphs', 
                                                      'Sentences', 
                                                      'Sentences (with comma separation)',
                                                      'Clusters'])],
                         outputs=gr.HTML(label='Sentiment estimate'))
        with gr.Tab("Topic Modeling"):
            gr.Interface(fn=model_topics, 
                         inputs=[gr.Textbox(label='Column with texts'), 
                                 gr.Number(label='Number of desired topics'), 
                                 gr.File(label='Your dataframe with text data')], 
                         outputs=gr.DataFrame(label='Dataframe with topics'))
        with gr.Tab("VADER Dataframe analysis"):
            gr.Interface(fn=vader_dataframe_analysis, 
                         inputs=[gr.Textbox(label='Column with texts'), 
                                 gr.File(label='Your dataframe with text data')], 
                         outputs=gr.DataFrame(label='Dataframe with sentiments'))
    gui.launch()