from processing_methods import *
from spacy import displacy
import random

def vader_to_html(vader_dict: dict) -> str:
    """
    Converts a dictionary to a HTML string. Colors the text depending on the sentiment.\n
    Parameters:
    - vader_dict: A dictionary from vader_processing method.\n
    Returns a string of a HTML text.
    """
    html = "<div style='font-size: 16px; line-height: 1.5;'>"
    for text, emotion in vader_dict.items():
        background_color = get_background_color(emotion)
        html += f"<p style='background-color: {background_color}; padding: 10px; margin: 5px;'>{text}</p>"
    
    html += "</div>"
    
    return html

def get_background_color(emotion: str) -> str:
    """
    Converts a sentiment to the color.\n
    Parameters:
    - emotion: Given sentiment.\n
    Returns a hexadecimal code of the color depending on the sentiment.
    """
    if emotion == 'Positive':
        return '#aaffaa'
    elif emotion == 'Negative':
        return '#ffaaaa'
    elif emotion == 'Neutral':
        return '#cccccc'
    else:
        return '#ffffff'

def generate_random_light_color() -> str:
    """
    Generates random light RGB color (rgb(200+, 200+, 200+)).\n
    Return a string of hexadecimal code of the color.
    """
    red = random.randint(200, 255)
    green = random.randint(200, 255)
    blue = random.randint(200, 255)
    color_code = "#{:02X}{:02X}{:02X}".format(red, green, blue)
    return color_code

def lang_detect_to_html(text: str, lang_dict: dict) -> str:
    """
    Converts language detection results into HTML.\n
    Parameters:
    - text: Source text to process;
    - lang_dict: Dict of spans and corresponding languages.\n
    Returns HTML text as a string.
    """
    html = "<div style='font-size: 16px; line-height: 1.5;'>"
    lang_color_dict = {}
    for span, language in lang_dict.items():
        background_color = ""
        if language not in lang_color_dict.keys():
            background_color = generate_random_light_color()
            lang_color_dict[language] = background_color
        else:
            background_color = lang_color_dict[language]
        html += f"<p style='background-color: {background_color}; padding: 10px; margin: 5px;'>{text[span[0]:span[1]]}</p>"
    for language, color in lang_color_dict.items():
        html += f"<div style='display: inline-block; width: 20px; height: 20px; background-color: {color}; margin: 5px;'></div>"
        html += f"<span style='margin-right: 10px;'>{language}</span>"
    return html

def doc_analysis_display(doc):
    """
    Performs POS, DEP and NER analysis on spacy Doc and renders it in HTML.\n
    Parameters:
    - doc: spacy Doc.\n
    Returns a list[list], HTML string and HTML string.
    """
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
        pos_tokens.extend([[token.text, token.pos_], [" ", None]])
    return (pos_tokens, ent_html, dep_html)

def multilang_analysis_to_html(text: str) -> tuple:
    """
    Splits texts into separate spacy Docs depending on languages and renders POS, NER and Deps for all docs as a whole.\n
    Parameters:
    - text: Source text to process.\n
    Returns a tuple of POS lists, NER HTML string and Deps HTML string.
    """
    docs = pos_ner_multilang(text)
    ents = []
    deps = []
    pos = []
    for doc in docs:
        pt, e, d = doc_analysis_display(doc)
        ents.append(e)
        deps.append(d)
        pos.append(pt)
    ents_render = "".join(ents)
    deps_render = "".join(deps)
    pos_render = sum(pos, [])
    return (pos_render, ents_render, deps_render)