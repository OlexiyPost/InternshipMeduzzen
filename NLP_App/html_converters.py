from processing_methods import nlp
from spacy import displacy

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

def ent_dep_to_html(text: str):
    """
    Performs POS, DEP and NER analysis and renders it in HTML.\n
    Parameters:
    - text: Source text to analyze.\n
    Returns a spacy.Doc, HTML string and HTML string.
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
    return (doc, ent_html, dep_html)