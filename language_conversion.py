# pip install googletrans==4.0.0-rc1

from googletrans import Translator

translator = Translator()

def convert_lang(text):
    """
    Converts an English string to Hindi using Google Translate.
    """
    # Special case
    if text.strip() == "Reload the page and try with another better image":
        return "पेज को पुनः लोड करें और एक बेहतर चित्र के साथ पुनः प्रयास करें"
    
    # Translate directly to Hindi
    translated = translator.translate(text, src='en', dest='hi')
    return translated.text
