from lxml.html import fromstring
import translators as ts
import re
from deep_translator import GoogleTranslator
                             
def detect_chinese_characters(text):
    """Detects if there are any Chinese characters in the text."""
    return bool(re.search(r'[\u4e00-\u9fff]', text))

def lxml_unescape(text):
    """Unescapes HTML entities in text."""
    return fromstring(text).text_content()

def translate_text(text):
    """Translates only sentences/paragraphs containing Chinese characters."""
    # Split text into sentences or paragraphs (customize this depending on input format)
    parts = re.split(r'(\n+|(?<=[.!?])\s+)', text)  # Keeps delimiters (newlines and spacing)

    translated_parts = []
    for part in parts:
        if detect_chinese_characters(part):
            # Translate only the part with Chinese characters
            translated_part = GoogleTranslator(source='chinese (simplified)', target='english').translate(part)
            translated_parts.append(translated_part)
        else:
            # Keep non-Chinese parts unchanged
            translated_parts.append(part)

    # Reconstruct the text
    return ''.join(translated_parts)