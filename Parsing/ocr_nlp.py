import re
import unicodedata

def clean_ocr_text(text: str) -> str:
    """
    NLP pipeline to deeply clean raw OCR output.
    """
    if not text:
        return ""
        
    # 1. Normalize encoding / remove escape sequences (e.g. \x0c form feed from tesseract)
    text = unicodedata.normalize('NFKD', text).encode('ascii', 'ignore').decode('utf-8')
    text = re.sub(r'[\x00-\x08\x0b-\x0c\x0e-\x1f\x7f-\x9f]', '', text)
    
    # 2. Fix line breaks & spacing
    text = re.sub(r'\n+', '\n', text)
    text = re.sub(r'[ \t]+', ' ', text)
    
    # 3. Normalize characters (OCR-specific fixes)
    text = text.replace('‘', "'").replace('’', "'").replace('”', '"').replace('“', '"')
    text = text.replace('|', 'I') # Common tesseract artifact hallucination
    # Remove spaces before punctuation (e.g., "Hello ," -> "Hello,")
    text = re.sub(r' +([,.:;?])', r'\1', text)
    
    # 4. Strip / clean whitespace
    text = text.strip()
    
    # 5. Optional Spelling Correction (Uncomment and run 'pip install autocorrect' to use)
    # from autocorrect import Speller
    # spell = Speller(lang='en')
    # text = spell(text)
    
    return text
