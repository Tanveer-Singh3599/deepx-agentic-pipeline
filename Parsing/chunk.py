import re
from ocr_nlp import clean_ocr_text

MAX_TOKENS = 400
OVERLAP_TOKENS = 50

def generate_chunks(text: str, start_index: int = 1, page: int = 1, section: str = "General") -> tuple[list[dict], int]:
    """
    Generates intelligent structured chunks prioritising paragraph bounds.
    If a paragraph breaches MAX_TOKENS, falls back to a sliding window token chunker.
    Returns the compiled structures array and the continuing chunk indexing boundary.
    """
    if not text:
        return [], start_index
        
    # Isolate textual paragraphs
    # Using regex to catch variations like \n \n or \r\n\r\n
    raw_paragraphs = re.split(r'\n\s*\n', text)
    
    structured_chunks = []
    chunk_index = start_index
    section_title = section
    section_num = 0
    
    for raw_para in raw_paragraphs:
        cleaned_para = raw_para.strip()
        if not cleaned_para:
            continue
            
        # Push through the full NLP cleanup pipeline
        cleaned_para = clean_ocr_text(cleaned_para)
            
        # Re-collapse remaining inside newlines to avoid false embedding breaks within a paragraph
        cleaned_para = re.sub(r'\n+', ' ', cleaned_para)
        
        words = cleaned_para.split()
        num_tokens = len(words)
        
        # Simple heuristic for detecting a section header
        if 0 < num_tokens <= 12 and not cleaned_para.endswith(('.', ':', '!', '?')):
            section_title = cleaned_para
            section_num += 1
        
        if num_tokens <= MAX_TOKENS:
            # Paragraph is safe. Keep intact.
            structured_chunks.append({
                "chunk_id": f"c{chunk_index}",
                "text": cleaned_para,
                "embedding": None,
                "metadata": {
                    "section": section_title,
                    "section_number": section_num,
                    "chunk_index": chunk_index,
                    "page": page
                }
            })
            chunk_index += 1
        else:
            # Paragraph breached MAX_TOKEN limit. Trigger fallback window logic.
            step_size = MAX_TOKENS - OVERLAP_TOKENS
            if step_size <= 0:
                step_size = MAX_TOKENS # fail-safe
                
            for window_start in range(0, num_tokens, step_size):
                window_end = window_start + MAX_TOKENS
                window_words = words[window_start:window_end]
                window_text = " ".join(window_words)
                
                structured_chunks.append({
                    "chunk_id": f"c{chunk_index}",
                    "text": window_text,
                    "embedding": None,
                    "metadata": {
                        "section": section_title,
                        "section_number": section_num,
                        "chunk_index": chunk_index,
                        "page": page
                    }
                })
                chunk_index += 1
                
    return structured_chunks, chunk_index
