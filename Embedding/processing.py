import re
import hashlib
import json

# --- Flexible Legal Domain Anchors ---
# These are "Must-Have" keywords that indicate a document belongs to our RAG knowledge base.
# Shifting from blocklists to a flexible allowlist for intelligence/speed.
LEGAL_FINANCIAL_ANCHORS = [
    # Core legal terms
    "act", "act,", "section", "subsection", "clause", "subclause", "provision",
    "statute", "regulation", "rule", "rules", "amendment", "amended",
    "notification", "circular", "guideline", "framework", "ordinance",
    "bill", "code", "schedule", "article",

    # Legal process & documents
    "judgment", "order", "order no", "case", "petition", "petitioner",
    "respondent", "plaintiff", "defendant", "appeal", "tribunal",
    "hearing", "verdict", "litigation", "proceedings", "affidavit",
    "writ", "jurisdiction", "bench", "court", "supreme court", "high court",

    # Compliance & governance
    "compliance", "non-compliance", "violation", "penalty", "fine",
    "sanction", "enforcement", "adjudication", "inspection", "audit",
    "disclosure", "filing", "reporting", "governance", "oversight",
    "risk management", "due diligence", "internal control",

    # Regulatory bodies (India-focused)
    "sebi", "rbi", "nclt", "nclat", "irda", "pfrda", "mca",
    "nism", "nsdl", "cdsl", "stock exchange", "exchange", "nse", "bse",

    # Securities & markets
    "securities", "market", "capital market", "primary market", "secondary market",
    "equity", "debt", "derivatives", "futures", "options", "index",
    "trading", "settlement", "clearing", "margin", "volatility",

    # Investment instruments
    "share", "shares", "stock", "bond", "debenture", "mutual fund",
    "etf", "ipo", "fpo", "rights issue", "bonus", "dividend",
    "yield", "return", "portfolio", "asset", "liability",

    # Participants
    "investor", "retail investor", "institutional investor",
    "broker", "sub-broker", "dealer", "trader", "merchant banker",
    "registrar", "custodian", "depository", "issuer",

    # Corporate & company law
    "company", "board", "director", "independent director",
    "shareholder", "stakeholder", "agm", "egm", "resolution",
    "merger", "acquisition", "takeover", "insolvency", "bankruptcy",

    # Financial terms
    "revenue", "profit", "loss", "ebitda", "balance sheet",
    "cash flow", "valuation", "market cap", "liquidity",
    "leverage", "capital", "funding",

    # Misc high-signal phrases
    "advisory", "compliance report", "inspection report",
    "show cause notice", "scn", "order dated", "as per regulation",
    "in accordance with", "subject to", "hereby", "thereof"
]

def clean_text(text: str) -> str:
    """Robust cleaner for legal text. Normalizes whitespace, strips junk, and cleans citations."""
    if not text:
        return ""
        
    # Remove HTML tags (fallback for messy scrapes)
    text = re.sub(r'<[^>]*>', ' ', text)
    
    # Normalize character encodings (common in legal PDFs)
    text = text.replace('\u201c', '"').replace('\u201d', '"').replace('\u2018', "'").replace('\u2019', "'")
    
    # Strip line breaks from citations like "Reg-\nulation"
    text = re.sub(r'-\s*\n\s*', '', text)
    
    # 🕵️ NEW: Smart Legal Filtering for OCR Boilerplate
    # 1. Strip Digital Signatures (e.g. "Digitally signed by... Date: ... IST")
    text = re.sub(r'(?i)digitally\s+signed\s+by.*?\d{2}/\d{2}/\d{4}.*?IST', '', text)
    
    # 2. Strip standard OCR Timestamps (e.g. "Date: 05/06/2025 13:15:17 IST")
    text = re.sub(r'(?i)date:\s*\d{2}/\d{2}/\d{4}\s*\d{2}:\d{2}:\d{2}\s*IST', '', text)
    
    # 3. Strip PAN cards and generic alphanumeric IDs (approximate)
    text = re.sub(r'\b[A-Z]{5}[0-9]{4}[A-Z]{1}\b', '[PAN_REDACTED]', text)
    
    # 4. Strip generic "Page X of Y" or "Verified" boilerplate
    text = re.sub(r'(?i)page\s+\d+\s+of\s+\d+', '', text)
    text = re.sub(r'(?i)verify\s+this\s+on.*?\s+', '', text)
    
    # Collapse multiple spaces and various newline sequences
    text = re.sub(r'\s+', ' ', text)
    
    return text.strip()

def is_meaningful_legal_content(text: str) -> bool:
    """
    Evaluates semantic density to filter out noise/metadata chunks.
    Heuristic: Expect reasonable alphabetic density and word count.
    """
    if not text or len(text.strip()) < 50:
        return False
        
    # 1. Semantic Density Check: alpha_chars / total_chars
    alpha_count = sum(c.isalpha() for c in text)
    density = alpha_count / len(text)
    
    if density < 0.6: # Too many numbers/symbols usually indicates noise/tables/IDs
        return False
        
    # 2. Word Count Check: Ensure it's more than a single fragment
    words = text.split()
    if len(words) < 15:
        return False
        
    # 3. Boilerplate keywords check (exclusionary)
    if any(keyword in text.lower() for keyword in ["digitally signed", "verification code"]):
        return False
        
    return True

def has_legal_domain_anchors(text: str) -> bool:
    """
    Flexible Domain Filter: Only returns True if the text contains 
    at least three unique domain-specific anchor words.
    """
    if not text:
        return False
        
    t = text.lower()
    return sum(1 for anchor in LEGAL_FINANCIAL_ANCHORS if anchor in t) >= 3

def chunk_text(text: str, chunk_size: int = 400, overlap: int = 50) -> list[dict]:
    """Paragraph-aware chunking with sliding window and deterministic SHA256 IDs."""
    if not text:
        return []
        
    # Split into rough paragraphs (look for double spacing or generic terminators)
    raw_blocks = re.split(r'(?<=\. )\s+|\n\n', text)
    
    chunks = []
    current_block = []
    current_token_count = 0
    
    for block in raw_blocks:
        block = block.strip()
        if not block:
            continue
            
        words = block.split()
        block_len = len(words)
        
        # If a single block is massive, split it forcibly
        if block_len > chunk_size:
            for i in range(0, block_len, chunk_size - overlap):
                sub_words = words[i:i + chunk_size]
                sub_text = " ".join(sub_words)
                chunks.append(sub_text)
            continue
            
        # Normal accumulation
        if current_token_count + block_len > chunk_size:
            # Ship current buffer
            chunks.append(" ".join(current_block))
            # Start new buffer with overlap from previous
            overlap_words = current_block[-overlap:] if len(current_block) > overlap else current_block
            current_block = overlap_words + words
            current_token_count = len(current_block)
        else:
            current_block.extend(words)
            current_token_count += block_len
            
    if current_block:
        chunks.append(" ".join(current_block))
        
    # Deduplicate and build final mapped records with IDs
    final_records = []
    seen_hashes = set()
    
    for i, chunk_text in enumerate(chunks):
        chunk_text = chunk_text.strip()
        if not chunk_text:
            continue
            
        # Deterministic SHA256 mapping
        h = hashlib.sha256(chunk_text.encode()).hexdigest()
        if h in seen_hashes:
            continue
        
        seen_hashes.add(h)
        final_records.append({
            "_id": h,
            "text": chunk_text,
            "index": i
        })
        
    return final_records

if __name__ == "__main__":
    test_text = "Regulation 12. Margin Maintenance. Brokers must maintain 20% margin. This is a very long sentence to test the chunking and normalization logic across different boundaries."
    cleaned = clean_text(test_text)
    processed = chunk_text(cleaned, chunk_size=10)
    print(f"Generated {len(processed)} chunks.")
    for p in processed:
        print(f"ID: {p['_id'][:10]}... | Text: {p['text']}")
