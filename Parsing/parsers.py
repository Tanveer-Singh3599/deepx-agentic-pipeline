import io
import urllib.request
import pytesseract

# Configure Tesseract explicitly because Windows often doesn't add it to the system PATH.
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

from PIL import Image
from Ingestion.store import get_database, STAGED_COLLECTION_NAME
from ocr_nlp import clean_ocr_text
from chunk import generate_chunks


def img(record):
    """
    Parses an image document using Tesseract for OCR text extraction 
    and reserves a field for semantic image captioning.
    """
        
    record_id = record.get("_id")
    image_url = record.get("cloudinaryUrl", record.get("cloudinary_url"))
    
    if not image_url:
        raise ValueError(f"No image URL found for record {record_id}")

    # 1. Download the image via HTTP into an in-memory byte buffer
    req = urllib.request.Request(image_url, headers={'User-Agent': 'Mozilla/5.0'})
    with urllib.request.urlopen(req) as response:
        img_bytes = response.read()
        
    # Open binary buffer directly as a Pillow Image
    image_obj = Image.open(io.BytesIO(img_bytes))
    
    # 2. Extract OCR Text using PyTesseract
    # Note: Tesseract Engine must be installed on your local OS for this to work natively.
    raw_ocr_text = pytesseract.image_to_string(image_obj)
    
    # Execute NLP cleanup pipeline
    ocr_text = clean_ocr_text(raw_ocr_text)
    
    # 3. Draft Image Caption
    # Tesseract is purely OCR. Authentic generative captioning (e.g. "a blue sports car")
    # typically requires a Vision Model (like HuggingFace BLIP). We store the schema field for it here.
    caption = "VISION_MODEL_CAPTION_PLACEHOLDER"
    
    # Generate unified schema chunks from extracted text
    cleaned_chunks, _ = generate_chunks(raw_ocr_text)
    
    # 4. Inject parsed assets securely back into MongoDB
    db = get_database()
    collection = db[STAGED_COLLECTION_NAME]
    
    # Router.py will still natively update status to 'processed' once we safely return
    collection.update_one(
        {"_id": record_id},
        {"$set": {
            "doc_id": str(record_id),
            "chunks": cleaned_chunks
        }},
        upsert=True
    )
    return cleaned_chunks, raw_ocr_text

def pdfs(record):
    """
    Parses a PDF document natively using PyMuPDF (fitz).
    """
    try:
        import fitz
    except ImportError:
        raise ImportError("Missing dependencies. Please run: pip install pymupdf")
    
    record_id = record.get("_id")
    pdf_url = record.get("cloudinaryUrl", record.get("cloudinary_url"))
    
    if not pdf_url:
        raise ValueError(f"No PDF URL found for record {record_id}")

    # 1. Securely stream the PDF document into an in-memory buffer
    req = urllib.request.Request(pdf_url, headers={'User-Agent': 'Mozilla/5.0'})
    with urllib.request.urlopen(req) as response:
        pdf_bytes = response.read()

    # 2. Extract contents directly using PyMuPDF (fitz)
    full_text = ""
    metadata = {}
    page_count = 0
    cleaned_chunks = []
    chunk_index = 1
    
    with fitz.open(stream=pdf_bytes, filetype="pdf") as doc:
        page_count = doc.page_count
        metadata = doc.metadata
        for page_num, page in enumerate(doc, start=1):
            page_text = page.get_text()
            full_text += page_text + "\n"
            
            # 3. Chunk the text logically based on native paragraphs
            page_chunks, chunk_index = generate_chunks(page_text, start_index=chunk_index, page=page_num)
            cleaned_chunks.extend(page_chunks)
    
    # 4. Process the overall raw text through the NLP standard pipeline
    cleaned_text = clean_ocr_text(full_text)
    
    # 5. Bind the natively extracted variables securely back out to MongoDB
    db = get_database()
    collection = db[STAGED_COLLECTION_NAME]
    
    collection.update_one(
        {"_id": record_id},
        {"$set": {
            "doc_id": str(record_id),
            "chunks": cleaned_chunks
        }},
        upsert=True
    )
    return cleaned_chunks, full_text

def docs(record):
    """
    Parses a Microsoft Word Document (.docx) natively using python-docx.
    """
    try:
        import docx
    except ImportError:
        raise ImportError("Missing dependencies. Please run: pip install python-docx")
        
    record_id = record.get("_id")
    doc_url = record.get("cloudinaryUrl", record.get("cloudinary_url"))
    
    if not doc_url:
        raise ValueError(f"No document URL found for record {record_id}")

    # 1. Securely stream the Word document into an in-memory buffer
    req = urllib.request.Request(doc_url, headers={'User-Agent': 'Mozilla/5.0'})
    with urllib.request.urlopen(req) as response:
        doc_bytes = response.read()

    # 2. Initialize python-docx precisely over the byte stream
    import zipfile
    try:
        word_document = docx.Document(io.BytesIO(doc_bytes))
    except zipfile.BadZipFile:
        raise ValueError("File is not a valid modern Word document (.docx). Older .doc files and incorrect MIME types are unsupported.")
    
    # Aggregate text recursively from all document paragraphs
    full_text = "\n".join([para.text for para in word_document.paragraphs])
        
    # 3. Extract parameter chunks BEFORE NLP pipeline merges paragraph spaces
    cleaned_chunks, _ = generate_chunks(full_text)
    
    # 4. Process the raw text through the standard NLP pipeline
    cleaned_text = clean_ocr_text(full_text)
    
    # 5. Bind the natively extracted variables securely back out to MongoDB
    db = get_database()
    collection = db[STAGED_COLLECTION_NAME]
    
    collection.update_one(
        {"_id": record_id},
        {"$set": {
            "doc_id": str(record_id),
            "chunks": cleaned_chunks
        }},
        upsert=True
    )
    return cleaned_chunks, full_text

def texts(record):
    """
    Parses a plain text document (.txt, .md, .csv etc) natively.
    """
    record_id = record.get("_id")
    text_url = record.get("cloudinaryUrl", record.get("cloudinary_url"))
    
    if not text_url:
        raise ValueError(f"No text URL found for record {record_id}")

    # 1. Securely stream the text document into memory
    req = urllib.request.Request(text_url, headers={'User-Agent': 'Mozilla/5.0'})
    with urllib.request.urlopen(req) as response:
        text_bytes = response.read()

    # Decode bytes securely natively (ignoring broken unicode chars)
    full_text = text_bytes.decode('utf-8', errors='ignore')
        
    # 2. Extract paragraph chunks BEFORE NLP pipeline flattens vertical line breaks
    cleaned_chunks, _ = generate_chunks(full_text)
    
    # 3. Process the raw text through the NLP modular pipeline to fix rogue lines/spacing
    cleaned_text = clean_ocr_text(full_text)
    
    # 4. Bind the natively extracted variables securely back out to MongoDB
    db = get_database()
    collection = db[STAGED_COLLECTION_NAME]
    
    collection.update_one(
        {"_id": record_id},
        {"$set": {
            "doc_id": str(record_id),
            "chunks": cleaned_chunks
        }},
        upsert=True
    )
    return cleaned_chunks, full_text
