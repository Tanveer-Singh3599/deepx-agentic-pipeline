from pydantic import BaseModel, Field, HttpUrl, ConfigDict
from pydantic.alias_generators import to_camel
from pydantic import field_serializer
from typing import List, Optional, Dict, Any

class FileUploadRecord(BaseModel):
    # This configuration automatically maps camelCase JSON keys to Python's snake_case
    model_config = ConfigDict(
        alias_generator=to_camel,
        populate_by_name=True
    )

    id: str = Field(..., alias="_id",description="Mongodb object Id")
    uploaded_by: str = Field(..., description="User ID of the uploader")
    sector: str = Field(...)
    original_name: str = Field(...)
    mime_type: str = Field(...)
    size_bytes: int = Field(..., ge=0, description="File size in bytes. Must be a positive integer.")
    file_hash: str = Field(...)
    cloudinary_url: HttpUrl = Field(..., description="Validates that the input is a properly formatted URL")
    cloudinary_id: str = Field(...)
    
    # Fields with default values (matching your 'queued' and 1)
    status: str = Field(default="queued")
    version: int = Field(default=1)
    supported: bool = Field(default=False)

    # Serializer to convert the URL object to a string automatically
    @field_serializer('cloudinary_url')
    def serialize_url(self, url: HttpUrl) -> str:
        return str(url)

# --- New Hallucination Detection Schemas ---

class DocumentMetadata(BaseModel):
    primary_doc_id: str
    uploaded_by: str
    analysis_timestamp: str
    status: str

class Metrics(BaseModel):
    total_chars: int
    hallucinated_chars: int
    unverified_chars: int
    hallucination_rate_percentage: float
    unverified_rate_percentage: float

class AdvancedAnalysis(BaseModel):
    hallucination_type: Optional[str] = None
    context_drift_warning: Optional[str] = None
    agent_reasoning: str = Field(..., description="Highly verbose, exhaustive analytical reasoning. Use Markdown for structure.")

class TraceabilityEntry(BaseModel):
    chunk_id: str
    evidence_text: str
    cloudinary_url: Optional[str] = None
    page_number: Optional[int] = None
    section: Optional[str] = None

class CharSpan(BaseModel):
    start: int
    end: int

class Verdict(BaseModel):
    claim_id: str
    claim_text: str
    char_span: CharSpan
    status: str # SUPPORTED | CONTRADICTED | UNVERIFIED
    verification_source: str # LOCAL_RAG | WEB_SEARCH_ESCALATION
    is_flagged: bool = Field(False, description="Set to True if the claim is a severe hallucination or high-risk contradiction.")
    flag_reason: Optional[str] = Field(None, description="Detailed reason for flagging this specific content.")
    advanced_analysis: AdvancedAnalysis
    traceability_matrix: List[TraceabilityEntry]

class AnalysisReport(BaseModel):
    document_metadata: DocumentMetadata
    executive_summary: str = Field(..., description="Highly verbose, multi-paragraph analytical summary for professional auditors. Use Markdown formatting.")
    metrics: Metrics
    verdicts: List[Verdict]
    