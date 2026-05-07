# DeepX — Agentic Document Verification Pipeline

## Overview

DeepX is an AI-powered document verification system built during a 24-hour hackathon.
The platform analyzes uploaded documents, extracts factual claims, retrieves supporting evidence, and performs hallucination analysis using an agentic AI workflow.

The project was designed with a modular backend architecture to simulate real-world AI verification pipelines used in RAG evaluation and AI observability systems.

---

# Features

* AI-based claim extraction from documents
* Evidence retrieval pipeline
* Hallucination and credibility analysis
* Agentic workflow orchestration
* Async background task processing
* REST API-based backend services
* Modular and scalable architecture
* Structured schema validation using Pydantic

---

# Tech Stack

## Backend

* Python
* FastAPI
* asyncio
* Pydantic

## AI / Agentic Workflow

* LangGraph
* Gemini API

## Database

* MongoDB

## Other

* REST APIs
* Background workers
* Modular service architecture

---

# System Architecture

```text
                ┌──────────────────┐
                │   User Uploads   │
                │    Document      │
                └────────┬─────────┘
                         │
                         ▼
                ┌──────────────────┐
                │  Claim Extraction │
                └────────┬─────────┘
                         │
                         ▼
                ┌──────────────────┐
                │ Evidence Retrieval│
                └────────┬─────────┘
                         │
                         ▼
                ┌──────────────────┐
                │ Verification Agent│
                └────────┬─────────┘
                         │
                         ▼
                ┌──────────────────┐
                │ Hallucination    │
                │ Analysis Report  │
                └──────────────────┘
```

---

# Project Structure

```bash
deepx/
│
├── Agent/
├── Embedding/
├── Parsing/
├── Schemas/
├── main.py
├── requirements.txt
└── README.md
```

---

# How It Works

1. User uploads a document.
2. The system extracts factual claims from the content.
3. Evidence retrieval modules fetch related contextual data.
4. Verification agents analyze whether claims are supported.
5. The system generates a hallucination analysis report.

---

# Running Locally

## 1. Clone Repository

```bash
git clone <your-repo-url>
cd deepx
```

## 2. Create Virtual Environment

```bash
python -m venv venv
```

### Windows

```bash
venv\Scripts\activate
```

### Linux / Mac

```bash
source venv/bin/activate
```

---

## 3. Install Dependencies

```bash
pip install -r requirements.txt
```

---

## 4. Setup Environment Variables

Create a `.env` file:

```env
MONGO_URI=your_mongodb_uri
GEMINI_API_KEY=your_api_key
```

---

## 5. Run Server

```bash
uvicorn main:app --reload
```

---

# API Endpoints

| Method | Endpoint       | Description              |
| ------ | -------------- | ------------------------ |
| POST   | `/upload`      | Upload document          |
| GET    | `/status/{id}` | Check processing status  |
| GET    | `/report/{id}` | Get hallucination report |

---

# Hackathon Context

This project was built during a 24-hour hackathon by a 5-member team.

My role included:

* designing the overall system architecture,
* planning modular backend workflows,
* integrating independently developed components,
* and leading technical coordination across the team.

AI-assisted coding tools were used to accelerate implementation under strict time constraints.

---

# Future Improvements

* Real vector database integration
* Semantic search with embeddings
* Docker deployment
* Authentication & authorization
* CI/CD pipeline
* Better observability and logging
* Unit and integration tests

---

# Disclaimer

This project was built as a rapid prototype for experimentation and learning purposes during a hackathon. Certain components were simplified to optimize development speed and delivery.

---
