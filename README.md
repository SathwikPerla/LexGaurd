# ⚖️ LEXGUARD — AI Rights & Contract Intelligence System

> **Don't sign what you don't understand.**  
> LEXGUARD analyzes legal documents and tells you exactly what you're agreeing to — in plain English.

![LEXGUARD](https://img.shields.io/badge/Built%20For-PromptWars%202025-blue)
![Stack](https://img.shields.io/badge/Stack-Next.js%20%2B%20FastAPI%20%2B%20Claude-green)
![GCP](https://img.shields.io/badge/Deployed%20On-Google%20Cloud%20Run-orange)
![Tests](https://img.shields.io/badge/Tests-146%20Passing-brightgreen)

---

## 🧠 What is LEXGUARD?

LEXGUARD is an AI-powered contract intelligence platform that analyzes legal and quasi-legal documents to identify potentially harmful, exploitative, ambiguous, or high-risk clauses — before you agree to them.

Upload any employment contract, SaaS terms of service, rental agreement, privacy policy, or vendor agreement and get a full risk intelligence report in seconds.

This is **not a summarizer**. It is a multi-agent AI reasoning system that thinks like a lawyer on your behalf.

---

## ✨ Features

- **4-Agent Sequential Pipeline** — Extractor → Risk Analyzer → Legal Reasoner → Negotiation Advisor
- **Clause-by-Clause Analysis** — Every clause extracted, labeled, and explained
- **Risk Scoring** — Each clause scored 1–10 with RED / YELLOW / GREEN classification
- **Plain English Explanations** — What each clause actually means for YOU
- **Scenario Consequence Simulation** — "If you sign this and X happens, Y follows"
- **Negotiation Recommendations** — What to push back on, what is predatory vs standard
- **Ambiguity Detection** — Flags contradictions and vague language
- **Benchmark Comparison** — Compares clauses against industry standard benchmarks
- **OCR Support** — Handles scanned PDF documents
- **Accessible UI** — Color + number + text label on every risk card (WCAG AA compliant)

---

## 🏗️ Architecture

```
Document Upload (PDF / DOCX / Scanned)
           ↓
    Document Parser + OCR
           ↓
   ┌───────────────────┐
   │  Agent 1          │  EXTRACTOR
   │  Clause extraction│  Labels clause types, detects ambiguity
   └───────────────────┘
           ↓
   ┌───────────────────┐
   │  Agent 2          │  RISK ANALYZER
   │  Risk scoring     │  Scores 1-10, RED/YELLOW/GREEN, benchmark comparison
   └───────────────────┘
           ↓
   ┌───────────────────┐
   │  Agent 3          │  LEGAL REASONER
   │  Consequences     │  Real-world implications, scenario simulation
   └───────────────────┘
           ↓
   ┌───────────────────┐
   │  Agent 4          │  NEGOTIATION ADVISOR
   │  Recommendations  │  Pushback suggestions, alternative wording
   └───────────────────┘
           ↓
    Unified Risk Report
```

---

## 🛠️ Tech Stack

| Layer | Technology |
|---|---|
| Frontend | Next.js (TypeScript) |
| Backend | FastAPI (Python) |
| AI Core | Claude (Anthropic API) |
| Agent Orchestration | LangGraph |
| Document Parsing | PyMuPDF + python-docx |
| OCR | Google Cloud Vision API |
| Embeddings | Sentence Transformers (all-MiniLM-L6-v2) |
| Vector Database | ChromaDB |
| Deployment | GCP Cloud Run |
| Storage | GCP Cloud Storage |

---

## 🚀 Getting Started

### Prerequisites

- Python 3.11+
- Node.js 18+
- Google Cloud SDK
- Anthropic API key

### Backend Setup

```bash
cd backend
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# Create .env file
cp .env.example .env
# Add your ANTHROPIC_API_KEY to .env

uvicorn main:app --reload --port 8000
```

### Frontend Setup

```bash
cd frontend
npm install

# Create .env.local
echo "NEXT_PUBLIC_API_URL=http://localhost:8000" > .env.local

npm run dev
```

Open [http://localhost:3000](http://localhost:3000)

---

## 🧪 Running Tests

```bash
cd backend
source .venv/bin/activate
pytest tests/ -v
# 146 tests, all passing
```

---

## ☁️ Deployment (GCP Cloud Run)

```bash
# Deploy backend
cd backend
gcloud run deploy lexguard-backend \
  --source . \
  --region us-central1 \
  --allow-unauthenticated \
  --memory 2Gi \
  --min-instances 1 \
  --set-env-vars ANTHROPIC_API_KEY=your-key-here

# Deploy frontend
cd ../frontend
gcloud run deploy lexguard-frontend \
  --source . \
  --region us-central1 \
  --allow-unauthenticated \
  --set-env-vars NEXT_PUBLIC_API_URL=https://your-backend-url.run.app
```

---

## 📋 Example Use Cases

- Detecting restrictive **non-compete clauses** in employment agreements
- Identifying hidden **cancellation penalties** in subscription contracts
- Highlighting broad **IP ownership transfers** in freelance agreements
- Detecting excessive **personal data collection** in privacy policies
- Identifying one-sided **arbitration mechanisms** in platform terms
- Detecting ambiguous **liability limitations** in vendor agreements

---

## ⚠️ Disclaimer

LEXGUARD is an AI-powered tool for informational purposes only. It does **not** constitute legal advice and is **not** a substitute for consultation with a qualified legal professional. Always review contracts with a licensed attorney before signing.

---

## 👨‍💻 Built By

**Sathwik Perla** — Built for PromptWars 2025

---

*LEXGUARD — Know what you sign.*