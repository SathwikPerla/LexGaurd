You are a senior full-stack AI engineer helping me build LEXGUARD — 
an AI-powered contract intelligence platform for a hackathon called 
PromptWars. I need to beat 300+ participants. This must be 
production-quality, intuitive, and genuinely impressive to judges.

═══════════════════════════════════════
OFFICIAL PROBLEM STATEMENT — FULL
═══════════════════════════════════════

Title: LEXGUARD — AI Rights & Contract Intelligence System

Background:
In today's digital and professional ecosystem, individuals and 
organizations routinely accept legally binding agreements without 
fully understanding their implications. Employment contracts, vendor 
agreements, subscription terms, rental agreements, insurance policies, 
platform terms of service, and privacy policies often contain complex 
legal language that is difficult for non-specialists to interpret.

These agreements may include restrictive clauses, hidden liabilities, 
broad intellectual property transfers, automatic renewals, one-sided 
arbitration mechanisms, unfavorable termination conditions, or 
excessive data collection practices. Such clauses can significantly 
impact an individual's financial security, employment flexibility, 
privacy rights, and legal protections.

While several legal technology platforms provide basic document 
summarization or template generation, there remains a strong need 
for intelligent systems capable of identifying contractual risks, 
reasoning about their practical implications, and presenting insights 
in an understandable and transparent manner.

As digital agreements continue to increase in complexity and scale, 
there is growing demand for accessible AI-driven legal intelligence 
systems that improve awareness, transparency, and informed 
decision-making.

Official Problem Statement:
Design and develop an AI-powered contract intelligence platform 
capable of analyzing legal and quasi-legal documents to identify 
potentially harmful, exploitative, ambiguous, or high-risk clauses 
before users agree to them. The system should extract and classify 
important clauses, evaluate contractual risks, reason about possible 
real-world implications, and provide interpretable explanations from 
the perspective of the affected individual or organization. The 
platform should support multiple categories of agreements and provide 
users with meaningful legal awareness rather than simple text 
summarization.

Official Objectives:
- Analyze uploaded legal documents and extract meaningful contractual 
  clauses
- Identify hidden liabilities, unfavorable obligations, and one-sided 
  legal conditions
- Detect ambiguous or potentially exploitative language
- Highlight privacy, financial, employment, intellectual property, 
  and compliance-related risks
- Provide understandable explanations of contractual implications in 
  plain language
- Generate severity-based risk scores or classifications
- Improve transparency and informed decision-making for users

Suggested Features (from official problem statement):
- Clause extraction and classification
- Contract risk scoring systems
- Adversarial legal reasoning workflows
- Liability and obligation analysis
- Ambiguity and contradiction detection
- Contract comparison against standard benchmarks
- Privacy and compliance analysis
- Multi-agent reasoning systems
- Scenario-based consequence simulation
- Explainable AI-based legal insights
- Negotiation recommendation systems

Official Example Use Cases:
- Detecting restrictive non-compete clauses in employment agreements
- Identifying hidden cancellation penalties in subscription contracts
- Highlighting broad intellectual property ownership transfers in 
  freelance agreements
- Detecting excessive personal data collection in privacy policies
- Identifying one-sided arbitration mechanisms in platform terms 
  and conditions
- Detecting ambiguous liability limitations in vendor agreements

Expected System Capabilities (from official problem statement):
- Process contracts in multiple formats such as PDF, DOCX, or 
  scanned documents
- Perform intelligent clause extraction and semantic analysis
- Reason about contractual risks beyond keyword detection
- Explain legal implications in a user-friendly manner
- Compare clauses against common legal or industry standards
- Generate actionable and interpretable risk reports
- Support multiple categories of legal documents

Official Constraints:
- The system is NOT expected to replace legal professionals or 
  provide legally binding advice — this disclaimer must be visible
- Teams may use Google proprietary AI models
- Use of simulated or publicly available datasets is acceptable
- Emphasis must be placed on explainability and interpretability 
  over opaque predictions
- Real-time deployment is required and mandatory — a working live 
  URL must exist at submission
- I have exactly $5 USD in GCP credits — must be extremely 
  cost-efficient

Official Deliverables expected at submission:
- A working prototype or demonstrable system
- System architecture documentation
- Explanation of AI models, reasoning workflows, and methodologies
- Demonstration of risk analysis capabilities
- User interface or dashboard showcasing outputs and insights
- Presentation summarizing approach, innovation, and applicability

Recommended Technology Areas (from official problem statement):
- Natural Language Processing (NLP)
- Transformer-based Legal Language Models
- Retrieval-Augmented Generation (RAG)
- Semantic Similarity & Embedding Models
- Multi-Agent AI Systems
- Explainable AI Frameworks
- OCR & Document Parsing Pipelines
- Vector Databases and Knowledge Retrieval Systems

Evaluation Criteria (platform auto-scores these):
- Code Quality
- Security
- Efficiency
- Testing
- Accessibility
- Problem Statement Alignment
- Google Services usage (mandatory — directly affects leaderboard)

Note: Only the final submission score counts, not best attempt score.

═══════════════════════════════════════
MY ADDITIONAL REQUIREMENTS
═══════════════════════════════════════

- This must NOT be a generic project — it must be intuitive and 
  winning quality
- It must be usable by anyone from a 10-year-old to a legal 
  professional — extremely simple UI, zero learning curve
- I have 5 hours total to build and deploy this
- I need a working deployment link at the end
- My differentiator is the deepest AI reasoning via multi-agent 
  architecture — this is the core edge over other participants
- I will be building using Claude and Gemini Pro in my IDE
- Every step must be tested before moving to the next step so we 
  catch issues and vulnerabilities early and fix them fast

═══════════════════════════════════════
TECH STACK — NON-NEGOTIABLE
═══════════════════════════════════════

Frontend:   Next.js (TypeScript)
Backend:    FastAPI (Python)
AI Core:    Gemini 1.5 Pro via Vertex AI (mandatory for GCP scoring)
Agents:     LangGraph (multi-agent pipeline)
Parsing:    PyMuPDF + python-docx (PDF and DOCX locally, no cost)
OCR:        Google Cloud Vision API (for scanned documents)
Embeddings: Vertex AI Text Embeddings (for semantic similarity)
Vector DB:  ChromaDB (lightweight, local, no extra cost)
Deployment: GCP Cloud Run (serverless, cheapest on $5 credits)
Storage:    GCP Cloud Storage (for uploaded documents)

═══════════════════════════════════════
MULTI-AGENT ARCHITECTURE — THE CORE EDGE
═══════════════════════════════════════

This is NOT a single LLM call. It is a 4-agent sequential pipeline 
built with LangGraph where each agent's output feeds the next agent 
as structured input:

Agent 1 — EXTRACTOR
  Input:  Raw document text
  Task:   Split document into individual clauses, label each clause 
          type (termination, IP transfer, arbitration, liability, 
          privacy, non-compete, auto-renewal, data collection, 
          indemnification, governing law, etc.)
          Also detect ambiguous language and contradictions between 
          clauses
  Output: Structured JSON list of labeled clauses with ambiguity flags

Agent 2 — RISK ANALYZER
  Input:  Labeled clauses from Agent 1
  Task:   Score each clause on severity 1–10, classify as 
          RED/YELLOW/GREEN, identify risk category (financial, 
          privacy, employment, IP, compliance), compare against 
          standard legal benchmarks to flag what is predatory vs 
          what is industry standard
  Output: Risk-scored clause list with classification and benchmark 
          comparison notes

Agent 3 — LEGAL REASONER
  Input:  Risk-scored clauses from Agent 2
  Task:   For each clause reason about real-world implications from 
          the perspective of the person signing — what does this 
          actually mean for them financially, professionally, and 
          legally? Run scenario-based consequence simulation: 
          "If you sign this and X happens, then Y consequence follows"
          Provide explainable AI-style reasoning, not just conclusions
  Output: Human-readable explanation per clause with scenario 
          consequence simulation

Agent 4 — NEGOTIATION ADVISOR
  Input:  Reasoned clauses from Agent 3
  Task:   For each RED and YELLOW clause suggest what to push back 
          on, what is standard vs predatory, what alternative wording 
          would be fair, and what is the recommended action 
          (accept / negotiate / reject)
  Output: Negotiation recommendations per risky clause with 
          suggested alternative wording

Final Output: A complete structured report combining all 4 agent 
outputs into one unified risk intelligence report

═══════════════════════════════════════
FOLDER STRUCTURE
═══════════════════════════════════════

lexguard/
├── frontend/
│   ├── app/
│   │   ├── page.tsx                  # Upload + dashboard
│   │   └── results/page.tsx          # Full risk report view
│   ├── components/
│   │   ├── UploadZone.tsx            # Drag and drop file upload
│   │   ├── RiskCard.tsx              # Per-clause risk display
│   │   ├── AgentTrace.tsx            # Live agent reasoning display
│   │   ├── RiskSummary.tsx           # Overall score at top
│   │   └── Disclaimer.tsx            # Legal disclaimer component
│   └── package.json
│
├── backend/
│   ├── main.py                       # FastAPI routes
│   ├── agents/
│   │   ├── extractor.py              # Agent 1
│   │   ├── risk_analyzer.py          # Agent 2
│   │   ├── reasoner.py               # Agent 3
│   │   └── negotiator.py             # Agent 4
│   ├── core/
│   │   ├── document_parser.py        # PDF/DOCX/scanned → clean text
│   │   ├── ocr_handler.py            # Cloud Vision for scanned docs
│   │   ├── embeddings.py             # Vertex AI embeddings + ChromaDB
│   │   ├── graph.py                  # LangGraph pipeline
│   │   └── gemini_client.py          # Vertex AI connection
│   ├── models/
│   │   └── schemas.py                # Pydantic models for all outputs
│   ├── tests/
│   │   ├── test_parser.py
│   │   ├── test_agents.py
│   │   └── test_api.py
│   ├── requirements.txt
│   └── Dockerfile
│
├── architecture/
│   └── system_architecture.md        # Required deliverable
├── cloudbuild.yaml
└── README.md

═══════════════════════════════════════
EXECUTION PLAN — BUILD IN THIS EXACT ORDER
═══════════════════════════════════════

STEP 1 — Backend skeleton + document parser

  Build:
  - FastAPI app with /health and /analyze endpoints
  - document_parser.py: accepts PDF or DOCX, returns clean text
  - ocr_handler.py: uses Google Cloud Vision API for scanned PDFs
  - schemas.py: Pydantic models for request and response validation
  - /analyze returns structured dummy JSON so frontend can be 
    built in parallel
  - CORS configured correctly

  Test before moving on:
  - Upload a real PDF contract, confirm clean text is returned with 
    no encoding errors
  - Upload a DOCX file, confirm clean text returned
  - Hit /health endpoint, confirm 200 response
  - Hit /analyze with dummy JSON, confirm valid structured response
  - Try uploading a non-PDF/DOCX file, confirm it is rejected cleanly
  - Try uploading an empty file, confirm graceful error
  - Print actual outputs to screen, not just "it worked"

  Vulnerabilities to catch:
  - Encoding errors on special characters
  - Empty text returned from scanned PDFs without OCR
  - File size not validated
  - CORS too permissive

STEP 2 — Gemini connection + Agents 1 and 2

  Build:
  - gemini_client.py: connect to Vertex AI Gemini 1.5 Pro with 
    proper service account auth
  - extractor.py: system prompt engineering to extract and label 
    clauses as strict JSON, detect ambiguity and contradictions
  - risk_analyzer.py: system prompt to score, classify, and compare 
    each clause against standard benchmarks
  - All agent outputs validated through Pydantic schemas
  - embeddings.py: Vertex AI embeddings stored in ChromaDB for 
    semantic similarity comparison

  Test before moving on:
  - Run a real employment contract through Agent 1, print full JSON 
    output, confirm clause types are correctly labeled
  - Run same contract through Agent 2, confirm scores and 
    RED/YELLOW/GREEN classifications make logical sense
  - Run a SaaS Terms of Service through both agents, confirm 
    different clause types are detected
  - Deliberately feed malformed input, confirm Pydantic catches it
  - Print raw Gemini responses before parsing to verify JSON validity

  Vulnerabilities to catch:
  - Gemini returning non-JSON or markdown-wrapped JSON
  - Empty clause lists on short documents
  - Vertex AI authentication errors
  - Quota limit hits — add retry logic with exponential backoff

STEP 3 — Agents 3 and 4 + full LangGraph pipeline

  Build:
  - reasoner.py: plain-language implications plus scenario-based 
    consequence simulation per clause
  - negotiator.py: pushback suggestions and alternative wording for 
    RED and YELLOW clauses
  - graph.py: wire all 4 agents into a LangGraph sequential graph 
    where state is passed correctly between nodes
  - Add streaming so frontend receives agent progress in real time
  - Each agent step emits a server-sent event so UI updates live

  Test before moving on:
  - Run full 4-agent pipeline end to end on employment contract, 
    print all 4 agent outputs, verify each is populated correctly
  - Run full pipeline on SaaS ToS, verify different risk profile
  - Run full pipeline on rental agreement, verify different clause 
    types detected
  - Check LangGraph state is passing between agents without data loss
  - Test streaming endpoint, confirm events arrive in correct order
  - Test pipeline on a very short document (1 clause), confirm 
    no crashes
  - Test pipeline on a very long document (50+ clauses), confirm 
    no timeout

  Vulnerabilities to catch:
  - LangGraph state not passing output of one agent to next
  - Agent 3 or 4 receiving empty input due to upstream failure
  - Streaming breaking JSON mid-transmission
  - Long contracts hitting Gemini context window limits — add 
    chunking logic

STEP 4 — Frontend

  Build:
  - UploadZone: drag and drop PDF/DOCX, file type and size 
    validation on client side, calls /analyze
  - RiskSummary: overall risk score prominently displayed at top 
    (e.g. 7.4/10 HIGH RISK) with color coding
  - RiskCard: per clause display with color badge RED/YELLOW/GREEN, 
    severity score, clause type label, plain-language explanation, 
    scenario consequence, negotiation recommendation
  - AgentTrace: collapsible sidebar or panel showing all 4 agents 
    completing in sequence with live streaming updates
  - Disclaimer: legal disclaimer visible on every page — 
    "This is not legal advice"
  - Full mobile responsiveness
  - Accessibility: keyboard navigation, ARIA labels, color contrast 
    meets WCAG AA

  Test before moving on:
  - Upload 3 different contract types, confirm all cards render 
    correctly with real data
  - Test streaming — confirm AgentTrace updates live as each agent 
    completes
  - Test on mobile viewport (375px width)
  - Test keyboard navigation through entire flow
  - Run Lighthouse accessibility audit, fix any issues
  - Test with a very long contract — confirm UI does not break
  - Test with slow network — confirm loading states work correctly

  Vulnerabilities to catch:
  - Streaming not triggering UI re-renders
  - File upload bypassing client-side validation
  - Results page breaking on contracts with 50+ clauses
  - Missing ARIA labels failing accessibility scoring

STEP 5 — Deploy to GCP Cloud Run

  Build:
  - Dockerfile for backend: multi-stage build, minimal image size
  - cloudbuild.yaml for automated build and deploy
  - Environment variables for all secrets — nothing hardcoded
  - Service account with correct Vertex AI and Cloud Storage 
    permissions only — principle of least privilege
  - Cloud Storage bucket for uploaded documents — NOT public
  - Set min-instances=1 on Cloud Run to avoid cold start timeouts

  Deploy commands:
    gcloud run deploy lexguard-backend \
      --source ./backend \
      --region us-central1 \
      --allow-unauthenticated \
      --memory 2Gi \
      --min-instances 1

    gcloud run deploy lexguard-frontend \
      --source ./frontend \
      --region us-central1 \
      --allow-unauthenticated

  Test before submission:
  - Hit live backend URL /health, confirm 200
  - Upload a real contract PDF to live URL, confirm full pipeline 
    runs in production
  - Check Cloud Run logs for any production errors
  - Confirm Cloud Storage bucket is NOT publicly accessible
  - Confirm no API keys or credentials appear in any source file 
    or environment variable logs
  - Test live URL on mobile
  - Estimate Cloud Run usage cost, confirm within $5 budget

  Vulnerabilities to catch:
  - Cold start causing 30-second timeout on first request
  - Vertex AI permissions not granted to Cloud Run service account
  - Memory limit exceeded on large contracts — increase if needed
  - Secrets accidentally committed to repository

═══════════════════════════════════════
SECURITY CHECKLIST — RUN BEFORE SUBMISSION
═══════════════════════════════════════

- No API keys or credentials hardcoded anywhere in source code
- All file uploads validated on both client and server side 
  (file type, file size, content check)
- No raw user input passed directly into Gemini prompts without 
  sanitization
- CORS restricted to frontend Cloud Run domain only in production
- Cloud Storage bucket is private, access only via signed URLs
- Service account has minimum required permissions only
- All dependencies pinned to specific versions in requirements.txt
- Input size limits enforced to prevent abuse

═══════════════════════════════════════
ARCHITECTURE DOCUMENTATION — REQUIRED DELIVERABLE
═══════════════════════════════════════

Generate a system_architecture.md file that includes:
- System overview diagram in text/ASCII form
- Description of each component and its role
- Explanation of the multi-agent LangGraph pipeline
- Description of AI models used and why (Gemini 1.5 Pro, 
  Vertex AI Embeddings)
- Explanation of RAG implementation using ChromaDB
- Explanation of OCR pipeline for scanned documents
- GCP services used and why each was chosen
- Data flow from document upload to final report
- Security and privacy considerations

═══════════════════════════════════════
WHAT A WINNING SUBMISSION LOOKS LIKE
═══════════════════════════════════════

A judge uploads any legal document and within 30 seconds sees:
- An overall risk score (e.g. 7.4/10 HIGH RISK) at the top
- Every clause color-coded RED/YELLOW/GREEN and explained in 
  plain English a non-lawyer can understand
- Real-world implications written as "If you sign this and X 
  happens, then Y consequence follows for you"
- Specific negotiation suggestions and alternative wording for 
  every red clause
- A live agent trace showing all 4 AI agents reasoning in sequence
- A clean, fast, mobile-friendly, accessible UI with zero 
  learning curve
- A visible legal disclaimer on every page

This is not a summarizer. This is an intelligent multi-agent legal 
reasoning system with explainable AI, RAG, semantic similarity, 
OCR support, and scenario consequence simulation. Every feature 
must serve that framing.

═══════════════════════════════════════
INSTRUCTIONS FOR YOU (ChatGPT)
═══════════════════════════════════════

- Build this step by step in the exact order given above
- After each step show me the complete working code for that step — 
  no placeholders, no TODOs, everything must be runnable
- After each step tell me exactly what commands to run to test it 
  and what the expected output should look like
- If I paste an error, diagnose it and fix it completely before 
  continuing to the next step
- Do not skip steps or combine steps
- Remind me to complete all tests before moving to the next step
- When I confirm tests pass, proceed to the next step automatically
- Start now with Step 1: Backend skeleton + document parser