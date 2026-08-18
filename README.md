# AI Academic Advisor — CockroachDB State Course Registration System

<div align="center">

![CockroachDB State AI Advisor](https://img.shields.io/badge/CockroachDB%20State-AI%20Advisor-1E407C?style=for-the-badge)
![CockroachDB](https://img.shields.io/badge/CockroachDB-Cloud-6933FF?style=for-the-badge&logo=cockroachdb)
![AWS S3](https://img.shields.io/badge/AWS-S3-FF9900?style=for-the-badge&logo=amazons3)
![Gemini](https://img.shields.io/badge/Google-Gemini%203.5-4285F4?style=for-the-badge&logo=google)
![Ollama](https://img.shields.io/badge/Ollama-Local%20LLM-black?style=for-the-badge)
![Python](https://img.shields.io/badge/Python-3.11+-3776AB?style=for-the-badge&logo=python)

**A full-stack AI-powered academic advising platform for university course registration, built with CockroachDB, AWS S3, and a multi-tier LLM fallback engine.**

[Live Demo](#running-locally) · [Architecture](#architecture) · [CockroachDB Integration](#cockroachdb-integration) · [AWS Integration](#aws-integration)

</div>

---

## 📋 Table of Contents

- [Overview](#overview)
- [Architecture](#architecture)
- [Features](#features)
- [Technology Stack](#technology-stack)
- [CockroachDB Integration](#cockroachdb-integration)
- [AWS Integration](#aws-integration)
- [AI Agent & LLM Fallback Chain](#ai-agent--llm-fallback-chain)
- [Database Schema](#database-schema)
- [Installation & Setup](#installation--setup)
- [Running Locally](#running-locally)
- [Student Profiles & Major Permissions](#student-profiles--major-permissions)
- [API Reference](#api-reference)
- [Performance Optimizations](#performance-optimizations)
- [Project Structure](#project-structure)

---

## Overview

The **AI Academic Advisor** is a hackathon submission that reimagines the university course registration experience. Students interact with an AI-powered academic advisor that understands their major, academic history, career goals, and course prerequisites — then recommends the best courses for the upcoming semester from a **live catalog of 8,983 courses across 242 departments**.

The system is built on **CockroachDB Cloud Serverless** as the primary persistent memory engine, **AWS S3** for durable report storage, and a **4-tier LLM fallback chain** (Gemini → Ollama local → Mistral Cloud → Direct vector results) ensuring zero-downtime AI responses.

---

## Architecture

See [`ARCHITECTURE-3.pdf`](./ARCHITECTURE-3.pdf) for the full system architecture diagram.

```
┌─────────────────────────────────────────────────────────────┐
│                    STUDENT BROWSER                          │
│  ┌───────────────┐  ┌─────────────────┐  ┌──────────────┐  │
│  │  Login Page   │  │  Course Catalog  │  │  AI Chat     │  │
│  │  (Major Pick) │  │  (8,983 courses) │  │  Panel       │  │
│  └───────────────┘  └─────────────────┘  └──────────────┘  │
└───────────────────────────┬─────────────────────────────────┘
                            │ HTTPS
┌───────────────────────────▼─────────────────────────────────┐
│              FLASK APP SERVER (Hosted on AWS EC2)           │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌───────────┐   │
│  │ /api/chat│  │/api/enroll│  │/api/drop │  │/api/student│  │
│  └────┬─────┘  └────┬──────┘  └────┬─────┘  └─────┬─────┘  │
└───────┼─────────────┼──────────────┼───────────────┼────────┘
        │             │              │               │
┌───────▼─────────────▼──────────────▼───────────────▼────────┐
│                    AGENT ENGINE (agent.py)                   │
│                                                              │
│  Tier 1: Google Gemini 3.5 Flash-Lite  (~2s)                │
│  Tier 2: Ollama Local (mistral:instruct / llama3.1:8b) (~5s)│
│  Tier 3: Mistral Cloud API             (~4s)                 │
│  Tier 4: Direct CockroachDB vector results  (<1s)           │
└───────────────────────┬──────────────────────────────────────┘
                        │
        ┌───────────────┼────────────────┐
        │               │                │
┌───────▼───────┐ ┌─────▼────────┐ ┌────▼──────────┐
│  CockroachDB  │ │    AWS S3    │ │  Ollama Local  │
│  Cloud Server │ │   Bucket     │ │  localhost:    │
│  (5 memory    │ │  (Reports)   │ │  11434         │
│   types)      │ │              │ │                │
└───────────────┘ └──────────────┘ └────────────────┘
```

---

## Features

| Feature | Description |
|:---|:---|
| 🎓 **Full Course Catalog** | Browse all 8,983 courses across 242 departments |
| 🔒 **Major Discipline Permissions** | Enrollment restricted to your major's allowed departments |
| 🤖 **AI Academic Advisor** | Semantic course recommendations powered by vector embeddings |
| 📚 **Academic History Protection** | Completed past-term courses cannot be dropped (permanent record) |
| ⚡ **Two-Level Search Cache** | CockroachDB `search_cache` table + LRU in-process cache for sub-3s responses |
| 🔄 **4-Tier LLM Fallback** | Gemini → Ollama (local) → Mistral → Direct results — always responds |
| 📊 **S3 Report Storage** | Every AI recommendation session saved to AWS S3 as structured JSON |
| 🧠 **5 Memory Types** | CockroachDB stores conversation history, user context, task state, vectors, and transactional data |

---

## Technology Stack

| Layer | Technology |
|:---|:---|
| **Backend** | Python 3.11+, Flask 3.0 |
| **Primary Database** | CockroachDB Cloud Serverless (PostgreSQL wire protocol) |
| **Vector Search** | pgvector extension on CockroachDB, `all-MiniLM-L6-v2` embeddings |
| **Cloud Storage** | AWS S3 (boto3) |
| **LLM — Primary** | Google Gemini 3.5 Flash-Lite (`google-genai`) |
| **LLM — Fallback 1** | Ollama local (`mistral:instruct`, `llama3.1:8b`) |
| **LLM — Fallback 2** | Mistral Cloud API (`mistral-small-latest`) |
| **Frontend** | Vanilla HTML/CSS/JavaScript, responsive 3-column academic layout |
| **Architecture** | Single-shot agent with pre-fetched tool results (vs multi-turn agentic loop) |
| **Deployment** | Hosted on AWS EC2 |

---

## CockroachDB Integration

CockroachDB Cloud Serverless is the **primary persistence engine** for this application, implementing **5 distinct memory types** as described in `ARCHITECTURE-3.pdf`:

### Connection

```python
# db.py — connects via PostgreSQL wire protocol (psycopg3)
def get_db_connection():
    db_url = os.environ.get("COCKROACHDB_URL")
    return psycopg.connect(db_url, row_factory=dict_row)
```

### 5 Memory Types

| # | Memory Type | Table | Purpose |
|:--|:---|:---|:---|
| 1 | **Conversation History** | `conversations` | Stores every user/assistant chat turn by `session_id` for multi-turn context |
| 2 | **User Context** | `students`, `student_courses` | Student profile, major, completed courses, grades |
| 3 | **Task State** | `recommendation_sessions` | Lifecycle state of each AI session (`in_progress` → `complete`) |
| 4 | **Vector Embeddings** | `courses.embedding` (pgvector) | 384-dim course embeddings for semantic cosine similarity search |
| 5 | **Transactional Data** | `courses`, `departments`, `student_courses` | Real-time enrollments, drops, and full course catalog |

### CockroachDB Tools (3 Tool Pattern)

```python
# tools.py
def mcp_check_eligibility(student_id, course_id)   # MCP Server: prerequisite graph query
def vector_semantic_search(goal_text, limit, allowed_depts)  # pgvector cosine similarity
def ccloud_health_check(cluster_name)               # ccloud CLI cluster monitoring
```

### Two-Level Search Cache

The system implements a two-level cache to eliminate repeated pgvector scan latency (~1100ms per query):

```python
# L1: In-process LRU (instant, Python process lifetime)
@lru_cache(maxsize=256)
def _cached_encode(text: str): ...

# L2: CockroachDB search_cache table (2ms lookup, 24h TTL, survives restarts)
CREATE TABLE search_cache (
    query_hash   TEXT PRIMARY KEY,
    query_text   TEXT NOT NULL,
    results_json TEXT NOT NULL,
    hit_count    INT  DEFAULT 1,
    expires_at   TIMESTAMPTZ DEFAULT current_timestamp() + INTERVAL '24 hours'
);
```

---

## AWS Integration

### Amazon S3 — Recommendation Report Storage

Every AI advisor session generates a structured JSON report automatically saved to S3:

```python
# s3_utils.py
def save_recommendation_report(student_id, session_id, goal, recommendation_text):
    report = {
        "student_id": student_id,
        "session_id": session_id,
        "goal": goal,
        "recommendation": recommendation_text,
        "timestamp": datetime.utcnow().isoformat()
    }
    s3_client.put_object(
        Bucket=bucket_name,
        Key=f"reports/student_{student_id}/session_{session_id}.json",
        Body=json.dumps(report, indent=2),
        ContentType='application/json'
    )
```

Reports are organized in S3 as:
```
s3://ai-course-advisor-nihal/
└── reports/
    ├── student_student_123/
    │   ├── session_<uuid>.json
    │   └── session_<uuid>.json
    ├── student_student_456/
    └── student_student_789/
```

S3 uploads run **asynchronously in a background thread** so they never block the HTTP response:

```python
# app.py — async fire-and-forget S3 save after response is returned
threading.Thread(target=_post_save, daemon=True).start()
return jsonify({...})  # returns immediately to the browser
```

---

## AI Agent & LLM Fallback Chain

The agent uses a **single-shot architecture** (vs multi-turn agentic loop) for maximum speed:

```
OLD (multi-turn):  Gemini call → Tool call → Gemini call → Gemini JSON call  (~15s)
NEW (single-shot): Python pre-fetch tools in parallel → 1 Gemini call         (~2s)
```

### Fallback Chain

```
1. Google Gemini 3.5 Flash-Lite  [PRIMARY — ~2s]
        ↓ fails (rate limit / quota)
2. Ollama Local — mistral:instruct  [FALLBACK 1 — ~5s, zero cost, no network]
        ↓ try llama3.1:8b if mistral fails
        ↓ both fail (Ollama not running)
3. Mistral Cloud API  [FALLBACK 2 — ~4s]
        ↓ fails (no API key / network down)
4. Direct CockroachDB vector results  [FALLBACK 3 — <1s, no LLM]
```

### Parallel Pre-fetching

Tool results are fetched **concurrently** before the LLM call using `ThreadPoolExecutor`:

```python
with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
    f_vec = pool.submit(vector_semantic_search, goal, 6, allowed_depts)
    f_mcp = pool.submit(mcp_check_eligibility, student_id, "")
    vector_courses = f_vec.result(timeout=8)
    mcp_info = f_mcp.result(timeout=8)
# → Feed both results into a single Gemini call
```

---

## Database Schema

```sql
-- Student profiles and enrollment history
CREATE TABLE students (
    id TEXT PRIMARY KEY,
    name TEXT,
    email TEXT,
    created_at TIMESTAMPTZ DEFAULT now()
);

-- Course catalog with pgvector embeddings (8,983 courses)
CREATE TABLE courses (
    id TEXT PRIMARY KEY,
    title TEXT,
    description TEXT,
    department_prefix TEXT,
    min_credits INT,
    max_credits INT,
    number INT,
    embedding VECTOR(384)  -- all-MiniLM-L6-v2 embeddings
);

-- 242 academic departments
CREATE TABLE departments (
    prefix TEXT PRIMARY KEY,
    name TEXT
);

-- Student enrollments and academic history
CREATE TABLE student_courses (
    student_id TEXT,
    course_id TEXT,
    semester TEXT,
    grade TEXT,
    PRIMARY KEY (student_id, course_id)
);

-- Multi-turn conversation memory (Memory Type 1)
CREATE TABLE conversations (
    id UUID DEFAULT gen_random_uuid() PRIMARY KEY,
    session_id TEXT,
    turn INT,
    role TEXT,
    content TEXT,
    created_at TIMESTAMPTZ DEFAULT now()
);

-- AI session lifecycle tracking (Memory Type 3)
CREATE TABLE recommendation_sessions (
    id TEXT PRIMARY KEY,
    student_id TEXT,
    status TEXT DEFAULT 'in_progress',
    created_at TIMESTAMPTZ DEFAULT now(),
    updated_at TIMESTAMPTZ DEFAULT now()
);

-- Vector search result cache (Performance optimization)
CREATE TABLE search_cache (
    query_hash TEXT PRIMARY KEY,
    query_text TEXT NOT NULL,
    results_json TEXT NOT NULL,
    hit_count INT DEFAULT 1,
    created_at TIMESTAMPTZ DEFAULT now(),
    expires_at TIMESTAMPTZ DEFAULT now() + INTERVAL '24 hours'
);

-- Course prerequisite graph
CREATE TABLE prerequisites_graph (
    course_id TEXT,
    prerequisite_id TEXT,
    PRIMARY KEY (course_id, prerequisite_id)
);
```

---

## Installation & Setup

### Prerequisites

- Python 3.11+
- [CockroachDB Cloud](https://cockroachlabs.cloud/) account (free tier works)
- AWS account with S3 bucket
- Google AI Studio API key (Gemini)
- [Ollama](https://ollama.ai/) (optional, for local LLM fallback)
- Mistral API key (optional, for cloud fallback)

### 1. Clone the Repository

```bash
git clone https://github.com/CJP-Hackathon/Hackathon.git
cd Hackathon
```

### 2. Create Virtual Environment

```bash
python3 -m venv venv
source venv/bin/activate   # macOS/Linux
# venv\Scripts\activate    # Windows
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure Environment Variables

```bash
cp .env.example .env
# Edit .env with your actual credentials
```

```env
# Required
GEMINI_API_KEY=your_gemini_api_key
COCKROACHDB_URL=postgresql://user:password@host:26257/defaultdb?sslmode=verify-full

# AWS S3 (required for report storage)
AWS_ACCESS_KEY_ID=your_aws_access_key_id
AWS_SECRET_ACCESS_KEY=your_aws_secret_key
AWS_REGION=us-east-1
AWS_S3_BUCKET_NAME=your-s3-bucket-name

# Optional fallbacks
MISTRAL_API_KEY=your_mistral_api_key
```

### 5. Set Up CockroachDB Schema & Seed Data

```bash
# Run the database seeder (loads 8,983 courses with embeddings)
python seed.py
```

> ⚠️ The seeder generates pgvector embeddings for all courses using `all-MiniLM-L6-v2`. This takes ~10-20 minutes on first run.

### 6. (Optional) Install Ollama for Local LLM Fallback

```bash
# macOS
brew install ollama

# Pull recommended models
ollama pull mistral:instruct   # 4.1GB — primary local model
ollama pull llama3.1:8b        # 4.9GB — secondary local model

# Start Ollama service
ollama serve
```

---

## Running Locally

```bash
# Activate virtual environment
source venv/bin/activate

# Start the Flask server
python app.py
```

Visit **[http://127.0.0.1:8080](http://127.0.0.1:8080)** in your browser.

---

## Student Profiles & Major Permissions

Three demo student accounts are pre-configured with distinct major disciplines:

| Student | Major | Student ID | Allowed Departments |
|:---|:---|:---|:---|
| **Alice Smith** | Computer Science | `student_123` | CMPSC, CMPEN, SWENG, DS, EE, ME, MATH, STAT, AERSP, IE, PHYS, EMET, EET |
| **Bob Johnson** | Architecture | `student_456` | ARCH, LARCH, ART, ARTH, ARTSA, DART, INART, EDSGN, AE, CIVE |
| **Carol Williams** | Law & Legal Studies | `student_789` | BLAW, CRIM, CRIMJ, PLSC, PHIL, SOC, PSYCH, HIST, ECON |

### Permission Rules

- **All students** can browse the **full 8,983-course catalog** and all 242 departments
- **Enrollment** is restricted to a student's major discipline departments
- Courses outside the student's discipline display a `🔒 Major Restricted` badge
- **Completed past-term courses** (Fall 2025) display `✓ Completed` and cannot be dropped — they are permanent academic record
- **Current-term enrollments** (Spring 2026) can be freely dropped

---

## API Reference

### `POST /api/chat`
Send a message to the AI academic advisor.

**Request:**
```json
{
  "student_id": "student_123",
  "message": "What AI and machine learning courses should I take?",
  "session_id": "optional-uuid"
}
```

**Response:**
```json
{
  "session_id": "uuid",
  "recommendation": {
    "agent_answer": "Based on your CS background, here are top AI courses...",
    "recommendations": [
      {
        "course_code": "CMPSC 442",
        "title": "Artificial Intelligence",
        "credits": 3,
        "fit": 96,
        "avg_grade": "B+",
        "explanation": "Core AI course covering search, logic, and ML fundamentals",
        "benefits": "Essential for any AI/ML career path",
        "tags": ["career match 0.96", "prereqs met", "high demand"]
      }
    ]
  },
  "s3_path": "s3://bucket/reports/student_student_123/session_uuid.json"
}
```

### `POST /api/enroll`
Enroll a student in a course (major discipline permissions enforced).

```json
{ "student_id": "student_123", "course_id": "CMPSC 442", "semester": "Spring 2026" }
```

### `POST /api/drop`
Drop a student from a course (past-term courses cannot be dropped).

```json
{ "student_id": "student_123", "course_id": "CMPSC 442" }
```

### `GET /api/courses?dept=CMPSC`
Returns all courses, optionally filtered by department prefix.

### `GET /api/departments`
Returns all 242 departments in the catalog.

### `GET /api/student/<student_id>`
Returns student profile including major, allowed departments, and academic history.

---

## Performance Optimizations

| Optimization | Before | After | Technique |
|:---|:---|:---|:---|
| LLM Roundtrips | 3 sequential API calls | 1 API call | Single-shot architecture with pre-fetched tools |
| Cold Start | ~6s model load delay | ~0s for users | `warm_model()` fires in background thread at server startup |
| Repeated queries | ~1100ms pgvector scan | ~2ms | CockroachDB `search_cache` table (24h TTL) + LRU in-process cache |
| S3 uploads blocking response | +1-2s per request | 0s | Background `threading.Thread(daemon=True)` after response returned |
| DB writes blocking response | Serial | Parallel | `ThreadPoolExecutor` for session init + course fetch simultaneously |

---

## Project Structure

```
hackathon/
├── app.py              # Flask application — all API routes
├── agent.py            # AI agent with 4-tier LLM fallback chain
├── tools.py            # CockroachDB tools: MCP, vector search, ccloud health
├── db.py               # CockroachDB connection and query functions
├── s3_utils.py         # AWS S3 report upload utilities
├── seed.py             # Database seeder (courses, embeddings, departments)
├── requirements.txt    # Python dependencies
├── .env.example        # Environment variable template
├── ARCHITECTURE-3.pdf  # System architecture diagram
├── static/
│   └── style.css       # Application styles — parchment academic theme
└── templates/
    ├── index.html       # Login page with major selection
    └── advisor.html     # Main advisor portal (catalog + AI chat)
```

---

## License

This project is licensed under the MIT License — see the [LICENSE](./LICENSE) file for details.

---

## Acknowledgements

- **CockroachDB** for providing Cloud Serverless infrastructure and pgvector support
- **Google DeepMind** for Gemini 3.5 Flash-Lite API
- **Sentence Transformers** (`all-MiniLM-L6-v2`) for local embedding generation
- **Ollama** for local LLM inference (`mistral:instruct`, `llama3.1:8b`)
- **Mistral AI** for cloud fallback API
- **CockroachDB State University** course catalog data
