# 🧠 NLP to SQL System (Agent-Based MCP Architecture)

A production-style **Natural Language to SQL (NL → SQL)** system that allows users to query a **PostgreSQL** database using plain English.

The system leverages **LLM agents**, a **multi-server MCP (Model Context Protocol)** architecture, and **safe SQL execution** with strict read-only defaults.

*Developed during an internship to explore agentic AI systems, backend orchestration, distributed tool execution, and database safety.*

---

## 🚀 Features

- 🔤 **Natural Language → SQL** generation using LLMs
- 🤖 **Agent-based** tool selection and execution
- 🧩 **Multi-server MCP** architecture with discovery & failover
- 🛡️ **Safe SQL execution**
  - Read-only mode enabled by **default**
  - Controlled write access (INSERT / UPDATE / DELETE)
- 🗄️ **PostgreSQL** backend
- 🧭 **Schema discovery** & browsing
- 🌐 **Interactive Streamlit UI**

---

## 🏗️ Architecture Overview



Streamlit UI
|
v
Agent Server (FastAPI + LLM)
|
v
Tool Gateway (REST → MCP)
|
v
MCP Servers (PostgreSQL over SSE)


---

## 🧩 Core Components

| Component | Description |
|-----------|-------------|
| **Streamlit UI** (`app.py`) | User interface for asking natural language questions. Displays generated SQL and execution results. |
| **Agent Server** (`agent_server.py`) | Uses an LLM to decide which tool to invoke. Routes requests via the Tool Gateway. |
| **Tool Gateway** (`tool_gateway.py`) | REST interface that forwards requests to MCP servers. |
| **MCP Server** (`mcp_server.py`) | Executes SQL safely against PostgreSQL. Exposes tools (`run_query`, `get_tables`) and resources (`postgres://schema/tables`). |
| **Router Client** (`client_utils.py`) | Handles multi-server discovery, routing, and failover. |

---

## 🛠️ Tech Stack



Python 3.11+  -   Streamlit  -   FastAPI  -   FastMCP
PostgreSQL    -   OpenRouter/Gemini  -   LangChain  -   Pydantic



---

## 📦 Installation & Setup

### 1️⃣ Clone the repository
```bash
git clone https://github.com/Ezhil64/nlp-to-sql-mcp.git
cd nlp-to-sql-mcp
```


### 2️⃣ Create virtual environment

```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS/Linux
source .venv/bin/activate
```


### 3️⃣ Install dependencies

```bash
pip install -r requirements.txt
```


### 4️⃣ Configure environment variables

Create `.env` file (don't commit it):

```env
GOOGLE_API_KEY=your_google_api_key
OPENROUTER_API_KEY=your_openrouter_api_key

DB_HOST=localhost
DB_PORT=5432
DB_NAME=postgres
DB_USER=postgres
DB_PASSWORD=your_password

BACKEND_URLS=http://127.0.0.1:3101/sse,http://127.0.0.1:3102/sse
AGENT_SERVER_URL=http://127.0.0.1:8002
```


---

## ▶️ Running the System

### 1️⃣ Start MCP Server(s)

**Terminal 1:**

```bash
python mcp_server.py
```

**Optional second server (Terminal 2):**

```bash
# Windows
$env:MCP_PORT=3102
# macOS/Linux
export MCP_PORT=3102

python mcp_server.py
```


### 2️⃣ Start Tool Gateway

**Terminal 3:**

```bash
uvicorn tool_gateway:app --port 8001
```


### 3️⃣ Start Agent Server

**Terminal 4:**

```bash
uvicorn agent_server:app --port 8002
```


### 4️⃣ Start Streamlit UI

**Terminal 5:**

```bash
streamlit run app.py
```

**Open:** http://localhost:8501

## 🖼️ Screenshots

<img width="1919" height="971" alt="Streamlit Home" src="https://github.com/user-attachments/assets/d15baf6d-8e10-43c8-be0c-047125a74909" />

*Streamlit UI showing the main interface for natural language to SQL queries.*

---

<img width="1637" height="933" alt="Generated SQL" src="https://github.com/user-attachments/assets/0fa9ce5d-454d-4a37-b6db-1c1913b79edd" />

*LLM-generated SQL query produced from a natural language question.*

---

<img width="1628" height="921" alt="Query Results" src="https://github.com/user-attachments/assets/b8ffc767-3362-4480-8925-31216fde6336" />

*Query execution results displayed in tabular format via the Streamlit UI.*

---

## 🧪 Example Queries

```
"Show employees with salary greater than 80000"
"List all tables in the database" 
"Insert a new employee with name Alice, salary 90000, age 30"
```


---

## 🔐 Security Notes

- ✅ All API keys managed via **environment variables**
- ✅ `.env` excluded via **.gitignore**
- ✅ **SQL validation** prevents dangerous operations
- ✅ **Read-only execution** enforced by default

---

## 🎯 Learning Outcomes

- Built an **agent-based AI system**
- Designed **distributed MCP architecture**
- Implemented **safe database tool execution**
- Integrated **LLMs into backend workflows**
- Practiced **production-grade Git \& security hygiene**

---

## 👤 Author

**Ezhil Aadhithyan**
[GitHub](https://github.com/Ezhil64)

⭐ **Star the repo if you find it useful!**


