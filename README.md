# Project Title

INFIX-LEAD-AGENT

A brief, description of what our AI agent or project does.

---

## 🚀 Key Features
- **Intelligent Routing:** Automatically handles different user intents.
- **Advanced RAG:** Retrieval-Augmented Generation using [Your Vector DB].
- **Multi-Turn Memory:** Maintains context across complex conversations.

## 🛠️ Tech Stack
- **Framework:** LangChain / LangGraph
- **LLM:** Google Gemini 1.5 Flash
- **Vector DB:** ChromaDB
- **UI:** Streamlit

## 📦 Installation

```bash
# Clone the repository
git clone 
```

# make sure ur at the root 
```
## on WINDOWS
# Create the environment
python -m venv venv

# Activate the environment
.\venv\Scripts\activate
```
```
## on MAC
# Create the environment
python3 -m venv venv

# Activate the environment
source venv/bin/activate
```
```
# Install dependencies
pip install -r requirements.txt

```

``` 
# to run the streamlit app
streamlit run app.py
```

```
```
INFLX-LEAD-AGENT/
├── data/
│   ├── manuals/
│   │   └── autostream_info.md
│   └── faqs.csv
├── scripts/
│   ├── ingest.py
│   └── batch_summarize.py
├── vector_db/
│   ├── chroma.sqlite3
│   └── ...
├── .env
├── app.py
├── graph.py
├── leads.json
└── requirements.txt

```

## 🏗️ Architecture Explanation
This project utilizes LangGraph to manage the conversational flow and state persistence. I chose LangGraph because its graph-based architecture is ideal for handling the non-linear transitions between intent identification, RAG-powered knowledge retrieval, and tool execution.

State Management: Memory is retained across 5-6 conversation turns using LangGraph's built-in state management. The state object tracks the current conversation turn, identifies the user's intent shift, and stores lead details (Name, Email, Platform) as they are collected sequentially. This prevents the agent from triggering the lead capture tool prematurely before all required data is gathered.


## 📱 WhatsApp Deployment Strategy
To deploy this agent to WhatsApp, I would use Webhooks to create a real-time communication bridge between the Meta Business API and the agent backend.

Webhook Integration: Set up an API endpoint (using Flask or FastAPI) to receive incoming POST requests from the WhatsApp Business API whenever a user sends a message.

Session Identification: Use the user's phone number as a unique session ID to map incoming messages to their specific LangGraph state, ensuring continuity in memory.

Response Loop: The webhook server passes the user input to the agent, which processes it through the RAG or Lead Capture nodes and returns a response.

Outgoing Message: The backend then sends the agent's response back to the user via the WhatsApp Send Message API.
