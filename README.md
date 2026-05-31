# 🤖 AI Agent with Tools & Memory — Built with LangChain + Gemini

A conversational AI agent built using LangChain, Google Gemini, and Streamlit. This project was my hands-on exploration of how LLM-powered agents work — starting from a basic chain all the way to a full agent with tools, memory, and a chat UI.

---

## 💡 What I Built & Why

I wanted to understand the difference between just calling an LLM and actually building an **agent** — something that can reason, decide which tool to use, and remember the conversation. So I broke the learning into three stages and then combined everything into a Streamlit web app.

---

## 📁 Project Structure

```
langchain-ai-agent/
│
├── chain.py          # Stage 1: Basic LLM chain, single question-answer
├── memory.py         # Stage 2: Chain with conversation memory
├── agent.py          # Stage 3: ReAct Agent with tools and memory
├── agent_app.py      # Stage 4: Full Streamlit chat UI for the agent
├── .env              # Your API key (never pushed to GitHub)
├── .gitignore        # Ignores .env
└── README.md
```

---

## 🧠 How I Approached This — Stage by Stage

### Stage 1 — Simple Chain (`chain.py`)
The starting point. I connected a prompt template to the Gemini LLM using LangChain's pipe (`|`) operator — this is called LCEL (LangChain Expression Language). One input goes in, one response comes out. No memory, no tools. Just understanding how a basic chain works.

```python
chain = prompt | llm
response = chain.invoke({"input": "What is RAG in AI?"})
```

### Stage 2 — Adding Memory (`memory.py`)
A single-turn response is not enough for a real assistant. In this stage I added conversation memory using `RunnableWithMessageHistory`. The prompt now has a `MessagesPlaceholder` — a slot where the full conversation history gets injected before every call to the LLM. This way the model remembers what was said earlier in the conversation.

Each conversation is stored by a `session_id`, which means multiple users could have separate memory sessions.

```python
response1 = conversation_with_memory.invoke({"input": "Hi, my name is Sakshi"}, config=config)
response2 = conversation_with_memory.invoke({"input": "What is my name?"}, config=config)
# The model correctly remembers the name from the first message
```

### Stage 3 — Building the Agent (`agent.py`)
This is where it gets interesting. An agent is different from a chain — instead of just generating text, it can **decide to use tools** based on what the user asks. I used the **ReAct pattern** (Reason + Act), where the agent runs a loop:

```
Thought → Action (call a tool) → Observation (see result) → Thought → ... → Final Answer
```

I built two tools using the `@tool` decorator:
- **Calculator** — evaluates any math expression safely
- **Knowledge Base** — answers questions about AI concepts like RAG, LangChain, and vector databases

Memory is maintained as a running list of `HumanMessage` and `AIMessage` objects passed on every invocation.

### Stage 4 — Streamlit Chat UI (`agent_app.py`)
Finally I wrapped the agent in a proper web interface using Streamlit. Key things I handled:
- `st.session_state` — persists chat history and display messages across Streamlit reruns
- `@st.cache_resource` — loads the LLM and agent only once, not on every user message
- `st.chat_input` + `st.chat_message` — gives it a clean ChatGPT-style interface
- Sidebar with tool descriptions and example queries
- Clear Chat button to reset the conversation

---

## 🛠️ Tech Stack

| Tool | Purpose |
|------|---------|
| Python 3.12 | Core language |
| LangChain | Agent, chain, and tool framework |
| LangGraph | ReAct agent execution |
| Google Gemini 2.5 Flash | LLM backend |
| Streamlit | Web UI |
| python-dotenv | API key management |

---

## ⚙️ Setup & Run

**1. Clone the repo**
```bash
git clone https://github.com/sakshi-maurya1/langchain-ai-agent
cd langchain-ai-agent
```

**2. Install dependencies**
```bash
pip install langchain langchain-community langchain-core langchain-google-genai langgraph streamlit python-dotenv
```

**3. Add your API key**

Create a `.env` file in the root folder:
```
GOOGLE_API_KEY=your_google_api_key_here
```
Get your free key at: [aistudio.google.com](https://aistudio.google.com)

**4. Run the scripts individually**
```bash
python chain.py       # Test basic chain
python memory.py      # Test memory
python agent.py       # Test agent in terminal
```

**5. Run the Streamlit app**
```bash
streamlit run agent_app.py
```

---

## 🔑 Key Concepts I Learned

**What is an AI Agent?**
An agent is an LLM that doesn't just respond — it decides what action to take. Given a goal, it reasons about which tool to use, calls it, observes the result, and continues until it reaches an answer.

**ReAct Pattern**
The agent loop: Thought → Action → Observation → repeat. This is what makes agents autonomous rather than just reactive.

**Why Memory Matters**
Without memory, every message is treated as a fresh conversation. With memory, the agent maintains context — making it actually useful for multi-turn tasks.

**Chain vs Agent**
A chain follows a fixed sequence of steps. An agent dynamically decides its own steps based on the input. Chains are faster and more predictable; agents are more flexible and powerful.

---

## 🚀 What I'd Add Next

- RAG pipeline — let the agent answer questions from uploaded documents
- More tools — weather API, web search, database queries
- Deploy on Streamlit Cloud or Hugging Face Spaces
- Persistent memory using a vector database instead of in-memory storage
