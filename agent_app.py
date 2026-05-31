import streamlit as st
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage, AIMessage
from langgraph.prebuilt import create_react_agent
from dotenv import load_dotenv
import math

load_dotenv()

# --- Page Config ---
st.set_page_config(page_title="AI Agent Assistant", page_icon="🤖")
st.title("🤖 AI Agent with Tools & Memory")
st.write("This agent can do math, answer AI questions, and remembers your conversation!")

# --- Tools ---
@tool
def calculator(expression: str) -> str:
    """Useful for mathematical calculations. Input should be a math expression like '25 * 4' or 'sqrt(144)'."""
    try:
        result = eval(expression, {"__builtins__": {}}, {"math": math, "sqrt": math.sqrt})
        return str(result)
    except Exception as e:
        return f"Error: {e}"

@tool
def knowledge_base(query: str) -> str:
    """Useful for answering questions about AI concepts like RAG, LangChain, vector databases, and AI agents."""
    kb = {
        "rag": "RAG stands for Retrieval Augmented Generation. It fetches relevant documents and passes them to an LLM to generate accurate answers.",
        "langchain": "LangChain is a framework for building LLM-powered applications including agents, chains, and RAG pipelines.",
        "vector database": "A vector database stores embeddings and enables semantic similarity search. Examples: FAISS, Pinecone, ChromaDB, Weaviate.",
        "ai agent": "An AI agent uses an LLM to decide which tools to use and in what order to complete a goal autonomously.",
    }
    query_lower = query.lower()
    for key, value in kb.items():
        if key in query_lower:
            return value
    return "I don't have specific information on that topic."

# --- Agent (cached so it doesn't reload every time) ---
@st.cache_resource
def load_agent():
    llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0)
    return create_react_agent(llm, [calculator, knowledge_base])

agent = load_agent()

# --- Session Memory ---
# This keeps chat history across interactions in the same browser session
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "messages_display" not in st.session_state:
    st.session_state.messages_display = []

# --- Display past messages ---
for msg in st.session_state.messages_display:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])

# --- Chat Input ---
user_input = st.chat_input("Ask me anything — math, AI concepts, or just chat!")

if user_input:
    # Show user message
    with st.chat_message("user"):
        st.write(user_input)
    st.session_state.messages_display.append({"role": "user", "content": user_input})

    # Run agent with full memory
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                st.session_state.chat_history.append(HumanMessage(content=user_input))
                
                result = agent.invoke({"messages": st.session_state.chat_history})
                
                ai_message = result["messages"][-1]
                st.session_state.chat_history.append(ai_message)
                
                st.write(ai_message.content)
                st.session_state.messages_display.append({
                    "role": "assistant",
                    "content": ai_message.content
                })

            except Exception as e:
                st.error(f"An error occurred: {e}")

# --- Sidebar info ---
with st.sidebar:
    st.subheader("🛠️ Available Tools")
    st.write("**Calculator** — does any math expression")
    st.write("**Knowledge Base** — answers questions on RAG, LangChain, vector DBs, AI agents")
    st.divider()
    st.subheader("💬 Try asking:")
    st.code("What is 15 multiplied by 37?")
    st.code("What is a vector database?")
    st.code("What is RAG and how does it differ from fine-tuning?")
    st.divider()
    if st.button("🗑️ Clear Chat"):
        st.session_state.chat_history = []
        st.session_state.messages_display = []
        st.rerun()