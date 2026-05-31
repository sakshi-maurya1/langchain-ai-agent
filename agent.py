from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage, AIMessage
from langchain.agents import create_react_agent
from dotenv import load_dotenv
import math

load_dotenv()

llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0)

# --- TOOL 1: Calculator ---
@tool
def calculator(expression: str) -> str:
    """Useful for mathematical calculations. Input should be a math expression like '25 * 4' or 'sqrt(144)'."""
    try:
        result = eval(expression, {"__builtins__": {}}, {"math": math, "sqrt": math.sqrt})
        return str(result)
    except Exception as e:
        return f"Error: {e}"

# --- TOOL 2: Simple Knowledge Base ---
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

# --- Agent Setup (LangGraph style) ---
tools = [calculator, knowledge_base]

agent = create_react_agent(llm, tools)

# --- Memory: just keep a message list ---
chat_history = []

def chat(user_input: str):
    chat_history.append(HumanMessage(content=user_input))
    result = agent.invoke({"messages": chat_history})
    ai_message = result["messages"][-1]
    chat_history.append(ai_message)
    return ai_message.content

# --- Run it ---
print("\n--- Query 1: Math ---")
print(chat("What is 15 multiplied by 37, then add 120?"))

print("\n--- Query 2: Knowledge ---")
print(chat("What is a vector database?"))

print("\n--- Query 3: Memory ---")
print(chat("Can you summarise what we discussed so far?"))