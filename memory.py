from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from dotenv import load_dotenv

load_dotenv()

# 1. Initialize the Gemini model
llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0)

# 2. Setup the prompt layout with a placeholder for the chat history
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant."),
    MessagesPlaceholder(variable_name="history"),
    ("human", "{input}")
])

# 3. Create the standard execution chain
chain = prompt | llm

# 4. Create an in-memory dictionary to hold the conversation history
store = {}

def get_session_history(session_id: str) -> InMemoryChatMessageHistory:
    if session_id not in store:
        store[session_id] = InMemoryChatMessageHistory()
    return store[session_id]

# 5. Wrap the chain to automatically manage and inject memory history
conversation_with_memory = RunnableWithMessageHistory(
    chain,
    get_session_history,
    input_messages_key="input",
    history_messages_key="history",
)

# --- Test the conversation memory ---

# First interaction
config = {"configurable": {"session_id": "user_session_1"}}
response1 = conversation_with_memory.invoke({"input": "Hi, my name is Sakshi"}, config=config)
print(f"AI: {response1.content}\n")

# Second interaction (It will successfully remember your name)
response2 = conversation_with_memory.invoke({"input": "What is my name?"}, config=config)
print(f"AI: {response2.content}")
