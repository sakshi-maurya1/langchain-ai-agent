from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Initialize the Gemini model (it automatically detects GOOGLE_API_KEY from your .env)
llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0)

# Define your prompt template
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant."),
    ("human", "{input}")
])
    
# Create the chain using LangChain Expression Language (LCEL)
chain = prompt | llm

# Invoke the chain
response = chain.invoke({"input": "What is RAG in AI?"})

# Print the final text content
print(response.content)
