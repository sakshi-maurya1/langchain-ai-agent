import streamlit as st
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv

# 1. Load environment variables (.env file)
load_dotenv()

# 2. Configure Streamlit Page Layout
st.set_page_config(page_title="Gemini AI Assistant", page_icon="🤖")
st.title("🤖 My Gemini AI Assistant")
st.write("Type your question below and get an instant answer from Gemini!")

# 3. Setup the LangChain components
llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0)
prompt_template = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant."),
    ("human", "{input}")
])
chain = prompt_template | llm

# 4. Create the Frontend UI Form
with st.form("my_form"):
    user_input = st.text_input("Your Question:", placeholder="What is RAG in AI?")
    submit_button = st.form_submit_button("Ask Gemini")

# 5. Handle the form submission
if submit_button and user_input:
    with st.spinner("Thinking..."):
        try:
            # Run your LangChain chain
            response = chain.invoke({"input": user_input})
            
            # Display the result in a clean box
            st.subheader("Answer:")
            st.write(response.content)
            
        except Exception as e:
            st.error(f"An error occurred: {e}")
