import os
from dotenv import load_dotenv
import streamlit as st
from langchain.prompts import PromptTemplate
from langchain_groq import ChatGroq
from langchain.chains.summarize import load_summarize_chain
from langchain.docstore.document import Document

load_dotenv() 

# App Config
st.set_page_config(page_title="My Text Summarization App using Langchain", page_icon="🦜")
st.title("🦜 My Text Summarization App using Langchain") 

# Input and validate Groq API Key
if "api_key_valid" not in st.session_state:
    st.session_state.api_key_valid = False

if not st.session_state.api_key_valid:
    # Input field for Groq API Key
    groq_api_key = st.text_input("Enter your Groq API Key", value="", type="password")
    
    if st.button("Validate API Key"):
        if groq_api_key.strip():
            try:
                # Attempt to initialize the ChatGroq object with the provided API key
                llm = ChatGroq(groq_api_key=groq_api_key, model="gemma2-9b-it")
                st.session_state.groq_api_key = groq_api_key
                st.session_state.api_key_valid = True
                st.success("API Key validated successfully!")
            except Exception as e:
                st.error(f"Invalid API Key: {e}")
        else:
            st.error("Please provide a valid API Key.")

# Show input area and summarize button after API Key is validated
if st.session_state.api_key_valid:
    # Prompt Template
    prompt_template = """
    Provide a summary of the following content in 300 words:
    Content: {text}
    """
    prompt = PromptTemplate(template=prompt_template, input_variables=["text"])

    # Text input area for summarization
    user_input = st.text_area("Enter text or sentences to summarize:", height=200)

    # Summarize button
    if st.button("Summarize"):
        if not user_input.strip():
            st.error("Please enter text to summarize.")
        else:
            try:
                with st.spinner("Summarizing..."):
                    # Instantiate LLM using the validated API key
                    llm = ChatGroq(groq_api_key=st.session_state.groq_api_key, model="gemma2-9b-it")
                    
                    # Chain for summarization
                    chain = load_summarize_chain(llm, chain_type="stuff", prompt=prompt)
                    
                    # Prepare input in the correct format
                    input_docs = [Document(page_content=user_input)]
                    
                    # Run the summarization
                    output_summary = chain.run(input_docs)
                    
                    st.success("Summary:")
                    st.write(output_summary)
            except Exception as e:
                st.error(f"Exception: {e}")
