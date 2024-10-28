import streamlit as st
import pandas as pd
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Cassandra
from pandasai import SmartDataframe
from langchain_groq import ChatGroq
from dotenv import load_dotenv
import cassio
import os

# Load environment variables
load_dotenv()
ASTRA_DB_APPLICATION_TOKEN = os.getenv('ASTRA_DB_APPLICATION_TOKEN')
ASTRA_DB_ID = os.getenv('ASTRA_DB_ID')
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# Initialize the AstraDB session
session = cassio.init(
    token=ASTRA_DB_APPLICATION_TOKEN, 
    database_id=ASTRA_DB_ID
)

# Initialize embeddings and vector store for PDF search
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

pdf_vector_store = Cassandra(
    embedding=embeddings,
    table_name="qa_mini_demo",
    session=None,
    keyspace=None
)

# Function to chat with CSV data using GROQ
def chat_with_csv(df, query):
    llm = ChatGroq(
        groq_api_key=GROQ_API_KEY, 
        model_name="llama3-70b-8192",
        temperature=0.2
    )
    pandas_ai = SmartDataframe(df, config={"llm": llm})
    result = pandas_ai.chat(query)
    return result

# Set up Streamlit UI
st.set_page_config(layout="wide")
st.title("Multi-Retriever Chatbot")

# Static CSV loading for CSV-based retrieval
csv_file_path = "pokemon.csv"  # Update with your static file path
data = pd.read_csv(csv_file_path)

# Choose retrieval method
retrieval_option = st.radio("Select Retrieval Method", ("PDF-Based", "CSV/Excel-Based"))

# Input query
user_query = st.text_input("Enter your query:")



# Function to retrieve results based on selected option
def retrieve_from_astra(query, retriever):
    if retriever == "PDF-Based":
        pdf_retriever = pdf_vector_store.as_retriever()
        response = pdf_retriever.invoke(query)
    elif retriever == "CSV/Excel-Based":
        response = chat_with_csv(data, query)
    else:
        response = "Invalid retriever selected."
    return response

# Perform retrieval and display results
if st.button("Get Answer"):
    if user_query:
        response = retrieve_from_astra(user_query, retrieval_option)
        st.write("Response:")
        st.write(response)
    else:
        st.write("Please enter a query.")
