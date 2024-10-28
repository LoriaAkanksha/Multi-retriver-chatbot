import streamlit as st
import pandas as pd
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Cassandra
from pandasai import SmartDataframe
from langchain_groq import ChatGroq
from dotenv import load_dotenv
import cassio
import os
import warnings
import PIL.Image
import requests
import google.generativeai as genai
from chromadb.utils.embedding_functions import OpenCLIPEmbeddingFunction
from chromadb.utils.data_loaders import ImageLoader
import chromadb

warnings.filterwarnings("ignore")

# Load environment variables
load_dotenv()
ASTRA_DB_APPLICATION_TOKEN = os.getenv('ASTRA_DB_APPLICATION_TOKEN')
ASTRA_DB_ID = os.getenv('ASTRA_DB_ID')
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
api_key = os.getenv("api_key")

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

# Configure Generative AI and ChromaDB for image retrieval
genai.configure(api_key=api_key)
chroma_client = chromadb.PersistentClient(path="/home/akanksha/Desktop/Chatbot/Vector_database")
image_loader = ImageLoader()
CLIP = OpenCLIPEmbeddingFunction()
image_vdb = chroma_client.get_or_create_collection(name="image", embedding_function=CLIP, data_loader=image_loader)
model = genai.GenerativeModel(model_name="gemini-1.5-pro")

# Static CSV loading for CSV-based retrieval
csv_file_path = "pokemon.csv"  # Update with your static file path
data = pd.read_csv(csv_file_path)

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

# Function to format and retrieve images based on query
def format_image_inputs(data):
    return [data['uris'][0][0], data['uris'][0][1]]

def query_db(query, results=3):
    return image_vdb.query(
        query_texts=[query],
        n_results=results,
        include=['uris', 'distances']
    )

def retrieve_images(query):
    prompt = f"Based on the following description: '{query}', retrieve relevant images that match this description."
    response = model.generate_content([prompt])
    response_text = response.text
    image_identifiers = response_text.splitlines()

    images = []
    for identifier in image_identifiers[:3]:  # Limit to 3 images
        results = query_db(identifier.strip())
        image_paths = format_image_inputs(results)
        for path in image_paths:
            if len(images) >= 3:  # Stop once 3 images are retrieved
                break
            images.append(path)
    return images

# Set up Streamlit UI
st.set_page_config(layout="wide")
st.title("Multi-Retriever Chatbot")

# Choose retrieval method
retrieval_option = st.radio("Select Retrieval Method", ("PDF-Based", "CSV/Excel-Based", "Image-Based"))

# Input query
user_query = st.text_input("Enter your query:")

# Function to retrieve results based on selected option
def retrieve_from_astra(query, retriever):
    if retriever == "PDF-Based":
        pdf_retriever = pdf_vector_store.as_retriever()
        response = pdf_retriever.invoke(query)
    elif retriever == "CSV/Excel-Based":
        response = chat_with_csv(data, query)
    elif retriever == "Image-Based":
        images = retrieve_images(query)
        return images  # Return images list instead of text response
    else:
        response = "Invalid retriever selected."
    return response

# Perform retrieval and display results
if st.button("Get Answer"):
    if user_query:
        response = retrieve_from_astra(user_query, retrieval_option)
        if retrieval_option == "Image-Based":
            st.write("Retrieved Images:")
            for image_path in response:
                image = PIL.Image.open(image_path)
                st.image(image, use_column_width=True)
        else:
            st.write("Response:")
            st.write(response)
    else:
        st.write("Please enter a query.")
