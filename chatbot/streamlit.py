import streamlit as st
import pandas as pd
import PIL.Image
import requests
import google.generativeai as genai
from chromadb.utils.embedding_functions import OpenCLIPEmbeddingFunction
from chromadb.utils.data_loaders import ImageLoader
import chromadb
from dotenv import load_dotenv
import os
import warnings
import cassio
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Cassandra
from pandasai import SmartDataframe
from langchain_groq import ChatGroq

warnings.filterwarnings("ignore")

# Load environment variables
load_dotenv()
api_key = os.getenv("api_key")
ASTRA_DB_APPLICATION_TOKEN = os.getenv('ASTRA_DB_APPLICATION_TOKEN')
ASTRA_DB_ID = os.getenv('ASTRA_DB_ID')
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# Initialize generative AI model
genai.configure(api_key=api_key)
model = genai.GenerativeModel(model_name="gemini-1.5-pro")

# Initialize Chromadb client and model for image retrieval
chroma_client = chromadb.PersistentClient(path="/home/akanksha/Desktop/Chatbot/Vector_database")
image_loader = ImageLoader()
CLIP = OpenCLIPEmbeddingFunction()
image_vdb = chroma_client.get_or_create_collection(name="image", embedding_function=CLIP, data_loader=image_loader)

# Initialize AstraDB session
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

# Function to format image inputs
def format_image_inputs(data):
    return [data['uris'][0][0], data['uris'][0][1]]

# Function to query image database
def query_db(query, results):
    return image_vdb.query(
        query_texts=[query],
        n_results=results,
        include=['uris', 'distances']
    )

# Set up Streamlit UI
st.set_page_config(layout="wide")
st.title("Multi-Retriever Chatbot")

# Static CSV loading for CSV-based retrieval
csv_file_path = "pokemon.csv"  # Update with your static file path
data = pd.read_csv(csv_file_path)

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
        prompt = f"""Based on the following description: "{query}", retrieve relevant images that match this description. 
                    Provide a list of image URLs or identifiers."""
        
        response = model.generate_content([prompt])
        response_text = response.text

        # Extract image identifiers or URLs from the response
        image_identifiers = response_text.splitlines()
        image_results = []

        # Retrieve a maximum of 3 images for each identifier
        for identifier in image_identifiers:
            results = query_db(identifier.strip(), results=3)  # Fetch 3 images from the database based on identifier
            image_paths = format_image_inputs(results)
            for image_path in image_paths:
                if len(image_results) >= 3:  # Limit to 3 images in total
                    break
                image_results.append(image_path)

        return image_results

    else:
        return "Invalid retriever selected."

# Perform retrieval and display results
if st.button("Get Answer"):
    if user_query:
        response = retrieve_from_astra(user_query, retrieval_option)
        st.write("Response:")
        if retrieval_option == "Image-Based":
            if isinstance(response, list) and response:
                for image_path in response:
                    sample_file = PIL.Image.open(image_path)
                    st.image(sample_file, caption="Retrieved Image", use_column_width=True)
            else:
                st.write("No images found.")
        else:
            st.write(response)
    else:
        st.write("Please enter a query.")
