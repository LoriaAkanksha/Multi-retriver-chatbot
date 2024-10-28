import streamlit as st
import PIL.Image
import requests
import google.generativeai as genai
from chromadb.utils.embedding_functions import OpenCLIPEmbeddingFunction
from chromadb.utils.data_loaders import ImageLoader
import chromadb
from dotenv import load_dotenv
import os
import warnings
import re

warnings.filterwarnings("ignore")

load_dotenv()
api_key = os.getenv("api_key")
genai.configure(api_key=api_key)


def format_image_inputs(data):
    return [data['uris'][0][0], data['uris'][0][1]]


def query_db(query, results):
    return image_vdb.query(
        query_texts=[query],
        n_results=results,
        include=['uris', 'distances']
    )


st.title("Image Retrieval Assistant")
st.write("Enter a description to retrieve relevant images.")

# Set up Chromadb client and model
chroma_client = chromadb.PersistentClient(path="/home/akanksha/Desktop/Chatbot/Vector_database")
image_loader = ImageLoader()
CLIP = OpenCLIPEmbeddingFunction()
image_vdb = chroma_client.get_or_create_collection(name="image", embedding_function=CLIP, data_loader=image_loader)
model = genai.GenerativeModel(model_name="gemini-1.5-pro")

query = st.text_input("Enter a detailed description for the image retrieval:")

if query and st.button("Retrieve Images"):
    try:
        prompt = f"""Based on the following description: "{query}", retrieve relevant images that match this description. 
                    Provide a list of image URLs or identifiers."""
        
        response = model.generate_content([prompt])
        response_text = response.text
        print(response_text)

        # Extract image identifiers or URLs from the response
        image_identifiers = response_text.splitlines()

        # Retrieve a maximum of 3 images for each identifier
        for identifier in image_identifiers:
            results = query_db(identifier.strip(), results=3)  # Fetch 3 images from the database based on identifier
            image_paths = format_image_inputs(results)
            for image_path in image_paths:
                if len(image_paths) >= 3:  # Limit to 3 images in total
                    break
                sample_file = PIL.Image.open(image_path)
                st.image(sample_file, caption=identifier.strip(), use_column_width=True)
    except Exception as e:
        st.error(f"An error occurred: {e}")
