from langchain.schema import Document  # Import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from cassandra.cluster import Cluster
from langchain_community.vectorstores import Cassandra
from langchain.indexes.vectorstore import VectorStoreIndexWrapper
import cassio
import os
import fitz  # PyMuPDF
from dotenv import load_dotenv
load_dotenv()

# Load API keys
groq_api_key = os.getenv('GROQ_API_KEY')
ASTRA_DB_APPLICATION_TOKEN = os.getenv('ASTRA_DB_APPLICATION_TOKEN')
ASTRA_DB_ID = os.getenv('ASTRA_pdf_DB_ID')
cassio.init(token=ASTRA_DB_APPLICATION_TOKEN, database_id=ASTRA_DB_ID)

# Define file paths for PDF documents
pdf_paths = [
    "Ai&ml.pdf",
    "ml.pdf",
]

# Load text from PDF documents
def load_pdf_text(file_path):
    text = ""
    with fitz.open(file_path) as doc:
        for page in doc:
            text += page.get_text()
    return text

# Convert PDFs to documents
docs_list = [Document(page_content=load_pdf_text(path)) for path in pdf_paths]  # Create Document objects

# Split documents into chunks
text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
    chunk_size=500, chunk_overlap=0
)
doc_splits = text_splitter.split_documents(docs_list)

# Initialize embeddings and vector store
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

astra_vector_store = Cassandra(
    embedding=embeddings,
    table_name="qa_mini_demo",
    session=None,
    keyspace=None
)

# Add documents to the vector store
astra_vector_store.add_documents(doc_splits)
print("Inserted %i document chunks." % len(doc_splits))

# Create vector store index and retriever
astra_vector_index = VectorStoreIndexWrapper(vectorstore=astra_vector_store)
retriever = astra_vector_store.as_retriever()

# Example retrieval query
rag_result = retriever.invoke("What is machine learning?", ConsistencyLevel="LOCAL_ONE")
print(rag_result)
