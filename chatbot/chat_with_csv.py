import pandas as pd
from cassandra.cluster import Cluster
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Cassandra
import cassio
import os
from dotenv import load_dotenv

load_dotenv()

# Load API keys and initialize AstraDB session
ASTRA_DB_APPLICATION_TOKEN = os.getenv('ASTRA_DB_APPLICATION_TOKEN')
ASTRA_DB_ID = os.getenv('ASTRA_DB_ID')
cassio.init(token=ASTRA_DB_APPLICATION_TOKEN, database_id=ASTRA_DB_ID)

# Define file path for the CSV file
csv_file_path = "pokemon.csv"

# Load CSV data
csv_data = pd.read_csv(csv_file_path)

# Print the data types of the columns for debugging
print("Data Types in CSV:")
print(csv_data.dtypes)

# Define a document class to hold records with an ID, page_content, and metadata
class Document:
    def __init__(self, id, page_content, metadata=None, **kwargs):
        self.id = id
        self.page_content = page_content  # Text content of the document
        self.metadata = metadata or {}  # Additional metadata
        self.__dict__.update(kwargs)  # Add additional attributes

# Convert data to a list of Document objects with string conversion for all values
csv_records = []
for i, record in enumerate(csv_data.to_dict(orient='records')):
    # Convert each record's values to strings
    record_str = {k: str(v) for k, v in record.items()}
    # Create Document object
    doc = Document(
        id=i,
        page_content=str(record), 
        metadata=record_str
    )
    csv_records.append(doc)

# Print the records to see what is being inserted
for record in csv_records:
    print(f"ID: {record.id}, Content: {record.page_content}, Metadata: {record.metadata}")

# Initialize embeddings (optional for text data) and vector store for AstraDB
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

# Initialize AstraDB vector store for CSV data
astra_vector_store_csv = Cassandra(
    embedding=embeddings,
    table_name="pokemoncsv_data_table",  # New table for CSV data
    session=None,
    keyspace=None
)

# Add CSV records to the AstraDB vector store
try:
    astra_vector_store_csv.add_documents(csv_records)
    print("Inserted %i CSV records." % len(csv_records))
except Exception as e:
    print("Error occurred while inserting records:", e)
