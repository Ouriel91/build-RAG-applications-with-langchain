import os
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_huggingface import HuggingFaceEmbeddings

from langchain_text_splitters  import CharacterTextSplitter
from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores import Chroma

embeds = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

# Configure path to docs and persist directory
docs_path = os.path.join(os.path.dirname(__file__), 'books')
file_path = os.path.join(docs_path, 'harry_potter.txt')
persist_directory = os.path.join(os.path.dirname(__file__), 'chroma_db')

# Check if Chroma DB already exists
if not os.path.exists(persist_directory):
    print("Creating Chroma DB...")

    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Document file not found: {file_path}")
    
    # Create text loader to load document
    loader = TextLoader(file_path, encoding='utf-8')
    documents = loader.load()

    # Split text into chunks
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    chunks = text_splitter.split_documents(documents)
    print(f"Number of chunks created: {len(chunks)}")
    # add metadata to each chunk
    for chunk in chunks:
        chunk.metadata = {"source": "harry potter"}

    print("Chunk sample:", chunks[42].page_content)

    # Create Chroma vector store
    print("Creating Chroma vector store...")
    db = Chroma.from_documents(
        documents=chunks,
        embedding=embeds,
        persist_directory=persist_directory
    )
    print("Fininished creating Chroma vector store.")
else:
    print("Chroma DB already exists. Skipping creation.")