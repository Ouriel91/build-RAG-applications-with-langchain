import os
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

embeds = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

# Configure path to docs and persist directory
docs_path = os.path.join(os.path.dirname(__file__), 'books')
persist_directory = os.path.join(os.path.dirname(__file__), 'chroma_db')

# Setup Chroma vector store
db = Chroma(
    embedding_function=embeds,
    persist_directory=persist_directory
)

retriver = db.as_retriever(
    search_type="similarity_score_threshold", 
    search_kwargs={"k": 3, "score_threshold": 0.002}
)
query = "Who is an evil character?"

chunks = retriver.invoke(query)
for i, chunk in enumerate(chunks):
    print(f"--- Chunk {i+1} ---")
    print(chunk.metadata['source'], ":", chunk.page_content)