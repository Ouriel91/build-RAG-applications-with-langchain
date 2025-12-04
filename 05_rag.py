import os
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_google_genai import ChatGoogleGenerativeAI

llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    temperature=0,
    max_tokens=None,
    timeout=None,
    max_retries=2,
    api_key='..' #complete with yours
)
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

def format_context(chunks):
    ls = []
    for i, chunk in enumerate(chunks):
        ls.append(f"Chunk {i+1}: \n{chunk.page_content}")
    return "\n\n".join(ls)

query = "Who is an evil character?"

chunks = retriver.invoke(query)
for i, chunk in enumerate(chunks):
    print(f"--- Chunk {i+1} ---")
    print(chunk.metadata['source'], ":", chunk.page_content)

query = "Who is an evil character?"
template = """Answer the question:{query} 
based on chunks from the harry potter books:
{context}"""
prompt_template = ChatPromptTemplate.from_template(template)
chunks = retriver.invoke(query)
context = format_context(chunks)
messages = prompt_template.invoke(input={"query": query, "context": context})
print("\n=== Final Prompt ===")
print(messages)
print("\n=== Response ===")
response = llm.invoke(messages)
print(response.content)
