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
    api_key='AIzaSyA6I5TYhyXuyJpUdmt5QdEtXewATN98ikk'
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


template = """Answer the question:{query} 
based on chunks from the harry potter books:
{context}"""
prompt_template = ChatPromptTemplate.from_template(template)

history = [SystemMessage(content="You are a helpful assistant. That answers shortly and clearly.")]

while True:
    user_input = input("You: ")
    if user_input.lower() in ["exit", "quit"]:
        break
    chunks = retriver.invoke(user_input)
    context = format_context(chunks)
    pv = prompt_template.invoke(input={"query": user_input, "context": context})
    llm_response = llm.invoke(history + pv.to_messages())
    print("Bot:", llm_response.content)
    history.append(HumanMessage(content=user_input))
    history.append(AIMessage(content=llm_response.content))