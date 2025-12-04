from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage

llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    temperature=0,
    max_tokens=None,
    timeout=None,
    max_retries=2,
    api_key='....' #complate yours
)

history = [SystemMessage(content="You are a swidish chef. You answer shortly in a friendly and humorous way.")]

while True:
    user_input = input("You: ")
    if user_input.lower() in ["exit", "quit"]:
        break
    history.append(HumanMessage(content=user_input))
    response = llm.invoke(history[:5])
    print("Bot:", response.content)
    history.append(AIMessage(content=response.content))
