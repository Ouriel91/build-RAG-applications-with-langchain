from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

print("=== ChatPromptTemplate Example1 ===")
template1 = "tell me a joke about {topic}."
prompt_template1 = ChatPromptTemplate.from_template(template1)
prompt1 = prompt_template1.invoke(input={"topic": "cats"})
print(prompt1)

print("\n=== ChatPromptTemplate Example2 ===")
template2 = "Tell me {count} jokes about {topic}."
prompt_template2 = ChatPromptTemplate.from_template(template2)
prompt2 = prompt_template2.invoke(input={"topic": "dogs", "count": 3})
print(prompt2)

print("\n=== ChatPromptTemplate with Messages Example ===")
chat_prompt_template = ChatPromptTemplate.from_messages([
    "system", "You are a helpful assistant.",
    "human", "Provide a brief summary of the following text: {text}"
])
prompt3 = chat_prompt_template.invoke(input={"text": "LangChain is a framework for building applications with LLMs."})
print(prompt3)