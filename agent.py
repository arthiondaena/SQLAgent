from langchain_community.document_loaders import UnstructuredMarkdownLoader
from langchain import hub
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.chat_models import init_chat_model
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.vectorstores import InMemoryVectorStore
from langgraph.prebuilt import create_react_agent
from langgraph.checkpoint.memory import MemorySaver
from langchain.agents import tool
from typing_extensions import List, TypedDict
from langchain_core.documents import Document
from langgraph.graph import START, StateGraph
import os
from utils import get_info_sqlalchemy, extract_code_blocks
from var import db_info, markdown_info, system_prompt_template
from groq import Groq
from langchain_community.utilities import SQLDatabase
from dotenv import load_dotenv
from langchain_groq import ChatGroq

load_dotenv()
uri = os.environ['POSTGRES_URI']

def load_markdown(markdown_path: str = "bingenbash.md"):
    loader = UnstructuredMarkdownLoader(markdown_path)

    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")
    vector_store = InMemoryVectorStore(embeddings)

    docs = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=3000, chunk_overlap=500)
    all_splits = text_splitter.split_documents(docs)

    # Index chunks
    _ = vector_store.add_documents(documents=all_splits)

    return vector_store

def rag(markdown_path: str = "bingenbash.md"):
    llm = init_chat_model("qwen-2.5-32b", model_provider="groq")
    vector_store = load_markdown(markdown_path)

    # Define prompt for question-answering
    prompt = hub.pull("rlm/rag-prompt")

    # Define state for application
    class State(TypedDict):
        question: str
        context: List[Document]
        answer: str


    # Define application steps
    def retrieve(state: State):
        retrieved_docs = vector_store.similarity_search(state["question"])
        return {"context": retrieved_docs}


    def generate(state: State):
        docs_content = "\n\n".join(doc.page_content for doc in state["context"])
        messages = prompt.invoke({"question": state["question"], "context": docs_content})
        response = llm.invoke(messages)
        return {"answer": response.content}

    # Compile application and test
    graph_builder = StateGraph(State).add_sequence([retrieve, generate])
    graph_builder.add_edge(START, "retrieve")
    graph = graph_builder.compile()

    return graph

def sqlChatInfo(uri: str = None):
    if uri is None:
        uri = os.environ['POSTGRES_URI']

    db_info = get_info_sqlalchemy(uri)
    markdown = markdown_info.format(**db_info)
    system_prompt = system_prompt_template.format(markdown_info=markdown)
    return system_prompt

def inference(prompt: str, system_prompt: str, model: str = None, api_key=None) -> str:
    if model is None:
        model = "llama-3.3-70b-versatile"

    if api_key is None:
        api_key = os.environ['GROQ_API_KEY']

    try:
        client = Groq(api_key=api_key)

    except Exception as e:
        print(e)

    chat_completion = client.chat.completions.create(
        model=model,
        messages=[{
            "role": "system",
            "content": system_prompt
        },
            {
                "role": "user",
                "content": prompt
            }
        ],
        max_tokens=3000,
    )
    return chat_completion.choices[0].message.content

def create_agent(markdown_path: str = "bingenbash.md"):
    # helper function for document retriever tool.
    graph = rag(markdown_path)

    @tool
    def document_retreiver(query: str) -> str:
        """Search and retrieve information about binge n bash, a private theatre and event celebration experience. This information contains everything related to binge n bash, including theatre prices, available food items, branch information, and more."""
        response = graph.invoke({"question": query})
        return response['answer']

    # Helper function for bookings_database tool.
    system_prompt = sqlChatInfo()

    @tool
    def bookings_database(question: str) -> str:
        """Return the answer to a question which will be retrieved from a database,
         which contains all the customer booking information
         The database contains information like mail, phone number, name, date, slot, booking date, booking time,"""
        query = inference(question, system_prompt)
        query = extract_code_blocks(query)[0]
        # print(query)
        db = SQLDatabase.from_uri(uri)
        result = db.run_no_throw(query)
        return result

    llm = ChatGroq(model="qwen-2.5-32b")
    agent = create_react_agent(llm, [document_retreiver, bookings_database])
    return agent

def smol_agent(markdown_path: str = "bingenbash.md"):
    from smolagents import HfApiModel, CodeAgent, tool

    # helper function for document retriever tool.
    graph = rag(markdown_path)

    @tool
    def document_retreiver(query: str) -> str:
        """Search and retrieve information about binge n bash, a private theatre and event celebration experience.
         This information contains everything related to binge n bash, including theatre prices, available food items,
         branch information, and more.
         Args:
            query: The question to be answered.
        """
        response = graph.invoke({"question": query})
        return response['answer']

    # Helper function for bookings_database tool.
    system_prompt = sqlChatInfo()

    @tool
    def bookings_database(question: str) -> str:
        """Return the answer to a question which will be retrieved from a database,
         which contains all the customer booking information
         The database contains information like mail, phone number, name, date, slot, booking date, booking time.
         Args:
            question: The question to be answered.
        """
        query = inference(question, system_prompt)
        query = extract_code_blocks(query)[0]
        # print(query)
        db = SQLDatabase.from_uri(uri)
        result = db.run_no_throw(query)
        return result

    agent = CodeAgent(
        tools=[document_retreiver, bookings_database], model=HfApiModel(), max_steps=15, planning_interval=3, verbosity_level=2
    )
    return agent

def run_lang():
    agent = create_agent()
    config = {"configurable": {"thread_id": "def234"}}

    def print_stream(stream):
        for s in stream:
            message = s["messages"][-1]
            if isinstance(message, tuple):
                print(message)
            else:
                message.pretty_print()

    while True:
        question = input("You: ")
        if question == "quit":
            break
        # inputs = {"messages": [("user", "What are the top 3 most booked theatres and their prices?")]}
        inputs = {"messages": [("user", question)]}
        # print(agent.invoke(question))

        print_stream(agent.stream(inputs, stream_mode="values"))

def run_smol():
    agent = smol_agent()
    while True:
        question = input("You: ")
        if question == "quit":
            break
        # inputs = {"messages": [("user", "What are the top 3 most booked theatres and their prices?")]}
        agent.run(question)

if __name__ == "__main__":
    run_lang()
    # run_smol()