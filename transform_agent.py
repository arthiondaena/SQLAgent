import os
from utils import get_info_sqlalchemy, extract_code_blocks
from var import db_info, markdown_info, system_prompt_template
from groq import Groq
from langchain_community.utilities import SQLDatabase
from smolagents import HfApiModel, CodeAgent, tool
from dotenv import load_dotenv

load_dotenv()
uri = os.environ['POSTGRES_URI']

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

def smol_agent():
    # Helper function for bookings_database tool.
    system_prompt = sqlChatInfo()

    @tool
    def bookings_database(question: str) -> str:
        """Return the answer to a question which will be retrieved from a database,
         which contains all the customer booking information
         The database contains information like mail, phone number, name, date, slot, booking date, booking time.
         Example function call
         >> bookings_database("What are the top 3 most booked theatres?")
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
        tools=[bookings_database], model=HfApiModel(), max_steps=20, verbosity_level=2, additional_authorized_imports=['matplotlib.pyplot']
    )
    return agent

def run_smol():
    agent = smol_agent()
    while True:
        question = input("You: ")
        if question == "quit":
            break
        # inputs = {"messages": [("user", "What are the top 3 most booked theatres and their prices?")]}
        agent.run(question)

if __name__ == '__main__':
    run_smol()
    # Plot a graph for number of bookings each theatre has received
