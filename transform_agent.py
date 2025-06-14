import conf
from utils import get_info_sqlalchemy, extract_code_blocks, sql_inference
from var import markdown_info, system_prompt_template
from smolagents import CodeAgent, tool, LiteLLMModel, OpenAIServerModel
from dotenv import dotenv_values
from sqlalchemy import create_engine, text
from typing import List, Tuple

env = dotenv_values()
uri = env['POSTGRES_URI']

def sqlChatInfo(uri: str = None) -> str:
    """Get the information about a database in the form of a system prompt for the SQL agent"""
    if uri is None:
        uri = env['POSTGRES_URI']

    db_info = get_info_sqlalchemy(uri)
    markdown = markdown_info.format(**db_info)
    system_prompt = system_prompt_template.format(markdown_info=markdown)
    return system_prompt

def smol_agent():
    # Helper function for bookings_database tool.
    system_prompt = sqlChatInfo()

    @tool
    def bookings_database(question: str) -> List[Tuple]:
        """Return the answer to a question which will be retrieved from a database,
         which contains all the customer booking information
         The functio returns a list of tuples. Each tuple contains the resultant row from the database query.
         The database contains information like mail, phone number, name, date, slot, booking date, booking time.
         DO NOT ask the bookings_database to retrieve all the booking information or all the theatre bookings.
         Example question:
         1. What are the top 3 most booked theatres?
         2. Who are the top 3 most visited customers?
         3. What are the number of bookings monthly?
         Example function call
         >> bookings_database("What are the top 3 most booked theatres?")
         Args:
            question: The question to be answered.
        """
        query = sql_inference(question, system_prompt)
        query = extract_code_blocks(query)[0]
        db = create_engine(uri)
        with db.connect() as conn:
            q_result = conn.execute(text(query)).fetchall()
        result = [t._tuple() for t in q_result]
        return result

    # model = LiteLLMModel(
    #     model_id=f"ollama/{conf.CODE_MODEL}",
    #     api_base=conf.BASE_URL,
    #     api_key=os.environ['CHAT_API_KEY'],
    #     keep_alive=-1
    # )

    model = OpenAIServerModel(
        model_id=conf.CODE_MODEL_1,
        api_base=conf.SQL_BASE_URL,
        api_key=env['SQL_MODEL_API_KEY']
    )

    agent = CodeAgent(
            tools=[bookings_database], model=model, max_steps=10, verbosity_level=2, additional_authorized_imports=['matplotlib.pyplot', 'ast', 'pandas']
    )
    return agent

def run_smol():
    agent = smol_agent()
    while True:
        question = input("You: ")
        if question == "quit":
            break
        # inputs = {"messages": [("user", "What are the top 3 most booked theatres and their prices?")]}
        output = agent.run(question)
        print("Agent: ", output)

if __name__ == '__main__':
    # smol_agent()
    run_smol()
    # Plot a graph for number of bookings each theatre has received
    # Plot a graph of individual theatre month wise bookings
    # Plot top 5 most booked theaters and top 5 most visited customers graphs side by side