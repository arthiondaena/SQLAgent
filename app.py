import streamlit as st
import os
import datetime
from utils import get_info_sqlalchemy, extract_code_blocks, sql_inference
from var import markdown_info, system_prompt_template
from smolagents import CodeAgent, tool, LiteLLMModel, OpenAIServerModel
from dotenv import dotenv_values
from sqlalchemy import create_engine, text
from typing import List, Tuple
import conf
import re
from smolagents.memory import ActionStep

env = dotenv_values()
uri = env['POSTGRES_URI']

def sqlChatInfo(uri: str = None) -> str:
    if uri is None:
        uri = env['POSTGRES_URI']
    db_info = get_info_sqlalchemy(uri)
    markdown = markdown_info.format(**db_info)
    system_prompt = system_prompt_template.format(markdown_info=markdown)
    return system_prompt

def smol_agent():
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

    model = OpenAIServerModel(
        model_id=conf.CODE_MODEL_1,
        api_base=conf.SQL_BASE_URL,
        api_key=env['SQL_MODEL_API_KEY']
    )

    agent = CodeAgent(
        tools=[bookings_database], model=model, max_steps=10, verbosity_level=2,
        additional_authorized_imports=['matplotlib.pyplot', 'ast', 'pandas']
    )
    return agent

def finetune_user_prompt(user_prompt: str) -> tuple:
    now = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    outputs_dir = "outputs"
    os.makedirs(outputs_dir, exist_ok=True)
    # Instruct agent to save output to file in outputs/ with current datetime as filename
    finetuned = (
        f"{user_prompt}\n"
        f"Please save any plot or output to a file in the '{outputs_dir}' directory. "
        f"Use the current datetime '{now}' as the filename (e.g., outputs/{now}.png or outputs/{now}.txt). "
        f"Return only the output file path as the final_answer."
    )
    return finetuned, now

def extract_code_from_text(text: str) -> str | None:
    """Extract code from the LLM's output."""
    pattern = r"```(?:py|python)?\s*\nn```"
    matches = re.findall(pattern, text, re.DOTALL)
    if matches:
        return "\n\n".join(match.strip() for match in matches)
    return None

def format_code_blocks(agent_logs: List) -> str:
    output_str = ""
    for i, log in enumerate(agent_logs):
        if isinstance(log, ActionStep):
            code_block = log.model_output
            output_str += f"# ---------------------- Step {i} ---------------------- \n"
            if code_block is not None:
                output_str += code_block
                output_str += "\n"

    return output_str

def display_agent_output(output_path: str):
    if output_path.endswith((".png", ".jpg", ".jpeg")):
        st.image(output_path)
    elif output_path.endswith(".txt"):
        with open(output_path, "r", encoding="utf-8") as f:
            st.text(f.read())
    elif output_path:
        st.write(f"Output file: {output_path}")
    else:
        st.warning("No output file generated.")

def main():
    st.set_page_config(page_title="SQL Agent Streamlit App", layout="wide")
    st.title("SQL Agent Streamlit UI")

    if "agent" not in st.session_state:
        st.session_state.agent = smol_agent()
    if "messages" not in st.session_state:
        st.session_state.messages = []

    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if msg.get("output_path"):
                display_agent_output(msg["output_path"])
            if msg.get("generated_code"):
                with st.expander("Show generated code", expanded=False):
                    st.code(msg["generated_code"], language="python")

    user_prompt = st.chat_input("Ask a question about bookings database:")
    if user_prompt:
        st.session_state.messages.append({"role": "user", "content": user_prompt})
        with st.chat_message("user"):
            st.markdown(user_prompt)

        finetuned_prompt, now = finetune_user_prompt(user_prompt)
        result = st.session_state.agent.run(finetuned_prompt)
        output_path = result
        steps = st.session_state.agent.memory.steps
        generated_code = format_code_blocks(steps)
        agent_msg = {
            "role": "assistant",
            "content": "Here is the result:",
            "output_path": output_path,
            "generated_code": generated_code
        }
        st.session_state.messages.append(agent_msg)
        with st.chat_message("assistant"):
            st.markdown("Here is the result:")
            display_agent_output(output_path)
            with st.expander("Show generated code", expanded=False):
                st.code(generated_code or "No code available.", language="python")

if __name__ == "__main__":
    main()
