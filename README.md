# SQLagent

SQLagent is a flexible agentic framework for interacting with SQL databases and answering natural language questions. It supports both general question-answering (using LangGraph and SmolAgents) and code generation tasks (such as plotting graphs from database queries).

## Features

- **agent.py**: Contains both LangGraph and SmolAgents-based agents for general question answering and retrieval-augmented generation (RAG).
- **transform_agent.py**: Uses SmolAgents' CodeAgent for code generation tasks, such as plotting graphs for monthly bookings.
- **conf.py**: Central configuration for model endpoints, model names, and prompt templates.

## Setup

### 1. Create a Virtual Environment

```sh
python -m venv .venv
```

Activate the virtual environment:

- On Windows:
  ```sh
  .venv\Scripts\activate
  ```
- On Unix/Mac:
  ```sh
  source .venv/bin/activate
  ```

### 2. Install Requirements

```sh
pip install -r requirements.txt
```

### 3. Environment Variables

Create a `.env` file in the project root with the following variables:

```
GROQ_API_KEY=your_groq_api_key
POSTGRES_URI=your_postgres_connection_uri
CHAT_API_KEY=your_ollama_or_openai_key
SQL_MODEL_API_KEY=your_sql_model_api_key
```

Other optional variables can be found in the provided `.env` sample.

### 4. Configuration

Edit `conf.py` to set the base URLs and model names for your agents and LLMs. For example:

```python
BASE_URL = "http://localhost:11434"
CODE_MODEL = "llama3.3:70b"
SQL_BASE_URL = "https://openrouter.ai/api/v1"
SQL_MODEL = "meta-llama/llama-3.3-70b-instruct:free"
```

Adjust these values according to your deployment and model preferences.

## Usage

### Running the General Agent

- **LangGraph agent:**  
  By default, running `python agent.py` will use the LangGraph agent via the `run_lang()` function.

  ```sh
  python agent.py
  ```

- **SmolAgents agent:**  
  To use the SmolAgents agent instead, open `agent.py` and comment out the `run_lang()` line in the `if __name__ == "__main__":` block, then uncomment the `run_smol()` line. Example:

  ```python
  # run_lang()
  run_smol()
  ```

  Then run:

  ```sh
  python agent.py
  ```

  This will prompt for user input and answer general questions about your database or markdown documentation.

### Running the Code Agent

- **SmolAgents CodeAgent (code generation, e.g., plotting):**

  ```sh
  python transform_agent.py
  ```

  This agent is suited for tasks like generating code to plot graphs from SQL data.

## Notes

- Ensure your `.env` variables are set correctly for database and API access.
- You can switch between different agent types and models by editing `agent.py`, `transform_agent.py`, and `conf.py`.
- For advanced configuration, refer to the comments in each file.