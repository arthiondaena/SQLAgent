import requests
from langchain_community.utilities import SQLDatabase
from langchain_community.tools.sql_database.tool import ListSQLDatabaseTool, InfoSQLDatabaseTool
from sqlalchemy import (
    create_engine,
    MetaData,
    inspect,
    Table,
    select,
    distinct
)
from sqlalchemy.schema import CreateTable
from sqlalchemy.exc import ProgrammingError
from sqlalchemy.engine import Engine
import re

def get_all_groq_model(api_key:str=None) -> list:
    """Uses Groq API to fetch all the available models."""
    if api_key is None:
        raise ValueError("API key is required")
    url = "https://api.groq.com/openai/v1/models"

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }

    response = requests.get(url, headers=headers)

    data = response.json()['data']
    model_ids = [model['id'] for model in data]

    return model_ids

def validate_api_key(api_key:str) -> bool:
    """Validates the Groq API key using the get_all_groq_model function."""
    if len(api_key) == 0:
        return False
    try:
        get_all_groq_model(api_key=api_key)
        return True
    except Exception as e:
        return False

def validate_uri(uri:str) -> bool:
    """Validates the SQL Database URI using the SQLDatabase.from_uri function."""
    try:
        SQLDatabase.from_uri(uri)
        return True
    except Exception as e:
        return False

def get_info(uri:str) -> dict[str, str] | None:
    """Gets the dialect name, accessible tables and table schemas using the SQLDatabase toolkit"""
    db = SQLDatabase.from_uri(uri)
    dialect = db.dialect
    # List all the tables accessible to the user.
    access_tables = ListSQLDatabaseTool(db=db).invoke("")
    # List the table schemas of all the accessible tables.
    tables_schemas = InfoSQLDatabaseTool(db=db).invoke(access_tables)
    return {'sql_dialect': dialect, 'tables': access_tables, 'tables_schema': tables_schemas}

def get_sample_rows(engine:Engine, table:Table, row_count: int = 3) -> str:
    """Gets the sample rows of a table using the SQLAlchemy engine"""
    # build the select command
    command = select(table).limit(row_count)

    # save the columns in string format
    columns_str = "\t".join([col.name for col in table.columns])

    try:
        # get the sample rows
        with engine.connect() as connection:
            sample_rows_result = connection.execute(command)  # type: ignore
            # shorten values in the sample rows
            sample_rows = list(
                map(lambda ls: [str(i)[:100] for i in ls], sample_rows_result)
            )

        # save the sample rows in string format
        sample_rows_str = "\n".join(["\t".join(row) for row in sample_rows])

    # in some dialects when there are no rows in the table a
    # 'ProgrammingError' is returned
    except ProgrammingError:
        sample_rows_str = ""

    return (
        f"{row_count} rows from {table.name} table:\n"
        f"{columns_str}\n"
        f"{sample_rows_str}"
    )

def get_unique_values(engine:Engine, table:Table) -> str:
    """Gets the unique values of each column in a table using the SQLAlchemy engine"""
    unique_values = {}
    for column in table.c:
        command = select(distinct(column))

        try:
            # get the sample rows
            with engine.connect() as connection:
                result = connection.execute(command)  # type: ignore
                # shorten values in the sample rows
                unique_values[column.name] = [str(u) for u in result]

            # save the sample rows in string format
            # sample_rows_str = "\n".join(["\t".join(row) for row in sample_rows])
            # in some dialects when there are no rows in the table a
            # 'ProgrammingError' is returned
        except ProgrammingError:
            unique_values[column.name] = ""

    output_str = f"Unique values of each column in {table.name}: \n"
    for column, values in unique_values.items():
        output_str += f"{column} has {len(values)} unique values: {' '.join(values[:20])}"
        if len(values) > 20:
            output_str += ", ...."
        output_str += "\n"

    return output_str

def get_info_sqlalchemy(uri:str) -> dict[str, str] | None:
    """Gets the dialect name, accessible tables and table schemas using the SQLAlchemy engine"""
    engine = create_engine(uri)
    # Get dialect name using inspector
    inspector = inspect(engine)
    dialect = inspector.dialect.name
    # Metadata for tables and columns
    m = MetaData()
    m.reflect(engine)

    tables = {}
    for table in m.tables.values():
        tables[table.name] = str(CreateTable(table).compile(engine)).rstrip()
        tables[table.name] += "\n\n/*"
        tables[table.name] += "\n" + get_sample_rows(engine, table)+"\n"
        tables[table.name] += "\n" + get_unique_values(engine, table)+"\n"
        tables[table.name] += "*/"

    return {'sql_dialect': dialect, 'tables': ", ".join(tables.keys()), 'tables_schema': "\n\n".join(tables.values())}

def extract_code_blocks(text):
    pattern = r"```(?:\w+)?\n(.*?)\n```"
    matches = re.findall(pattern, text, re.DOTALL)
    return matches

if __name__ == "__main__":
    from dotenv import load_dotenv
    import os
    load_dotenv()

    # uri = os.getenv("POSTGRES_URI")
    # print(get_info_sqlalchemy(uri))
    print(validate_api_key(os.getenv("GROQ_API_KEY")))