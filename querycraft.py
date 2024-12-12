import os
import streamlit as st
from dotenv import load_dotenv
from sqlalchemy import create_engine, inspect
from langchain_community.chat_models import ChatOpenAI
from langchain.agents import create_sql_agent
from langchain.agents.agent_toolkits import SQLDatabaseToolkit
from langchain_community.utilities import SQLDatabase
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain

# Load environment variables from .env file (for OpenAI API Key)
load_dotenv()
openai_key = os.getenv("OPENAI_API_KEY")

st.set_page_config(page_title="Database Query AI Agent", layout="wide")
st.title("Query Craft")

# Initialize session state variables
if 'connected' not in st.session_state:
    st.session_state.connected = False
if 'db_schema' not in st.session_state:
    st.session_state.db_schema = None
if 'db' not in st.session_state:
    st.session_state.db = None
if 'db_type' not in st.session_state:
    st.session_state.db_type = None
if 'client' not in st.session_state:
    st.session_state.client = None
if 'sql_agent' not in st.session_state:
    st.session_state.sql_agent = None
if 'collection_names' not in st.session_state:
    st.session_state.collection_names = []
if 'db_names' not in st.session_state:
    st.session_state.db_names = []
if 'db_name' not in st.session_state:
    st.session_state.db_name = ''

# Sidebar configuration
st.sidebar.header("Configuration")

# Database type selection
db_type = st.sidebar.selectbox("Select Database Type", ["SQL", "MongoDB"])

# Database URL input
db_url = st.sidebar.text_input(
    "Enter Database URL",
    help="For SQL, use SQLAlchemy connection string. For MongoDB, use MongoDB connection URI."
)

# Connect and Disconnect buttons
connect_button = st.sidebar.button("Connect")
disconnect_button = st.sidebar.button("Disconnect")

# Connect to the database
if connect_button and db_url:
    if db_type == "SQL":
        # SQL Database handling


        try:
            # Create SQLAlchemy engine
            engine = create_engine(db_url)

            # Create SQLDatabase object
            db = SQLDatabase(engine)

            # Retrieve schema information
            inspector = inspect(engine)
            tables = inspector.get_table_names()
            schema_str = ""
            for table_name in tables:
                columns = inspector.get_columns(table_name)
                schema_str += f"Table '{table_name}' has columns: "
                schema_str += ", ".join([column['name'] for column in columns])
                schema_str += "\n"

            # Initialize the LLM with model name
            llm_name = "gpt-3.5-turbo"
            llm = ChatOpenAI(api_key=openai_key, model_name=llm_name)

            # Create the SQLDatabaseToolkit
            toolkit = SQLDatabaseToolkit(db=db, llm=llm)

            # Create the SQL agent
            sql_agent = create_sql_agent(
                llm=llm,
                toolkit=toolkit,
                verbose=True,
                agent_executor_kwargs={'return_intermediate_steps': True}
            )

            # Store schema, db, and agent in session state
            st.session_state.db_schema = schema_str
            st.session_state.db = db
            st.session_state.sql_agent = sql_agent
            st.session_state.connected = True
            st.session_state.db_type = "SQL"

            st.success("Connected to the SQL database successfully!")
            st.markdown("### Database Schema:")
            st.text(schema_str)

        except Exception as e:
            st.error(f"An error occurred while connecting to the SQL database: {str(e)}")

    elif db_type == "MongoDB":
        # MongoDB handling
        from pymongo import MongoClient

        try:
            # Create MongoClient
            client = MongoClient(db_url)

            # List available databases
            db_names = client.list_database_names()
            st.session_state.db_names = db_names
            st.session_state.client = client
            st.session_state.connected = True
            st.session_state.db_type = "MongoDB"

            st.success(f"Connected to MongoDB successfully!")

        except Exception as e:
            st.error(f"An error occurred while connecting to MongoDB: {str(e)}")
else:
    if not db_url and connect_button:
        st.error("Please enter the database URL.")

# Always display database selection when connected to MongoDB
if st.session_state.connected and st.session_state.db_type == "MongoDB":
    st.sidebar.markdown("### Available Databases:")
    db_name = st.sidebar.selectbox(
        "Select Database",
        st.session_state.db_names,
        index=st.session_state.db_names.index(st.session_state.db_name) if st.session_state.db_name in st.session_state.db_names else 0,
        key='db_name_select'
    )
    if db_name != st.session_state.db_name:
        # Database has changed
        st.session_state.db_name = db_name
        st.session_state.db_schema = None  # Reset schema
        st.session_state.collection_names = []

    db = st.session_state.client[st.session_state.db_name]
    st.session_state.db = db

    # List collections
    collection_names = db.list_collection_names()
    st.session_state.collection_names = collection_names

    # Display the list of collections
    st.sidebar.markdown("### Available Collections:")
    st.sidebar.text("\n".join(collection_names))

    # Build combined schema for all collections
    if st.session_state.db_schema is None:
        schema_str = ""
        for collection_name in collection_names:
            collection = db[collection_name]
            sample_doc = collection.find_one()
            if sample_doc:
                schema_str += f"Collection '{collection_name}' sample document keys:\n"
                schema_str += ", ".join(sample_doc.keys())
                schema_str += "\n\n"
            else:
                schema_str += f"Collection '{collection_name}' is empty.\n\n"
        st.session_state.db_schema = schema_str

    st.markdown("### Database Schema:")
    st.text(st.session_state.db_schema)

# Disconnect from the database
if disconnect_button:
    st.session_state.connected = False
    st.session_state.db_schema = None
    st.session_state.db = None
    st.session_state.db_type = None
    st.session_state.sql_agent = None
    st.session_state.collection_names = []
    st.session_state.db_names = []
    st.session_state.db_name = ''
    if st.session_state.client:
        st.session_state.client.close()
        st.session_state.client = None
    st.success("Disconnected from the database.")

# Main query input
question = st.text_area("Enter your query in natural language:")

# Run Query button
if st.session_state.connected:
    if st.button("Run Query"):
        if question:
            if st.session_state.db_type == "SQL":
                # SQL Query Handling
                try:
                    # Use the stored SQL agent
                    with st.spinner("Running the query..."):
                        # Call the agent with return_only_outputs=False to get intermediate steps
                        response = st.session_state.sql_agent(question, return_only_outputs=False)
                    # Now, extract the SQL query from the intermediate steps
                    intermediate_steps = response['intermediate_steps']
                    # intermediate_steps is a list of tuples: (AgentAction, str)

                    # Let's iterate over the intermediate_steps and find the SQL query
                    sql_queries = []
                    for step in intermediate_steps:
                        action = step[0]
                        if action.tool == 'sql_db_query':
                            sql_queries.append(action.tool_input)

                    # Display all SQL queries
                    if sql_queries:
                        st.markdown("### Generated SQL Queries:")
                        for i, sql_query in enumerate(sql_queries, 1):
                            st.markdown(f"**Query {i}:**")
                            st.code(sql_query, language='sql')
                    else:
                        st.info("No SQL queries were generated.")
                    st.markdown("### Answer:")
                    st.write(response['output'])

                except Exception as e:
                    st.error(f"An error occurred: {str(e)}")
            elif st.session_state.db_type == "MongoDB":
                # MongoDB Query Handling using LLMChain
                try:
                    # Initialize the LLM with model name
                    llm_name = "gpt-3.5-turbo"
                    llm = ChatOpenAI(api_key=openai_key, model_name=llm_name)

                    # Prepare the prompt template
                    prompt_template = """
You are an expert MongoDB assistant. Given a user's natural language question and the schema of the MongoDB database, generate a MongoDB aggregation pipeline that answers the question.

**Important Instructions:**
- Analyze the user's question and determine which collection(s) are relevant.
- Output a JSON object with two keys: "collection" and "pipeline".
  - "collection": The name of the collection to run the pipeline on.
  - "pipeline": The aggregation pipeline as a JSON array.
- Ensure that the output is valid JSON.
- All keys and string values must be enclosed in double quotes.
- Do not include any comments or additional text; provide only the JSON output.

Schema:
{schema}

Question:
{question}

Provide only the JSON object with "collection" and "pipeline" keys, without any additional text.
"""
                    query_with_prompt = PromptTemplate(
                        template=prompt_template,
                        input_variables=["schema", "question"]
                    )

                    # Create the LLMChain
                    llm_chain = LLMChain(llm=llm, prompt=query_with_prompt, verbose=True)

                    # Generate the aggregation pipeline
                    with st.spinner("Generating MongoDB aggregation pipeline..."):
                        response = llm_chain.run({
                            "schema": st.session_state.db_schema,
                            "question": question
                        })

                    generated_text = response.strip()

                    # Parse the generated JSON output
                    import json
                    try:
                        # Handle cases where the model includes code blocks
                        import re
                        json_match = re.search(r'```json\s*(\{.*?\})\s*```', generated_text, re.DOTALL)
                        if json_match:
                            json_str = json_match.group(1)
                        else:
                            json_str = generated_text  # Fallback to use the entire content

                        output = json.loads(json_str)
                        collection_name = output['collection']
                        aggregation_pipeline = output['pipeline']

                    except (json.JSONDecodeError, KeyError) as e:
                        st.error("Failed to parse the JSON output.")
                        st.text(generated_text)
                        st.stop()

                    st.markdown("### Generated MongoDB Query:")
                    st.json(aggregation_pipeline)

                    # Execute the aggregation pipeline
                    with st.spinner("Running the aggregation pipeline..."):
                        if collection_name in st.session_state.collection_names:
                            collection = st.session_state.db[collection_name]
                            results = list(collection.aggregate(aggregation_pipeline))
                        else:
                            st.error(f"Collection '{collection_name}' does not exist in the database.")
                            st.stop()

                    st.markdown("### Query Results:")
                    st.write(results)

                except Exception as e:
                    st.error(f"An error occurred: {str(e)}")
        else:
            st.error("Please enter a query.")
else:
    st.info("Please connect to a database.")
