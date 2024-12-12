# QueryCraft

Query Craft is an AI-powered tool that simplifies database querying by translating natural language queries into SQL and MongoDB queries. This project utilizes Streamlit for the user interface and integrates OpenAI models for intelligent query generation.

## Features
- Connect to SQL or MongoDB databases using their respective URLs.
- Automatically retrieve and display the database schema.
- Generate and execute SQL and MongoDB queries based on natural language input.
- Display the results of the queries in a user-friendly interface.

## Requirements
- Python 3.7+
- An OpenAI API key

## Installation
1. Clone this repository:
   ```
   git clone <repository-url>
   cd <repository-folder>
   ```
2. Install dependencies:
   ```
   pip install -r requirements.txt
   ```
3. Create a `.env` file in the project root and add your OpenAI API key:
   ```
   OPENAI_API_KEY=your_openai_api_key
   ```

## Usage
1. Run the Streamlit app:
   ```
   streamlit run app.py
   ```
2. Enter the database URL in the sidebar.
3. Select the database type (SQL or MongoDB).
4. Click "Connect" to connect to the database.
5. For MongoDB, provide the connection URI in the format:
   ```
   mongodb+srv://<db_username>:<db_password>@cluster0.escsd.mongodb.net/?retryWrites=true&w=majority&appName=Cluster0
   ```
   After connecting to the cluster, select the desired database for querying.
6. Input your natural language query and click "Run Query".
7. View the generated query and results.

## UI 
### Query Craft UI
![image](https://github.com/user-attachments/assets/f5048e7a-b044-4556-8bbd-01121799609d)

### SQL Query Example
![image](https://github.com/user-attachments/assets/5f603b93-3940-42f3-a1ec-da877bdc3779)

### MongoDB Query Example
![image](https://github.com/user-attachments/assets/1a9bc39b-6be6-48f4-9e87-0f86a068fa2e)
![image](https://github.com/user-attachments/assets/da08d6b6-aa90-4921-9f8e-9fb0c14dac31)

## Dependencies
- Streamlit: For the interactive UI.
- Python-dotenv: To manage environment variables.
- SQLAlchemy: For SQL database connections.
- PyMongo: For MongoDB interactions.
- LangChain: For LLM-based query generation.

## Notes
- For SQL databases, provide the SQLAlchemy connection string (e.g., `sqlite:///example.db`).
- For MongoDB, provide the connection URI (e.g., `mongodb+srv://<db_username>:<db_password>@cluster0.escsd.mongodb.net/?retryWrites=true&w=majority&appName=Cluster0`).

## Limitations
- Ensure that your OpenAI API key has sufficient credits.
- MongoDB queries are generated using sample documents from collections; empty collections might limit functionality.

## License
This project is open-source and available under the MIT License.

## Acknowledgments
This project uses OpenAI's GPT models for query generation and Streamlit for the UI framework.
