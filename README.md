# Personalized News Briefing Generator
This project is a coding challenge for Serendipity AI company, aiming to create a personalised briefing AI agent.

This is a Streamlit web application that generates personalized news briefings for users based on their role and interests. It leverages a combination of Retrieval Augmented Generation (RAG) with a Weaviate vector database and Knowledge Graph (KG) querying with a Neo4j graph database to gather relevant information. An OpenAI Large Language Model (LLM) is then used to synthesize this information into a coherent and tailored briefing.

## Architecture

The application consists of the following main components:

1.  **Streamlit Frontend (`briefing_generator.py`)**:
    *   Provides the user interface for selecting roles, interests, retrieval methods, and LLM models.
    *   Displays the generated briefing and retrieved source documents.
    *   Orchestrates the data retrieval and briefing generation process.

2.  **Data Sources**:
    *   **Weaviate Vector Database**: Stores article chunks, vectorized for semantic search (RAG). The `vector_weaviate_setup.py` script is used for initial setup and data ingestion from a JSON source file (`sources_06-05-25.json`).
    *   **Neo4j Graph Database**: Stores articles and their relationships to topics (KG). The `kg_neo4j_setup.py` script is used to process the source JSON, classify articles into topics using an LLM, and populate the graph.

3.  **Retrieval Mechanisms**:
    *   **RAG-based Retrieval**: Queries the Weaviate database for article chunks semantically similar to the user's role and interests.
    *   **KG-based Retrieval**: Queries the Neo4j database for articles linked to the user's selected interests. The content of these articles is then fetched from Weaviate (where full text/chunks are stored).
    *   **Hybrid Retrieval**: Combines results from both RAG and KG methods, de-duplicating to provide a comprehensive set of documents.

4.  **Synthesis Engine (Langchain & OpenAI)**:
    *   Uses Langchain to structure the interaction with an OpenAI LLM (e.g., GPT-4o-mini).
    *   A prompt template guides the LLM to synthesize the retrieved document chunks into a personalized briefing, following specific formatting and citation guidelines.

5.  **Configuration (`.env` file)**:
    *   Stores API keys and connection URIs for Weaviate, Neo4j, and OpenAI.

## Key Features

*   **Personalized Briefings**: Generates news briefings tailored to user-defined roles (e.g., Healthcare Policy Analyst, Tech Startup Founder) and specific interests.
*   **Multiple Retrieval Strategies**:
    *   **RAG-based**: Utilizes semantic search over a vector database.
    *   **KG-based**: Leverages relationships in a knowledge graph to find relevant articles by topic.
    *   **Hybrid**: Combines the strengths of both RAG and KG retrieval.
*   **Selectable LLM Models**: Allows users to choose from different OpenAI LLM models for briefing synthesis.
*   **Interactive UI**: Built with Streamlit for easy interaction and configuration.
*   **Source Transparency**: Displays the retrieved documents that were used to generate the briefing, including metadata like source title, URL, and article ID.
*   **Modular Setup Scripts**: Separate scripts for initializing the Weaviate vector store (`vector_weaviate_setup.py`) and the Neo4j knowledge graph (`kg_neo4j_setup.py`).

## Prerequisites

*   Python 3.8+
*   Access to:
    *   Weaviate Cloud instance (or local Weaviate)
    *   OpenAI API key
    *   Neo4j Aura instance (or local Neo4j)
*   `git` (for cloning the repository)

## Setup Instructions

1.  **Clone the Repository**:
    ```bash
    git clone https://github.com/daniel1014/RAG_KG_AGENT.git
    cd RAG_KG_AGENT
    ```

2.  **Create a Virtual Environment and Install Dependencies**:
    ```bash
    python -m venv .venv
    source .venv/bin/activate  # On Windows use: .venv\Scripts\activate
    pip install -r requirements.txt
    ```

3.  **Set Up Environment Variables**:
    Copy `.env` file from my email or Create a `.env` file in the root directory of the project by copying the example below and filling in your credentials:
    ```env
    WEAVIATE_CLUSTER_URL="your_weaviate_cluster_url"
    WEAVIATE_API_KEY="your_weaviate_api_key"
    OPENAI_API_KEY="your_openai_api_key"
    NEO4J_URI="your_neo4j_uri"
    NEO4J_PASSWORD="your_neo4j_password"
    # NEO4J_DATABASE="neo4j" # Optional: specify if not using the default 'neo4j' database
    ```

4.  **Prepare the Data Source File**:
    Ensure the `sources_06-05-25.json` file is present in the root directory. This file contains the articles that will be ingested into the databases.

5.  **Set Up Weaviate Vector Database**:
*   **Set Up Weaviate Vector Database (Optional)**: This step is only necessary if you are setting up the database for the first time or wish to refresh its content.
*   The `vector_weaviate_setup.py` script will attempt to create the `ArticleChunk` collection if it doesn't exist, configured for `text2vec-openai`.
*   Run the script:
    ```bash
    python vector_weaviate_setup.py
    ```
*   This script will load data from `sources_06-05-25.json`, clean it, chunk it, and ingest it into your Weaviate `ArticleChunk` collection.

6.  **Set Up Neo4j Knowledge Graph**:
    *   **Set Up Neo4j Knowledge Graph (Optional)**: This step is only necessary if you are setting up the database for the first time or wish to refresh its content.
    *   The `kg_neo4j_setup.py` script will:
        *   Load data from `sources_06-05-25.json`.
        *   Use an LLM (GPT-4.1-mini) to classify articles into topics.
        *   Create `Article` and `Topic` nodes, and `HAS_TOPIC` relationships in Neo4j.
        *   **Note**: This script will delete all existing data in your Neo4j database before populating it.
    *   Run the script:
        ```bash
        python kg_neo4j_setup.py
        ```

## Running the Application

Once the setup is complete (virtual environment activated, dependencies installed, .env file configured, and databases populated):

1.  **Start the Streamlit Application**:
    ```bash
    streamlit run briefing_generator.py
    ```
2.  Open your web browser and navigate to the local URL provided by Streamlit (usually `http://localhost:8501`).
3.  In the application:
    *   Select your desired **Role**.
    *   Choose your **Interests**.
    *   Pick a **briefing generation method** (RAG-based, KG-based, or Hybrid).
    *   Select an **LLM Model**.
    *   Click "Generate Briefing".

## Scripts Overview

*   `briefing_generator.py`: The main Streamlit application file.
*   `vector_weaviate_setup.py`: Script to initialize and populate the Weaviate vector database with article chunks. It handles data loading, cleaning, chunking, and ingestion.
*   `kg_neo4j_setup.py`: Script to construct the knowledge graph in Neo4j. It involves LLM-based topic classification of articles and creating nodes and relationships.
*   `api.py`: Contains Pydantic models (`Article`, `TopicList`) used, particularly by `kg_neo4j_setup.py` for structured LLM output.
*   `requirements.txt`: Lists the Python dependencies for the project.
*   `sources_06-05-25.json`: The JSON data file containing the articles.

## Troubleshooting

*   **Missing API Keys/URIs**: Ensure your `.env` file is correctly configured and loaded. The application will show errors or warnings if essential configurations are missing.
*   **Database Connection Issues**: Verify that your Weaviate and Neo4j instances are running and accessible from the machine where you are running the application. Check network configurations and credentials.
*   **Dependency Errors**: Make sure all dependencies in `requirements.txt` are installed in your active virtual environment.
*   **LLM Errors**: Ensure your OpenAI API key is valid and has sufficient quota. 