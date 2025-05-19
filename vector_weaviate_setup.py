"""
One-off setup script for initializing and populating a Weaviate vector database with article chunks.

This script performs the following tasks:
1. Connects to a Weaviate cloud instance using the v4 client.
2. Creates an 'ArticleChunk' collection if it doesn't exist, configured with the text2vec-openai vectorizer.
3. Loads article data from a JSON file, cleans HTML content, and splits the text into manageable chunks.
4. Converts the chunks into Langchain Document objects.
5. Ingestes the documents into the Weaviate collection using Langchain's WeaviateVectorStore.

Dependencies:
- weaviate-client (v4)
- langchain-openai
- langchain-weaviate
- langchain-text-splitters
- langchain-core
- beautifulsoup4
- python-dotenv

Environment Variables:
- WEAVIATE_CLUSTER_URL: URL of the Weaviate cloud instance.
- WEAVIATE_API_KEY: API key for authenticating with Weaviate.
- OPENAI_API_KEY: API key for OpenAI, used by the text2vec-openai vectorizer.

Usage:
1. Ensure the required environment variables are set in a .env file.
2. Place the source JSON file (default: 'sources_06-05-25.json') in the same directory as this script.
3. Run the script: python vector_weaviate_setup.py

Note: This script is intended for one-off setup and data ingestion. It assumes the 'ArticleChunk' collection is manually created in the Weaviate console with the specified properties and vectorizer configuration.
"""

import os
import json
import weaviate
# from weaviate.auth import AuthApiKey # Not needed with connect_to_weaviate_cloud
from bs4 import BeautifulSoup
from langchain_openai import OpenAIEmbeddings # Needed for WeaviateVectorStore.from_documents
from dotenv import load_dotenv
from typing import List, Dict, Any
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document as LangchainDocument # Renamed to avoid conflict
from langchain_weaviate.vectorstores import WeaviateVectorStore # Langchain's Weaviate integration

# Import v4 specific modules
from weaviate.classes.init import Auth
import weaviate.classes.config as wvcc
import weaviate.classes.data as wvc_data # Needed for data object types if strict typing
from weaviate.collections.classes.batch import BatchResult

# Load environment variables from .env file
load_dotenv()

WEAVIATE_CLUSTER_URL = os.getenv("WEAVIATE_CLUSTER_URL")
WEAVIATE_API_KEY = os.getenv("WEAVIATE_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Weaviate class definition (now Collection in v4 terminology)
WEAVIATE_COLLECTION_NAME = "ArticleChunk" # Renaming class to reflect storing chunks

# Text splitting configuration
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200

def get_weaviate_client_v4():
    """Initializes and returns the Weaviate v4 client."""
    if not WEAVIATE_CLUSTER_URL or not WEAVIATE_API_KEY or not OPENAI_API_KEY:
        raise ValueError("Missing Weaviate Cluster URL, API key, or OpenAI API key in environment variables.")
    
    # Use v4 connect_to_weaviate_cloud method
    # See: https://weaviate.io/developers/weaviate/client-libraries/python/v3_v4_migration#instantiate-a-client
    client = weaviate.connect_to_weaviate_cloud(
        cluster_url=WEAVIATE_CLUSTER_URL,
        auth_credentials=Auth.api_key(WEAVIATE_API_KEY),
        headers={
            "X-OpenAI-Api-Key": OPENAI_API_KEY # Pass OpenAI key for Weaviate's text2vec-openai module
        }
    )
    
    # Check connection readiness
    try:
        client.is_ready()
        print("Weaviate client (v4) is ready.")
    except Exception as e:
        print(f"Weaviate client not ready: {e}")
        raise # Re-raise the exception if not ready

    return client

def create_collection(client: weaviate.WeaviateClient):
    """Creates the 'ArticleChunk' collection in Weaviate if it doesn't exist using v4 API."""
    
    # Define the collection configuration using v4 helper classes
    # See: https://weaviate.io/developers/weaviate/client-libraries/python/v3_v4_migration#create-a-collection
    collection_config = wvcc.Collection( 
        name=WEAVIATE_COLLECTION_NAME,
        description="Stores chunks of articles with their content and metadata.",
        vectorizer_config=wvcc.Configure.Vectorizer.text2vec_openai(
            model="text-embedding-3-small",
            dimensions=1536,
            vectorize_collection_name=False
        ),
        generative_config=wvcc.Configure.Generative.openai(), # Optional: Add generative module if needed for later RAG
        properties=[
            wvcc.Property(
                name="article_id",
                data_type=wvcc.DataType.TEXT,
                description="Unique identifier for the original article."
            ),
            wvcc.Property(
                name="title",
                data_type=wvcc.DataType.TEXT,
                description="Title of the original article."
            ),
             wvcc.Property(
                name="description",
                data_type=wvcc.DataType.TEXT,
                description="Original description of the article."
            ),
            # The 'content' property will be vectorized by the text2vec-openai module by default
            wvcc.Property(
                name="content",
                data_type=wvcc.DataType.TEXT,
                description="Cleaned textual content of the article chunk."
            ),
             wvcc.Property(
                name="url",
                data_type=wvcc.DataType.TEXT,
                description="URL of the original article."
            ),
            wvcc.Property(
                name="source_type",
                data_type=wvcc.DataType.TEXT,
                description="Type of the source."
            )
        ]
    )

    # Check if collection exists and create if not
    # In v4, existence check is different. We can use client.collections.get or list_all
    try:
        client.collections.get(WEAVIATE_COLLECTION_NAME)
        print(f"Collection '{WEAVIATE_COLLECTION_NAME}' already exists.")
    except weaviate.exceptions.UnexpectedStatusCodeException as e:
        # If get fails with a 404, the collection does not exist
        if "not found" in str(e):
             print(f"Collection '{WEAVIATE_COLLECTION_NAME}' not found. Creating...")
             client.collections.create(collection_config)
             print(f"Collection '{WEAVIATE_COLLECTION_NAME}' created successfully.")
        else:
            # Re-raise other unexpected errors
            raise
    except Exception as e:
         print(f"An error occurred while checking/creating collection: {e}")
         raise

def convert_chunks_to_langchain_documents(chunks_data: List[Dict[str, Any]]) -> List[LangchainDocument]:
    """Converts a list of chunk dictionaries to Langchain Document objects."""
    langchain_docs = []
    for chunk in chunks_data:
        doc = LangchainDocument(
            page_content=chunk.get("content", ""), 
            metadata={
                "article_id": chunk.get("article_id", ""),
                "title": chunk.get("title", ""),
                "url": chunk.get("url", ""),
                "source_type": chunk.get("source_type", ""),
                "description": chunk.get("description", "")
            }
        )
        langchain_docs.append(doc)
    return langchain_docs

def load_prepare_and_split_data(json_file_path: str = "sources_06-05-25.json") -> List[Dict[str, Any]]:
    """Loads data from JSON, cleans HTML, and prepares chunks as dictionaries."""
    all_chunk_dicts = []
    try:
        with open(json_file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except FileNotFoundError:
        print(f"Error: The file {json_file_path} was not found.")
        return []
    except json.JSONDecodeError:
        print(f"Error: Could not decode JSON from {json_file_path}.")
        return []

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        length_function=len,
        is_separator_regex=False,
    )

    for index, (doc_key, doc) in enumerate(data.get('reference_by_id', {}).items()):
        html_content = doc.get("text", "")
        soup = BeautifulSoup(html_content, "html.parser")
        cleaned_text = soup.get_text(separator=" ", strip=True)

        # Create a temporary LangchainDocument to leverage its metadata handling with splitter
        temp_doc_for_splitting = LangchainDocument(
            page_content=cleaned_text,
            metadata={
                "article_id": f"A{index+1}", 
                "title": doc.get("title", ""),
                "url": doc.get("url", ""),
                "source_type": doc.get("source_type", ""),
                "description": doc.get("description", "")
            }
        )
        
        split_docs = text_splitter.split_documents([temp_doc_for_splitting])

        for split_chunk_doc in split_docs:
            chunk_data = {
                "article_id": split_chunk_doc.metadata["article_id"],
                "title": split_chunk_doc.metadata["title"],
                "content": split_chunk_doc.page_content,
                "url": split_chunk_doc.metadata["url"],
                "source_type": split_chunk_doc.metadata["source_type"],
                "description": split_chunk_doc.metadata["description"]
            }
            all_chunk_dicts.append(chunk_data)
    
    print(f"Loaded, prepared, and split into {len(all_chunk_dicts)} chunk dictionaries.")
    return all_chunk_dicts

def main():
    """Main function to load data, prepare chunks, and ingest into Weaviate using Langchain."""
    print("Starting Weaviate data ingestion process using Langchain...")
    
    weaviate_client_v4 = None # Initialize to ensure it's closable in finally block
    try:
        # 1. Get Weaviate v4 client (needed by WeaviateVectorStore)
        weaviate_client_v4 = get_weaviate_client_v4()
        print("Successfully connected to Weaviate (v4 client for Langchain)." )

        # 2. Load, prepare, and split data into chunk dictionaries
        chunk_dictionaries = load_prepare_and_split_data()
        if not chunk_dictionaries:
            print("No data to ingest. Exiting.")
            return

        # 3. Convert chunk dictionaries to Langchain Document objects
        langchain_documents_for_ingestion = convert_chunks_to_langchain_documents(chunk_dictionaries)
        print(f"Converted {len(langchain_documents_for_ingestion)} chunks to Langchain Documents.")

        # 4. Define OpenAI embeddings (required by WeaviateVectorStore.from_documents)
        embeddings = OpenAIEmbeddings(model='text-embedding-3-small', dimensions=1536, openai_api_key=OPENAI_API_KEY)

        # 5. Ingest documents into Weaviate using Langchain's WeaviateVectorStore
        # This will add documents to the existing, manually created "ArticleChunk" collection.
        # Weaviate (server-side) should handle the vectorization if text2vec-openai is configured on the collection.
        # The `embedding` param in `from_documents` can sometimes be used by Langchain to generate embeddings if needed,
        # but if the Weaviate collection is correctly set up with a vectorizer, Weaviate will do it.
        print(f"Starting ingestion into Weaviate collection '{WEAVIATE_COLLECTION_NAME}'...")
        vector_store = WeaviateVectorStore.from_documents(
            client=weaviate_client_v4, # Pass the v4 client
            documents=langchain_documents_for_ingestion,
            embedding=embeddings, # Provide embeddings for Langchain compatibility
            index_name=WEAVIATE_COLLECTION_NAME, # Your manually created collection name
            by_text=False # Set to False. If True, it tries to embed with client-side func and may cause issues if Weaviate has its own vectorizer. We want Weaviate server to embed.
                          # However, the behavior also depends on Weaviate client version and Langchain adapter.
                          # If server-side embedding is desired (text2vec-openai in Weaviate schema), `by_text=False` is usually correct.
                          # The `embedding` object above will be used by Langchain to create query vectors if needed later by the retriever, ensuring consistency.
        )
        print(f"Successfully ingested {len(langchain_documents_for_ingestion)} documents into Weaviate collection '{WEAVIATE_COLLECTION_NAME}' via Langchain.")

    except ValueError as e:
        print(f"Configuration error: {e}")
    except Exception as e:
        print(f"An unexpected error occurred during Weaviate ingestion: {e}")
        import traceback
        traceback.print_exc()
    finally:
        if weaviate_client_v4:
            weaviate_client_v4.close()
            print("Weaviate client (v4) closed.")

if __name__ == "__main__":
    main()
    # Instructions for the King:
    # 1. Manually create the 'ArticleChunk' collection in your Weaviate console with the specified properties and text2vec-openai vectorizer.
    #    Properties: article_id (text), title (text), description (text), content (text, vectorized), url (text), source_type (text)
    #    Vectorizer: text2vec-openai, model: text-embedding-3-small, dimensions: 1536
    # 2. Ensure you have a .env file in the root of your project with:
    #    WEAVIATE_CLUSTER_URL="your_weaviate_cluster_url"
    #    WEAVIATE_API_KEY="your_weaviate_api_key"
    #    OPENAI_API_KEY="your_openai_api_key"
    # 3. Make sure 'sources_06-05-25.json' is in the same directory as this script, or update the path in `load_prepare_and_split_data`.
    # 4. Run this script from your terminal: python weaviate_setup.py
    # This will now use Langchain to upload the processed chunks to your pre-configured 'ArticleChunk' collection. 