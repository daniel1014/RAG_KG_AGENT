"""
briefing_generator.py

This script implements a Streamlit application for generating personalized news briefings.
It connects to a Weaviate vector database for RAG-based retrieval, a Neo4j graph
database for KG-based retrieval, and uses an OpenAI Large Language Model (LLM)
via Langchain to synthesize the briefing.

The application allows users to select a role (e.g., Healthcare Policy Analyst,
Tech Startup Founder), specify interests, choose a retrieval method (RAG, KG, Hybrid),
select an LLM model, and generates a briefing tailored to their input based on
the retrieved documents.

Key functionalities include:
- Loading configuration from environment variables.
- Connecting to Weaviate, Neo4j, and OpenAI services.
- Defining user roles and their associated interests.
- Building a Streamlit UI for user input.
- Retrieving relevant documents using RAG (vector search), KG (graph traversal),
  or a Hybrid approach combining both.
- Synthesizing the retrieved information into a coherent briefing using an LLM.
- Displaying the generated briefing and retrieved documents in the UI.

Dependencies:
- streamlit
- python-dotenv
- weaviate-client
- langchain
- langchain-openai
- langchain-weaviate
- neo4j

Environment variables required:
- WEAVIATE_CLUSTER_URL
- WEAVIATE_API_KEY
- OPENAI_API_KEY
- NEO4J_URI
- NEO4J_PASSWORD
"""

import streamlit as st
import os
from dotenv import load_dotenv
import weaviate
from weaviate.classes.init import Auth
import weaviate.classes.query as wvc_query # Added for Filter
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough, RunnableParallel # RunnablePassthrough might be removed if not used in new structure
from langchain_core.output_parsers import StrOutputParser
from langchain_weaviate.vectorstores import WeaviateVectorStore
from langchain.schema import Document
from typing import List, Dict, Any # Added Dict, Any

import neo4j # Added for Neo4j

# --- Configuration Loading ---

def load_configuration():
    """Loads environment variables and defines roles and interests."""
    load_dotenv()

    config = {
        "weaviate_cluster_url": os.getenv("WEAVIATE_CLUSTER_URL"),
        "weaviate_api_key": os.getenv("WEAVIATE_API_KEY"),
        "openai_api_key": os.getenv("OPENAI_API_KEY"),
        "neo4j_uri": os.getenv("NEO4J_URI"), # Added Neo4j URI
        "neo4j_user": "neo4j", # Default Neo4j user
        "neo4j_password": os.getenv("NEO4J_PASSWORD"), # Added Neo4j Password
        "roles_and_interests": {
            "Healthcare Policy Analyst": [
                "Healthcare reform",
                "Public health data",
                "Telemedicine",
                "Pharmaceutical innovation",
                "Health equity",
                "Government policy updates",
            ],
            "Tech Startup Founder": [
                "Artificial intelligence",
                "Venture capital",
                "Startup funding",
                "SaaS trends",
                "Product management",
            ],
        }
    }

    # Ensure essential environment variables are loaded
    if not config["weaviate_cluster_url"] or not config["weaviate_api_key"] or not config["openai_api_key"]:
        st.error("Missing Weaviate Cluster URL, API key, or OpenAI API key in environment variables. Please check your .env file.")
        st.stop() # Stop the app if essential variables are missing
    
    if not config["neo4j_uri"] or not config["neo4j_password"]:
        st.warning("Missing Neo4j URI or Password in environment variables. KG-based features will be unavailable.")
        # Don't stop the app, RAG might still work

    return config

# --- Weaviate Connection and Retriever Setup ---

@st.cache_resource # Cache the Weaviate client connection
def get_weaviate_client(_config: dict) -> weaviate.WeaviateClient | None:
    """Initializes and returns the Weaviate v4 client.

    Returns:
        weaviate.WeaviateClient | None: The initialized Weaviate client instance, or None if connection fails.
    """
    try:
        client = weaviate.connect_to_weaviate_cloud(
            cluster_url=_config["weaviate_cluster_url"],
            auth_credentials=Auth.api_key(_config["weaviate_api_key"]),
            headers={
                "X-OpenAI-Api-Key": _config["openai_api_key"] # Pass OpenAI key for Weaviate's text2vec-openai module
            }
        )
        client.is_ready()
        print("Weaviate client is ready.")
        return client
    except Exception as e:
        st.error(f"Failed to connect to Weaviate: {e}")
        st.stop() # Stop the app if connection fails
        return None # Should not reach here due to st.stop()

@st.cache_resource # Cache the retriever
def get_retriever(_weaviate_client: weaviate.WeaviateClient, _config: dict) -> callable:
    """
    Initializes and returns a custom retriever function that queries Weaviate directly
    and constructs Langchain Document objects, handling potential None values in content.

    Args:
        _weaviate_client: The initialized Weaviate client.
        _config: The application configuration dictionary.

    Returns:
        callable: A function that takes a query string (str) and returns a list of Langchain Document objects (List[Document]).
    """
    COLLECTION_NAME = "ArticleChunk"

    def custom_weaviate_retriever(query: str) -> List[Document]:
        print(f"DEBUG: custom_weaviate_retriever invoked with query: '{query}'")
        
        # Fetch 'text' and other relevant metadata properties
        return_properties = ["text", "title", "url", "article_id", "description", "source_type"]

        try:
            response = _weaviate_client.collections.get(COLLECTION_NAME).query.near_text(
                query=query,
                distance=0.5,    # threshold for similarity retrieval 
                limit=20,       # limit the maximum number of results to 20
                return_properties=return_properties,
                return_metadata=weaviate.classes.query.MetadataQuery(distance=True, score=True)
            )

            processed_docs: List[Document] = []
            if response and response.objects:
                for i, item in enumerate(response.objects):
                    # Extract content, defaulting to empty string if None or TEXT_KEY is missing
                    page_content = item.properties.get("text", "") # Get content (namely text) from properties
                    if page_content is None:
                        page_content = ""

                    metadata = {
                        "source_uuid": str(item.uuid), # Weaviate internal ID
                        "article_id": item.properties.get("article_id", ""),
                        "title": item.properties.get("title", "N/A"),
                        "url": item.properties.get("url", "#"),
                        "description": item.properties.get("description", ""),
                        "source_type": item.properties.get("source_type", "weaviate_chunk")
                    }
                    if item.metadata:
                        if item.metadata.distance is not None:
                             metadata["distance"] = item.metadata.distance

                    # --- Added logging for content and metadata ---
                    print(f"DEBUG: Processing vector chunk {i+1}/{len(response.objects)}")
                    print(f"DEBUG: {metadata}\nDEBUG: Content: {str(page_content)[:200]}...") # Log snippet
                    # --- End of added logging ---

                    processed_docs.append(Document(page_content=str(page_content), metadata=metadata))
                print(f"DEBUG: Weaviate direct query returned {len(response.objects)} objects.")
            else:
                print("DEBUG: No objects returned from Weaviate direct query.")
            
            return processed_docs

        except Exception as e:
            print(f"DEBUG: Error during custom_weaviate_retriever execution: {e}")
            import traceback
            print(f"DEBUG: Traceback: {traceback.format_exc()}")
            return [] # Return empty list on error

    # The original safe_retriever wrapper is no longer needed as we handle Document creation directly.
    return custom_weaviate_retriever

# --- Neo4j Connection Setup ---
@st.cache_resource
def get_neo4j_driver(_config: dict) -> neo4j.GraphDatabase.driver:
    """Initializes and returns the Neo4j driver."""
    if not _config.get("neo4j_uri") or not _config.get("neo4j_password"):
        st.warning("Neo4j URI or Password not configured. KG-based features will be disabled.")
        return None
    try:
        driver = neo4j.GraphDatabase.driver(
            _config["neo4j_uri"],
            auth=(_config["neo4j_user"], _config["neo4j_password"])
        )
        driver.verify_connectivity()
        print("Neo4j driver is ready.")
        return driver
    except Exception as e:
        st.error(f"Failed to connect to Neo4j: {e}")
        # st.stop() # Decide if app should stop; for now, allow RAG to work
        return None

# --- RAG Chain Setup ---

def get_synthesis_chain(llm: ChatOpenAI, rag_prompt_template: ChatPromptTemplate) -> RunnablePassthrough:
    """
    Defines the core synthesis part of the chain (prompt, LLM, output parser).
    Context will be injected before this chain.
    """

    # Removed duplicate prompt template definition. It's now passed as an argument.

    output_parser = StrOutputParser()

    # Removed debug prints - will capture in main and store in session state

    synthesis_chain = rag_prompt_template | llm | output_parser
    return synthesis_chain


# --- Document Retriever Functions ---

def get_rag_documents(query: str, weaviate_retriever_callable: callable) -> List[Document]:
    """Retrieves documents from Weaviate based on a query."""
    print(f"DEBUG: get_rag_documents called with query: '{query}'")
    try:
        docs = weaviate_retriever_callable(query)
        st.session_state.retrieved_docs_rag = docs # Store for potential separate display
        return docs
    except Exception as e:
        print(f"Error in get_rag_documents: {e}")
        st.error(f"Error retrieving RAG documents: {e}")
        return []

def get_kg_documents(
    selected_interests: List[str],
    neo4j_driver: neo4j.GraphDatabase.driver,
    weaviate_client: weaviate.WeaviateClient,
    config: dict
) -> List[Document]:
    """
    Retrieves documents from Neo4j based on interests, then enriches with content from Weaviate.
    """
    print(f"DEBUG: get_kg_documents called with interests: {selected_interests}")
    if not neo4j_driver:
        st.warning("Neo4j driver not available. Skipping KG-based retrieval.")
        return []
    if not weaviate_client: # Should not happen if app proceeds
        st.error("Weaviate client not available for KG document enrichment.")
        return []

    COLLECTION_NAME = "ArticleChunk" # Weaviate collection for chunks
    kg_docs: List[Document] = []
    processed_article_ids = set() # To avoid duplicate processing for same article_id

    cypher_query = """
    MATCH (a:Article)-[:HAS_TOPIC]->(t:Topic)
    WHERE t.topic_name IN $selected_interests OR t.topic_id IN $selected_interests
    RETURN DISTINCT a.article_id AS article_id, a.title AS title, a.description AS description, a.url AS url, a.source_type AS source_type
    """
    # In Neo4j, relationships are (a:Article)-[:HAS_TOPIC]->(t:Topic) based on kg_neo4j_setup.py

    try:
        with neo4j_driver.session(database=config.get("NEO4J_DATABASE", "neo4j")) as session: # Use default neo4j db if not specified
            results = session.run(cypher_query, selected_interests=selected_interests)
            neo4j_articles = [dict(record) for record in results]
        
        print(f"DEBUG: Neo4j query returned {len(neo4j_articles)} articles for interests: {selected_interests}")

        for article_data in neo4j_articles:
            article_id = article_data.get("article_id")
            if not article_id or article_id in processed_article_ids:
                continue
            
            processed_article_ids.add(article_id)
            print(f"DEBUG: Processing article_id {article_id} from Neo4j.")

            # Fetch all chunks for this article_id from Weaviate
            try:
                response = weaviate_client.collections.get(COLLECTION_NAME).query.fetch_objects(
                    filters=wvc_query.Filter.by_property("article_id").equal(article_id),
                    limit=100, # Assuming max 100 chunks per article
                    return_properties=["text"] # Only need content for concatenation
                )
                
                article_full_content = ""
                if response and response.objects:
                    # Sort chunks if order is important and available, e.g., by a chunk_id or offset
                    # For now, just concatenate in retrieved order.
                    for chunk_obj in response.objects:
                        chunk_content = chunk_obj.properties.get("text", "")
                        if chunk_content:
                            article_full_content += chunk_content + "\n\n" # Add spacing between chunks
                    article_full_content = article_full_content.strip()
                
                if not article_full_content:
                    print(f"DEBUG: No content found in Weaviate for article_id {article_id}. Skipping.")
                    continue

                # Create a single Document for the entire article from KG perspective
                doc_metadata = {
                    "article_id": article_id,
                    "title": article_data.get("title", "N/A"),
                    "url": article_data.get("url", "#"),
                    "description": article_data.get("description", ""),
                    "source_type": f"kg_derived ({article_data.get('source_type', 'N/A')})", # Indicate it's from KG
                    "retrieval_source": "KG"
                }
                kg_docs.append(Document(page_content=article_full_content, metadata=doc_metadata))
                print(f"DEBUG: Created KG document for article_id {article_id} with content length {len(article_full_content)}")

            except Exception as e_weaviate:
                print(f"DEBUG: Error fetching/processing Weaviate chunks for article_id {article_id}: {e_weaviate}")
                # Continue to next article from Neo4j

        st.session_state.retrieved_docs_kg = kg_docs # Store for potential separate display
        return kg_docs

    except Exception as e_neo4j:
        print(f"DEBUG: Error during KG retrieval from Neo4j: {e_neo4j}")
        st.error(f"An error occurred during KG-based document retrieval: {e_neo4j}")
        import traceback
        st.code(traceback.format_exc())
        return []


# --- Streamlit App Layout and Logic ---

def build_streamlit_ui(config: dict) -> tuple[str | None, List[str], str, bool, str]:
    """Builds the Streamlit UI and gets user input.

    Args:
        config: The application configuration dictionary containing roles and interests.

    Returns:
        tuple[str | None, List[str], str, bool, str]: A tuple containing the selected role,
                                                 list of selected interests,
                                                 selected generation method,
                                                 whether the generate button was clicked,
                                                 and the selected LLM model name.
    """
    st.title("📰 Personalized Briefing Generator")

    # Sidebar for method and LLM selection
    st.sidebar.title("Configuration")

    selected_method = st.sidebar.radio(
        "Choose your briefing generation method:",
        ("RAG-based", "KG-based", "Hybrid (RAG+KG)"),
        key="generation_method",
        index=0,     # Default to RAG-based
        help="RAG-based: Only uses Weaviate for document retrieval. KG-based: Only uses Neo4j for document retrieval. Hybrid (RAG+KG): Combines both methods."
    )

    # LLM Model Selection
    llm_models = [
        "gpt-4o-mini",
        "gpt-4.1-2025-04-14",
        "gpt-4.1-mini-2025-04-14",
        "gpt-4.1-nano-2025-04-14"
    ]
    selected_llm_model = st.sidebar.selectbox(
        "Select LLM Model:",
        llm_models,
        index=llm_models.index("gpt-4o-mini") # Default to gpt-4o-mini
    )

    # Role Selection
    roles = list(config["roles_and_interests"].keys())
    selected_role = st.selectbox("Select your Role", roles)

    # Interests Selection based on Role
    selected_interests = []
    if selected_role:
        available_interests = config["roles_and_interests"][selected_role]
        selected_interests = st.multiselect(
            "Select your Interests:",
            available_interests,
            default=available_interests[0] # Default to the first interest for the selected role
        )

    # Generate Briefing Button
    generate_button_clicked = st.button("Generate Briefing")

    return selected_role, selected_interests, selected_method, generate_button_clicked, selected_llm_model

def generate_and_display_briefing(
    selected_role: str,
    selected_interests: List[str],
    selected_method: str,
    synthesis_chain: RunnablePassthrough,
    weaviate_retriever_callable: callable, # For RAG part
    neo4j_driver: neo4j.GraphDatabase.driver, # For KG part
    weaviate_client: weaviate.WeaviateClient, # For KG part (enrichment)
    config: dict # For Neo4j session and other configs
):
    """Generates and displays the briefing using the selected method and RAG chain."""
    if not selected_role or not selected_interests:
        st.warning("Please select a Role and at least one Interest.")
        return

    # Initialize session state for retrieved docs if it doesn't exist
    if 'retrieved_docs' not in st.session_state:
        st.session_state.retrieved_docs = []
    if 'retrieved_docs_rag' not in st.session_state: # For specific debug
        st.session_state.retrieved_docs_rag = []
    if 'retrieved_docs_kg' not in st.session_state:
        st.session_state.retrieved_docs_kg = []


    with st.spinner(f"Generating your personalized briefing using {selected_method} method..."):
        query_for_rag = f"News and information for a {selected_role} interested in {', '.join(selected_interests)}."
        
        context_docs: List[Document] = []

        if selected_method == "RAG-based":
            context_docs = get_rag_documents(query_for_rag, weaviate_retriever_callable)
        elif selected_method == "KG-based":
            if neo4j_driver:
                context_docs = get_kg_documents(selected_interests, neo4j_driver, weaviate_client, config)
            else:
                st.warning("Neo4j connection not available. KG-based method cannot proceed.")
                context_docs = []
        elif selected_method == "Hybrid (RAG+KG)":
            rag_docs = get_rag_documents(query_for_rag, weaviate_retriever_callable)
            kg_docs_list: List[Document] = []
            if neo4j_driver:
                kg_docs_list = get_kg_documents(selected_interests, neo4j_driver, weaviate_client, config)
            
            # Combine and simple de-duplication by content
            combined_docs = rag_docs + kg_docs_list
            seen_contents = set()
            unique_docs = []
            for doc in combined_docs:
                # Check for non-empty page_content before adding to seen_contents
                if doc.page_content and doc.page_content.strip():
                    if doc.page_content not in seen_contents:
                        unique_docs.append(doc)
                        seen_contents.add(doc.page_content)
                elif not doc.page_content or not doc.page_content.strip(): # Retain docs with empty content if they have valuable metadata (though less likely for synthesis)
                    unique_docs.append(doc) # Or decide to filter them out

            context_docs = unique_docs
            print(f"DEBUG: Hybrid mode: RAG docs: {len(rag_docs)}, KG docs: {len(kg_docs_list)}, Unique combined: {len(context_docs)}")

        st.session_state.retrieved_docs = context_docs # For combined debug display

        # Prepare formatted context for the LLM prompt
        formatted_context_parts = []
        if context_docs:
            for i, doc in enumerate(context_docs):
                if doc.page_content and doc.page_content.strip():
                    title = doc.metadata.get("title", "Unknown Source")
                    url = doc.metadata.get("url", "#")
                    article_id_val = doc.metadata.get("article_id", f"doc-{i+1}")
                    
                    # Ensure URL is valid, otherwise LLM might struggle with markdown
                    if not (isinstance(url, str) and (url.startswith("http://") or url.startswith("https://"))):
                        url = "#" # Fallback for invalid URLs
                    
                    doc_text = f"Source Article ID: {article_id_val}\n"
                    doc_text += f"Source Title: {title}\n"
                    doc_text += f"Source URL: {url}\n"
                    doc_text += f"Content:\n{doc.page_content}\n---" # Added newline before content for clarity
                    formatted_context_parts.append(doc_text)
        
        final_formatted_context = "\n\n".join(formatted_context_parts)
        if not final_formatted_context:
            final_formatted_context = "No relevant information found in the knowledge sources for your query."
            st.info("No relevant information could be retrieved to generate the briefing.")
            # return # Optionally return if no context

        # Prepare input for the synthesis chain
        chain_input = {
            "role": selected_role,
            "interests": ", ".join(selected_interests), # Pass interests as a string
            "query": query_for_rag, # Thematic query for LLM's focus
            "context": final_formatted_context
        }
        
        # Display retrieved documents in an expander
        with st.expander("View Retrieved Chunks Documents"):
            docs_to_display = st.session_state.get('retrieved_docs', [])
            if docs_to_display:
                st.markdown(f"**Displaying {len(docs_to_display)} documents used for context for method: {selected_method}**")
                for i, doc in enumerate(docs_to_display):
                    st.markdown(f"#### Document {i+1} (Source: {doc.metadata.get('retrieval_source', 'RAG' if doc.metadata.get('distance') is not None else 'Unknown')})")
                    content_to_display = doc.page_content if doc.page_content is not None else ""
                    st.text_area(
                        f"Content (Article ID: {doc.metadata.get('article_id', 'N/A')}, Title: {doc.metadata.get('title', 'N/A')})", 
                        value=content_to_display, 
                        height=200,
                        key=f"doc_content_{i}_{selected_method}" # Unique key for widget
                    )
                    # Display all metadata for clarity
                    st.json(doc.metadata)
                    st.markdown("---")
            else:
                st.info("No documents were retrieved or processed for the briefing.")
        
        try:
            briefing = synthesis_chain.invoke(chain_input)
            # st.markdown("### Your Personalized Briefing") # Using markdown for subheader
            st.markdown(briefing) # Display briefing using markdown
            


        except Exception as e:
            st.error(f"An error occurred during briefing generation: {e}")
            st.info("Please ensure your Weaviate instance is running and accessible, and your API keys are correct.")
            # In a real app, you might add more specific error handling or logging
            import traceback
            st.code(traceback.format_exc())

# --- Main Application Flow ---

def main():
    """Main function to run the Streamlit application."""
    
    st.set_page_config(layout="wide", page_title="Personalized Briefing Generator") # Added page title
    
    # Initialize session state for retrieved docs and debug info
    if 'retrieved_docs' not in st.session_state:
        st.session_state.retrieved_docs = []
    if 'retrieved_docs_rag' not in st.session_state:
        st.session_state.retrieved_docs_rag = []
    if 'retrieved_docs_kg' not in st.session_state:
        st.session_state.retrieved_docs_kg = []


    # Load configuration
    config = load_configuration()
    
    # Setup Weaviate Client and Retriever (cached)
    weaviate_client = get_weaviate_client(config)
    # The 'retriever' from get_retriever is the custom_weaviate_retriever function itself
    weaviate_retriever_callable = get_retriever(weaviate_client, config) if weaviate_client else None


    # Setup Neo4j Driver (cached)
    neo4j_driver = get_neo4j_driver(config)

    # Build UI and get user input (includes selected_method and selected_llm_model)
    selected_role, selected_interests, selected_method, generate_button_clicked, selected_llm_model = build_streamlit_ui(config)

    # Setup LLM, Prompt Template, Output Parser, and Synthesis Chain
    # Components are defined here to capture types before chaining
    llm = ChatOpenAI(model=selected_llm_model, temperature=0, openai_api_key=config["openai_api_key"])

    # Define the RAG prompt template
    # This template takes 'context' (retrieved documents), 'role', 'interests', and 'query' as input
    rag_prompt_template = ChatPromptTemplate.from_messages([
        ("system", """You are a helpful AI assistant specializing in creating personalized weekly news briefings. Your task is to synthesize and summarize the key news and information from the provided context, focusing on topics relevant to the user's role and interests, to create an engaging and informative briefing.

The context below contains several documents. Each document entry provides 'Source Title', 'Source URL', and 'Content'.
Ensure *all* information in the briefing is strictly derived *only* from the provided context. Draw information from *multiple* sources within the context where available and relevant to provide a comprehensive summary.

Follow these guidelines:

Tone: Adopt an engaging and informative tone suitable for a weekly briefing.
Formatting: Use standard Markdown syntax (e.g., **bold**, *italics*, ### sub-header) for any formatting within the briefing, ensuring compatibility with Streamlit's Markdown rendering.
Structure:
1.  Start with a brief introductory sentence setting the stage for the briefing.
2.  Present the summarized news, integrating information from different sources where appropriate.
3.  Cite each piece of information with a standard Markdown link [Number](URL) (e.g., [1](http://example.com/article1)) after the relevant sentence or paragraph. The number should correspond to the source's position in the final 'References' list.
4.  A summary highlight of the briefing should be included at the end of the briefing.
5.  Conclude with a "References" section listing the full details of each source cited.

Citation and References:
-   Use simple numerical citations like [1], [2], etc., as clickable Markdown links ([Number](URL)) within the text.
-   Create a section at the very end titled "References".
-   In the "References" section, list each unique source cited in the briefing. For each source, use the format: [Number] [Source Title](URL), using Markdown syntax compatible with Streamlit. Use the 'Source Title' and 'Source URL' provided in the context for each document. Ensure the numbers in the text match the numbers in the reference list.
-   Do NOT use HTML <a> tags or plain text URLs; use only Streamlit-compatible Markdown link syntax ([text](URL)).
-   Do NOT include 'Source Article ID' in the output or citations.

Recency: Only use information from the context that is dated 06/04/25 or later.

Context:
{context}"""
        ),
        ("human", """Generate a personalized weekly news briefing for a user who is a {role} with the following interests: {interests}.
Query: {query}""")
    ])

    output_parser = StrOutputParser()

    # Create the synthesis chain
    synthesis_chain = get_synthesis_chain(llm, rag_prompt_template)

    # Generate and display briefing if button is clicked
    if generate_button_clicked:
        if not weaviate_retriever_callable and (selected_method == "RAG-based" or selected_method == "Hybrid (RAG+KG)"):
            st.error("Weaviate retriever is not available. RAG-based and Hybrid methods cannot proceed.")
        elif not neo4j_driver and (selected_method == "KG-based" or selected_method == "Hybrid (RAG+KG)"):
            st.warning("Neo4j driver not available. KG-based and Hybrid methods might be affected or unavailable.")
            # Allow to proceed if RAG part of Hybrid can still work, or if user chose RAG-only.
            if selected_method == "KG-based":
                 st.error("Cannot proceed with KG-based method due to Neo4j connection issue.")
                 return # Stop if KG-only and no Neo4j
        
        generate_and_display_briefing(
            selected_role,
            selected_interests,
            selected_method,
            synthesis_chain,
            weaviate_retriever_callable,
            neo4j_driver,
            weaviate_client, # Pass weaviate_client for KG enrichment
            config # Pass config for Neo4j database name etc.
        )
        
    # Neo4j driver is managed by @st.cache_resource, explicit closing is tricky with Streamlit's lifecycle.
    # Weaviate client is also managed by @st.cache_resource.
    # if neo4j_driver:
    #     neo4j_driver.close() # This might be called too often or at wrong times in Streamlit
    #     print("Neo4j driver closed.")


if __name__ == "__main__":
    main()