"""
This script is a one-off setup utility to construct an initial knowledge graph from a JSON file and upload it to Neo4j.

The process involves:
1. Loading a JSON file containing article data.
2. Parsing and cleaning the text to create Article objects.
3. Using an LLM-based topic classifier to categorize articles into predefined topics.
4. Uploading the articles and topics to Neo4j, creating nodes and relationships.

Dependencies:
- json, os, BeautifulSoup, Path, langchain_core, langchain_openai, asyncio, neo4j, dotenv
- api module containing Article and TopicList classes

Usage:
    Ensure the JSON file 'sources_06-05-25.json' is present in the same directory.
    Set environment variables for Neo4j connection (NEO4J_URI, NEO4J_PASSWORD).
    Run the script to process the data and upload to Neo4j.
"""

import json
import os
from bs4 import BeautifulSoup
from pathlib import Path
# from langchain_core.documents import Document
from langchain_core.runnables import RunnablePassthrough, Runnable
from langchain_core.prompts import SystemMessagePromptTemplate, HumanMessagePromptTemplate, ChatPromptTemplate
from langchain_openai import ChatOpenAI
# from langchain_experimental.graph_transformers import LLMGraphTransformer
import asyncio
from neo4j import GraphDatabase
from dotenv import load_dotenv
load_dotenv()

from api import Article, TopicList


# --- 1. Load JSON ---
with open("sources_06-05-25.json", "r", encoding="utf-8") as f:
    data = json.load(f)

# --- 2. Parse and clean text ---
documents = []

all_articles = []
for index, (doc_key, doc) in enumerate(data['reference_by_id'].items()): 
    article = Article(
        article_id=f'A{index+1}',
        title=doc.get("title", ""),
        description=doc.get("description", ""),
        url=doc.get("url", ""),
        source_type=doc.get("source_type", "")
    )
    all_articles.append(article)

topic_classifier = ChatOpenAI(temperature=0, model_name="gpt-4.1-mini-2025-04-14").with_structured_output(TopicList)

topic_classifier_system_prompt: SystemMessagePromptTemplate = SystemMessagePromptTemplate.from_template(
    """You are an AI assistant. You will be provided with a list of Article objects in JSON format via the variable `articles`. 
    Your task is to analyze these articles and classify them into topics.

    For each topic, create a Topic object with the following fields (as defined in the Topic model):
    - topic_id: a unique string identifier for the topic (e.g. "T1", "T2", etc.). Each topic should have SINGLE and IDENTICAL topic_id.
    - topic_name: a string that describes this topic. 
    - article_ids: a list of article_id strings for all articles that discuss and highly related to this topic. 

    Note:
    - if you think an article is not related to any topic, you can put it in a topic called "Others".
    - if you think an article is related to multiple topics, you can put it in multiple Topic objects' article_id lists.

    Collect all Topic objects into a list and return them as a TopicList object, strictly following the TopicList schema from the API.
    """
)

topic_classifier_human_prompt: HumanMessagePromptTemplate = HumanMessagePromptTemplate.from_template(
    """Analyse and classify the following articles into topics.
    
    Articles:
    {articles}
    """
)

prompt = ChatPromptTemplate.from_messages([topic_classifier_system_prompt, topic_classifier_human_prompt])

chain: Runnable = (
    {"articles": RunnablePassthrough()}
    | prompt
    | topic_classifier
)

response = chain.invoke(all_articles)

print(response.model_dump_json(indent=2))


# --- 3. Upload to Neo4j ---
def upload_to_neo4j(topic_list):
    # Neo4j connection details
    uri = os.getenv("NEO4J_URI")
    user = "neo4j"
    password = os.getenv("NEO4J_PASSWORD")
    driver = GraphDatabase.driver(uri, auth=(user, password))

    with driver.session() as session:
        session.run("MATCH (n) DETACH DELETE n")

        # Create Article nodes
        for article in all_articles:
            session.run("""
                MERGE (a:Article {article_id: $article_id})
                ON CREATE SET a.title = $title,
                              a.description = $description,
                              a.url = $url,
                              a.source_type = $source_type
                ON MATCH SET a.title = $title,
                             a.description = $description,
                             a.url = $url,
                             a.source_type = $source_type
            """,
            article_id=article.article_id,
            title=article.title,
            description=article.description,
            url=article.url,
            source_type=article.source_type
        )

        # Create Topic nodes
        for topic in topic_list.topics:
            session.run("""
                    MERGE (t:Topic {topic_id: $topic_id})
                    ON CREATE SET t.article_ids = $article_ids,
                                t.topic_name = $topic_name
                    ON MATCH SET t.article_ids = $article_ids,
                                t.topic_name = $topic_name
                """,
                topic_id=topic.topic_id,
                article_ids=topic.article_ids,
                topic_name=topic.topic_name
            )

        # Create HAS_TOPIC relationships between Article and Topic nodes
        for topic in topic_list.topics:
            for article_id in topic.article_ids:
                session.run("""
                    MATCH (a:Article {article_id: $article_id})
                    MATCH (t:Topic {topic_id: $topic_id})
                    MERGE (a)-[:HAS_TOPIC]->(t)
                """,
                article_id=article_id,
                topic_id=topic.topic_id
            )
    
    driver.close()

# Call the upload function after getting the response
upload_to_neo4j(response)

