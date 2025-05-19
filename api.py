from pydantic import BaseModel, Field
from typing import Literal


class Article(BaseModel):
    article_id: str = Field(description="The id of the article")
    title: str = Field(description="The title of the article")
    description: str = Field(description="The description of the article")
    url: str = Field(description="The URL of the article")
    source_type: str = Field(description="The type of the source of the article")

PredefinedTopics = Literal[
    "Healthcare reform", 
    "Public health data", 
    "Telemedicine", 
    "Pharmaceutical innovation", 
    "Health equity", 
    "Government policy updates",
    "Artificial intelligence", 
    "Venture capital", 
    "Startup funding", 
    "SaaS trends", 
    "Product management", 
    "Others"
]

class Topic(BaseModel):
    topic_id: str = Field(description="The id of the topic")
    article_ids: list[str] = Field(description="The ids of the articles that discuss the topic")
    topic_name: PredefinedTopics = Field(description="The name of the topic")


class TopicList(BaseModel):
    topics: list[Topic] = Field(description="A list of topics")