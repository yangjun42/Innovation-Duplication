from langchain_core import prompts

from langchain_openai import AzureChatOpenAI, AzureOpenAIEmbeddings

from langchain_community.vectorstores import AzureSearch

from tenacity import retry, wait_exponential, stop_after_attempt, retry_if_exception_type
from openai import RateLimitError

import faiss

import numpy as np

from pathlib import Path
import json
import pickle

from typing import List,Dict

from qdrant import VectorStore

CONFIG_PATH = Path(__file__).resolve().parent / "data" / "keys" / "azure_config.json"

DEFAULT_DIMS = 3072
DEFAULT_INDEX_PATH = Path(__file__).resolve().parent / "vector_store"

def get_openai_models(dimensions = DEFAULT_DIMS, temperature = 0 ):
    """
    @return Instance of AzureChatOpenAI.
            Instance of AzureOpenAIEmbeddings.
    """

    config_path  = Path(CONFIG_PATH)

    try:
        with config_path.open("r") as f:
            config = json.load(f)

    except FileNotFoundError:
        print("ERROR: config.json not found.")

    azure_config = config.get("azure_openai", {})
    azure_search_config = config.get("azure_search", {})
    

    llm = AzureChatOpenAI(
        azure_endpoint = azure_config["azure_endpoint"],
        api_key = azure_config["api_key"],
        api_version = azure_config["api_version"],

        azure_deployment = azure_config["chat_deployment"]["gpt-4.1"],

        temperature = temperature
    )

    embedding_model = AzureOpenAIEmbeddings(
        azure_endpoint = azure_config["azure_endpoint"],
        api_key = azure_config["api_key"],
        api_version = azure_config["api_version"],

        azure_deployment = azure_config["embedding_deployment"]["large"],
        dimensions = dimensions
    )


    search_endpoint = azure_search_config.get("azure_search_endpoint", "")
    search_key = azure_search_config.get("azure_search_key", "")
    index_name = azure_search_config.get("index_name", "")

    if search_endpoint and search_key and index_name:
        print("Using Azure AI Search as vector store.")
        vector_store = AzureSearch(
            azure_search_endpoint = search_endpoint,
            azure_search_key = search_key,
            index_name = index_name,
            embedding_model = embedding_model
        )
        using_local_store = False
    else:
        print("Azure AI Search config missing, using local vector store.")
        vector_store = VectorStore(embedding_model = embedding_model)

        using_local_store = True



    return llm, embedding_model, vector_store, using_local_store 





