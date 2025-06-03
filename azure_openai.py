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


# 运行时所有数据存在 Memory 中， 内存占用太高，应使用数据库代替。
class VectorStore:

    def __init__(self, embedding_model, save_path = DEFAULT_INDEX_PATH):

        self.embedding_model = embedding_model
        
        self.dimensions = self.embedding_model.dimensions

        self.file_paths = {
            "index": Path(save_path) / "index.faiss",
            "id_to_text": Path(save_path) / "id_to_text.pkl",
            "saved_texts": Path(save_path) / "saved_texts.pkl"
        }

        if self.file_paths["index"].exists():
            self.index = faiss.read_index(str(self.file_paths["index"]))

            if self.index.d != self.dimensions:
                raise ValueError(f"Load index failed, dimension doesn't fit.\
                                  Expect: {self.dimensions}, but got {self.index.d}")
        
            if self.file_paths["id_to_text"].exists():
                with self.file_paths["id_to_text"].open("rb") as f:
                    self.id_to_text = pickle.load(f)
            else:
                raise ValueError("Index ID Map doesn't exist.")
            
            self.next_id = len(self.id_to_text)


            if self.file_paths["saved_texts"].exists():
                with self.file_paths["saved_texts"].open("rb") as f:
                    self.saved_texts = pickle.load(f)
            else:
                raise ValueError("Failed to load saved texts.")

        else:
            self.index = faiss.IndexFlatIP(self.dimensions)
            self.id_to_text:Dict[int, str] = {}
            self.saved_texts = set()
            self.next_id = 0


    def save(self):
        faiss.write_index(self.index, str(self.file_paths["index"]))

        with self.file_paths["id_to_text"].open("wb") as f:
            pickle.dump(self.id_to_text, f)

        with self.file_paths["saved_texts"].open("wb") as f:
            pickle.dump(self.saved_texts, f)
    

    
    # For cache.
    def save_embeds_and_texts(self, embedding_matrixs, texts:List[str]):
        self.index.reset()

        self.index.add(embedding_matrixs)

        for text in texts:
            self.id_to_text[self.next_id] = text
            self.next_id += 1

        self.save()


    @retry(
        wait=wait_exponential(multiplier=2, min=5, max=60),
        stop=stop_after_attempt(5),
        retry=retry_if_exception_type(RateLimitError)
    )
    def safe_embed(texts: List[str], model):
        return model.embed_documents(texts)
    
    def embed_and_save(self,texts:List[str], batch_size = 10):

        # Remove Duplicate
        filtered_texts = []
        for text in texts:
            if text not in self.saved_texts:
                filtered_texts.append(text)
                self.saved_texts.add(text)

        if not filtered_texts:
            print("All texts are duplicates. No new embeddings added.")
            return
        
        embeddings = self.embedding_model.embed_documents(filtered_texts)
    
        # Batch
        for i in range(0, len(filtered_texts), batch_size):
            batch = filtered_texts[i:i + batch_size]
            try:
                embeddings = self.safe_embed(batch, self.embedding_model)
                self.index.add(embeddings)

                for text in batch:
                    self.id_to_text[self.next_id] = text
                    self.next_id += 1

                print(f"Added batch {i // batch_size + 1} with {len(batch)} texts.")
            except Exception as e:
                print(f"[Error] Failed to embed batch {i // batch_size + 1}: {e}")

        self.save()


    def search(self, texts:List[str], k = 3):

        embeddings = self.embedding_model.embed_documents(texts)

        embedding_matrix = np.array(embeddings, dtype=np.float32)

        D, I =  self.index.search(embedding_matrix, k = k)

        result_texts = []
        for i in I:
            # In case there is not result.
            match_texts = [ self.id_to_text.get(int(j),"[Not found]") for j in i]
            result_texts.append(match_texts)

        return D, result_texts




if __name__ == "__main__":
    llm, embedding_model = get_openai_models()

    query_prompt = prompts.PromptTemplate.from_template("""

        详细介绍:{name}。

    """)

    query_chain = query_prompt | llm

    answer = query_chain.invoke({"name":"Streamlit"})

    print(answer.content)


