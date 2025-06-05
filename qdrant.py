from qdrant_client import QdrantClient, models
import qdrant_client.http.exceptions as qdrant_exceptions

from langchain_community.vectorstores import Qdrant

from typing import List
import uuid


class VectorStore:
    def __init__(self, embedding_model, collection_name="innovations_knowledge", url="http://localhost:6333"):
        self.embedding_model = embedding_model
        self.dimensions = self.embedding_model.dimensions

        self.collection_name = collection_name

        self.qdrant = QdrantClient(url = url)

        self.init_collection()

    

    def init_collection(self):
        # Initialize collection if not exists.
        if not self.qdrant.collection_exists(collection_name = self.collection_name):

            self.qdrant.create_collection(
                collection_name = self.collection_name,
                vectors_config = models.VectorParams(size = self.dimensions, \
                                                     distance = models.Distance.COSINE),
                on_disk_payload = True
            )
        
        self.qdrant.create_payload_index(
            collection_name = self.collection_name,
            field_name = "text",
            field_schema = models.PayloadSchemaType.TEXT
        )


    @staticmethod
    def generate_hash_id(text: str) -> str:
        return str(uuid.uuid5(uuid.NAMESPACE_DNS, text))


    def load_cache(self, embeddings:List[List[float]], texts: List[str]):

        self.qdrant.delete_collection(collection_name = self.collection_name)
        self.init_collection()

        payload = [{"text": text} for text in texts]

        ids = [self.generate_hash_id(text) for text in texts]

        self.qdrant.upload_collection(
            collection_name = self.collection_name,
            ids = ids,
            vectors = embeddings,
            payload = payload,
            parallel = 1
        )
    
    def remove(self, texts: List[str]):

        ids = [self.generate_hash_id(text) for text in texts]
                    
        # Idempotent operation, automatically skips when id doesn't exist.

        try:
            self.qdrant.delete(
                collection_name = self.collection_name,
                points_selector = models.PointIdsList(ids)
            )
        except Exception as  e:
            print(f"[Warning][Qdrant.delete] Something may went wrong:{e}")


    def embed_and_save(self, texts: List[str]):

        # Avoid duplicate inputs.
        accepted_text_count = 0
        
        for text in texts:
            id = self.generate_hash_id(text)
            embedding = self.embedding_model.embed_query(text)

            # Check for existence by ID (lightweight)
            try:
                retrieved_points = self.qdrant.retrieve(
                    collection_name=self.collection_name,
                    ids=[id],
                    with_vectors = False,
                    with_payload = False
                )
                if retrieved_points:
                    continue
            except Exception as e:
                print(f"[Warning][Qdrant.retrieve] Something may went wrong:{e}") 

            self.qdrant.upsert(
                collection_name=self.collection_name,
                points=[
                    models.PointStruct(
                        id = id,
                        vector=embedding,
                        payload={"text": text}
                    )
                ]
            )
            accepted_text_count += 1
        

        print(f"Inserted {accepted_text_count} new vectors to Qdrant, \
                {len(texts) - accepted_text_count} duplicate texts skipped.")
        

    def search(self, queries: List[str], limit = 5):
        
        embeddings = self.embedding_model.embed_documents(queries)

        requests = [
            models.SearchRequest(
                vector = embedding,
                limit = limit,
                with_payload = True
            )
            for embedding in embeddings
        ]

        batch_results = self.qdrant.search_batch(
            collection_name = self.collection_name,
            requests = requests
        )

        result_texts = []
        result_scores = []

        for batch in batch_results:
            for result in batch:
                result_texts.append(result.payload["text"])
                result_scores.append(result.score)

        return result_scores, result_texts
    