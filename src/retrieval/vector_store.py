import chromadb
from chromadb.config import Settings

class VectorStore:
    def __init__(self, persist_directory: str):
        self.client = chromadb.Client(Settings(
            persist_directory=persist_directory,
            anonymized_telemetry=False
        ))
        
        # Create collections
        self.collections = {
            "novels": self.client.create_collection("novels"),
            "dialogues": self.client.create_collection("dialogues"),
            "profiles": self.client.create_collection("profiles"),
            "wiki": self.client.create_collection("wiki")
        }
    
    def add_documents(self, collection_name: str, documents: list, 
                     metadatas: list = None, ids: list = None):
        """Add documents to specified collection"""
        collection = self.collections[collection_name]
        collection.add(
            documents=documents,
            metadatas=metadatas,
            ids=ids
        )
    
    def search(self, collection_name: str, query: str, n_results: int = 5):
        """Search for similar documents in collection"""
        collection = self.collections[collection_name]
        results = collection.query(
            query_texts=[query],
            n_results=n_results
        )
        return results
