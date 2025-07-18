# coldBotv2/app/services/rag_service.py

import chromadb
from sentence_transformers import SentenceTransformer
import uuid
from typing import List, Dict, Any

class RAGService:
    _instance = None
    model = None
    client = None
    collection = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(RAGService, cls).__new__(cls)
            
            print("Initializing RAG Service...")
            cls.model = SentenceTransformer('all-MiniLM-L6-v2')
            cls.client = chromadb.PersistentClient(path="./chroma_db")
            cls.collection = cls.client.get_or_create_collection(name="coldbot_knowledge_v2")
            print("RAG Service initialized successfully.")
            
        return cls._instance

    def _build_where_clause(self, filter_metadata: Dict[str, Any]) -> Dict[str, Any]:
        """
        Baut eine gültige 'where'-Klausel für ChromaDB, auch für mehrere Bedingungen.
        """
        if not filter_metadata:
            return {}
        
        if len(filter_metadata) == 1:
            return filter_metadata
        
        # Wenn mehr als ein Filterkriterium vorhanden ist, muss die $and-Syntax verwendet werden.
        and_conditions = []
        for key, value in filter_metadata.items():
            and_conditions.append({key: value})
        
        return {"$and": and_conditions}

    def add_texts(self, texts: List[str], metadatas: List[Dict[str, Any]]):
        """
        Nimmt eine Liste von Texten, erstellt Vektoren und speichert sie mit den zugehörigen Metadaten.
        """
        if not texts:
            return

        embeddings = self.model.encode(texts).tolist()
        ids = [str(uuid.uuid4()) for _ in texts]

        self.collection.add(
            embeddings=embeddings,
            documents=texts,
            metadatas=metadatas,
            ids=ids
        )
        source_types = {meta.get('source', 'unknown') for meta in metadatas}
        print(f"Added {len(texts)} chunks with source types {source_types} to the knowledge base.")

    def search(self, query_text: str, n_results: int = 3, filter_metadata: Dict[str, Any] = None) -> List[Dict[str, Any]]:
        """
        Durchsucht die Wissensdatenbank und gibt die vollstaendigen Ergebnisse (Dokumente und Metadaten) zurueck.
        Kann optional nach Metadaten filtern.
        """
        if not query_text:
            return []
            
        query_embedding = self.model.encode([query_text]).tolist()
        
        # --- KORREKTUR: Benutze die neue Helper-Funktion ---
        where_clause = self._build_where_clause(filter_metadata)
        
        results = self.collection.query(
            query_embeddings=query_embedding,
            n_results=n_results,
            where=where_clause
        )
        
        combined_results = []
        if results and results['documents']:
            for i, doc in enumerate(results['documents'][0]):
                meta = results['metadatas'][0][i]
                combined_results.append({"document": doc, "metadata": meta})

        print(f"Found {len(combined_results)} results for query '{query_text}' with filter {where_clause}")
        return combined_results

    def get_all_by_meta(self, filter_metadata: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Ruft alle Einträge ab, die den Filterkriterien entsprechen.
        """
        # --- KORREKTUR: Benutze die neue Helper-Funktion ---
        where_clause = self._build_where_clause(filter_metadata)
        
        results = self.collection.get(where=where_clause)
        
        combined_results = []
        if results and results['documents']:
            for i, doc in enumerate(results['documents']):
                meta = results['metadatas'][i]
                combined_results.append({"document": doc, "metadata": meta})
        
        print(f"Retrieved {len(combined_results)} total entries with filter {where_clause}")
        return combined_results

# Globale Instanz
rag_service = RAGService()
