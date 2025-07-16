# coldBotv2/app/services/rag_service.py

import chromadb
from sentence_transformers import SentenceTransformer
import uuid

class RAGService:
    _instance = None
    model = None
    client = None
    collection = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(RAGService, cls).__new__(cls)
            
            print("Initializing RAG Service...")
            # Lade ein Modell, das Text in Vektoren umwandeln kann.
            # 'all-MiniLM-L6-v2' ist klein, schnell und gut.
            cls.model = SentenceTransformer('all-MiniLM-L6-v2')
            
            # Initialisiere die lokale Vektordatenbank
            # Wir speichern die Daten auf der Festplatte im Ordner 'chroma_db'
            cls.client = chromadb.PersistentClient(path="./chroma_db")
            
            # Erstelle oder lade eine "Collection" (wie eine Tabelle in einer DB)
            cls.collection = cls.client.get_or_create_collection(name="coldbot_knowledge")
            print("RAG Service initialized successfully.")
            
        return cls._instance

    def add_text(self, text: str, source_name: str):
        """
        Nimmt einen Text, zerlegt ihn in Stücke, erstellt Vektoren und speichert sie.
        """
        # Einfache Methode, um Text in Absätze zu zerlegen.
        chunks = [chunk for chunk in text.split('\n') if chunk.strip()]
        if not chunks:
            return

        # Erstelle die Vektor-Einbettungen für alle Chunks
        embeddings = self.model.encode(chunks).tolist()
        
        # Erstelle Metadaten und eindeutige IDs für jeden Chunk
        metadatas = [{"source": source_name} for _ in chunks]
        ids = [str(uuid.uuid4()) for _ in chunks]

        # Füge die Daten zur Datenbank hinzu
        self.collection.add(
            embeddings=embeddings,
            documents=chunks,
            metadatas=metadatas,
            ids=ids
        )
        print(f"Added {len(chunks)} chunks from '{source_name}' to the knowledge base.")

    def search(self, query_text: str, n_results: int = 3) -> str:
        """
        Durchsucht die Wissensdatenbank nach den relevantesten Informationen.
        """
        if not query_text:
            return ""
            
        # Erstelle einen Vektor für die Suchanfrage
        query_embedding = self.model.encode([query_text]).tolist()
        
        # Führe die Suche in der Datenbank durch
        results = self.collection.query(
            query_embeddings=query_embedding,
            n_results=n_results
        )
        
        # Formatiere die Ergebnisse zu einem einzigen Kontext-String
        context = "\n".join(doc for doc in results['documents'][0])
        print(f"Found context: {context}")
        return context

# Globale Instanz
rag_service = RAGService()
