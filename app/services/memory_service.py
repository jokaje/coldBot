# coldBotv2/app/services/memory_service.py

from collections import deque
from typing import Dict

class MemoryService:
    _instance = None
    # Speichert Konversationen im Format {conversation_id: deque([...])}
    conversations: Dict[str, deque] = {} 
    MAX_HISTORY_LENGTH = 20 # Erhöht, um mehr Runden (inkl. Tool-Nutzung) zu speichern

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(MemoryService, cls).__new__(cls)
        return cls._instance

    def get_history(self, conversation_id: str) -> list:
        """Holt die bisherige Konversationshistorie."""
        return list(self.conversations.get(conversation_id, []))

    # --- KORREKTUR: Die Funktion akzeptiert jetzt eine einzelne Nachricht ---
    def add_to_history(self, conversation_id: str, message: dict):
        """Fügt eine einzelne Nachricht (von user, assistant oder tool) zur Historie hinzu."""
        if conversation_id not in self.conversations:
            # Benutze eine deque für effizientes Anhängen und Begrenzen der Länge
            self.conversations[conversation_id] = deque(maxlen=self.MAX_HISTORY_LENGTH)
        
        self.conversations[conversation_id].append(message)
        print(f"History for {conversation_id} updated. Role '{message['role']}' added. Total length: {len(self.conversations[conversation_id])}")

# Globale Instanz
memory_service = MemoryService()
