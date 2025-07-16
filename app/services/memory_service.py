# coldBotv2/app/services/memory_service.py

from collections import deque

class MemoryService:
    _instance = None
    # Speichert Konversationen im Format {conversation_id: deque([...])}
    conversations = {} 
    MAX_HISTORY_LENGTH = 10 # Speichert die letzten 5 Runden (User + Bot)

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(MemoryService, cls).__new__(cls)
        return cls._instance

    def get_history(self, conversation_id: str) -> list:
        """Holt die bisherige Konversationshistorie."""
        return list(self.conversations.get(conversation_id, []))

    def add_to_history(self, conversation_id: str, user_message: dict, bot_message: dict):
        """Fügt einen neuen Austausch zur Historie hinzu."""
        if conversation_id not in self.conversations:
            # Benutze eine deque für effizientes Anhängen und Begrenzen der Länge
            self.conversations[conversation_id] = deque(maxlen=self.MAX_HISTORY_LENGTH)
        
        self.conversations[conversation_id].append(user_message)
        self.conversations[conversation_id].append(bot_message)
        print(f"History for {conversation_id} updated. Length: {len(self.conversations[conversation_id])}")

# Globale Instanz
memory_service = MemoryService()
