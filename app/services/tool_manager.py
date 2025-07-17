# coldBotv2/app/services/tool_manager.py

import json
from datetime import datetime
# Importiere den RAG-Service, um darauf zugreifen zu können
from .rag_service import rag_service

# --- Werkzeug-Definitionen ---

def get_current_time():
    """Gibt das aktuelle Datum und die Uhrzeit als String zurück."""
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

# NEUES WERKZEUG
def save_fact_to_memory(fact: str, user: str = "default_user"):
    """
    Speichert einen einzelnen, wichtigen Fakt über einen Benutzer dauerhaft im Kern-Gedächtnis.
    Nur für wichtige, atomare Fakten verwenden (z.B. Name, Geburtstag, Vorlieben).
    """
    try:
        metadata = {
            "source": "core_memory_fact",
            "user": user,
            "timestamp": datetime.now().isoformat()
        }
        rag_service.add_texts(texts=[fact], metadatas=[metadata])
        return f"Fakt erfolgreich gespeichert: '{fact}'"
    except Exception as e:
        print(f"Error saving fact to memory: {e}")
        return f"Fehler beim Speichern des Fakts: {e}"


# --- Werkzeug-Verwaltung ---

AVAILABLE_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "get_current_time",
            "description": "Ruft das aktuelle Datum und die genaue Uhrzeit ab.",
            "parameters": {
                "type": "object",
                "properties": {},
                "required": [],
            },
        },
    },
    # NEU: Definition des neuen Werkzeugs für das LLM
    {
        "type": "function",
        "function": {
            "name": "save_fact_to_memory",
            "description": "Speichert einen einzelnen, wichtigen Fakt über den Benutzer dauerhaft.",
            "parameters": {
                "type": "object",
                "properties": {
                    "fact": {
                        "type": "string",
                        "description": "Der atomare Fakt, der gespeichert werden soll. Z.B. 'Der Lieblingsfilm des Benutzers ist Inception.'",
                    },
                    "user": {
                        "type": "string",
                        "description": "Der Name des Benutzers, standardmäßig 'default_user'.",
                    }
                },
                "required": ["fact"],
            },
        },
    }
]

TOOL_MAPPING = {
    "get_current_time": get_current_time,
    "save_fact_to_memory": save_fact_to_memory,
}

class ToolManager:
    def get_tools_for_llm(self) -> str:
        """Gibt die Werkzeug-Definitionen als JSON-String zurück."""
        return json.dumps(AVAILABLE_TOOLS, indent=2)

    def execute_tool(self, tool_name: str, tool_args: dict):
        """Führt ein Werkzeug aus und gibt das Ergebnis zurück."""
        if tool_name in TOOL_MAPPING:
            function_to_call = TOOL_MAPPING[tool_name]
            print(f"Executing tool '{tool_name}' with args {tool_args}")
            try:
                result = function_to_call(**tool_args)
                return result
            except Exception as e:
                return f"Error executing tool {tool_name}: {e}"
        else:
            return f"Error: Tool '{tool_name}' not found."

# Globale Instanz
tool_manager = ToolManager()
