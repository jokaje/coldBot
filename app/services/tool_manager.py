# coldBotv2/app/services/tool_manager.py

import json
from datetime import datetime
from duckduckgo_search import DDGS
from .rag_service import rag_service

# --- Werkzeug-Definitionen ---

def get_current_time():
    """Gibt das aktuelle Datum und die Uhrzeit als String zurück."""
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

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

def web_search(query: str):
    """
    Durchsucht das Internet mit DuckDuckGo nach einer Anfrage und gibt eine Zusammenfassung der Top-3-Ergebnisse zurück.
    """
    print(f"--- Führe Websuche für '{query}' aus ---")
    try:
        with DDGS() as ddgs:
            results = [r for r in ddgs.text(query, max_results=3)]
            if not results:
                return "Die Websuche ergab keine Ergebnisse."
            
            summary = "Zusammenfassung der Websuche:\n"
            for result in results:
                summary += f"- Titel: {result.get('title', 'N/A')}\n  Snippet: {result.get('body', 'N/A')}\n\n"
            return summary
    except Exception as e:
        print(f"Fehler bei der Websuche: {e}")
        return f"Ein Fehler ist bei der Websuche aufgetreten: {e}"


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
                    }
                },
                "required": ["fact"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "web_search",
            "description": "Durchsucht das Internet nach einer Anfrage, um neue Informationen zu einem Thema zu finden.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Die Suchanfrage, z.B. 'Wie entstehen Sterne?'",
                    }
                },
                "required": ["query"],
            },
        },
    }
]

TOOL_MAPPING = {
    "get_current_time": get_current_time,
    "save_fact_to_memory": save_fact_to_memory,
    "web_search": web_search,
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
