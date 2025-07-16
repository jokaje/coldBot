# coldBotv2/app/services/tool_manager.py

import json
from datetime import datetime

# --- Werkzeug-Definitionen ---

def get_current_time():
    """Gibt das aktuelle Datum und die Uhrzeit als String zurück."""
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

# --- Werkzeug-Verwaltung ---

# Eine Liste aller verfügbaren Werkzeuge mit Beschreibungen für die KI.
# Das Format (JSON Schema) ist entscheidend, damit das LLM die Werkzeuge versteht.
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
    }
]

# Ein Dictionary, das den Namen eines Werkzeugs auf die auszuführende Python-Funktion abbildet.
TOOL_MAPPING = {
    "get_current_time": get_current_time,
}

class ToolManager:
    def get_tools_for_llm(self) -> str:
        """Gibt die Werkzeug-Definitionen als JSON-String zurück."""
        return json.dumps(AVAILABLE_TOOLS)

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
