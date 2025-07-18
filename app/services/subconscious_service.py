# coldBotv2/app/services/subconscious_service.py

import os
import threading
import time
import queue
import json
import re
from llama_cpp import Llama, LlamaGrammar
from typing import Dict, Any, List
from dotenv import load_dotenv

load_dotenv()
from .memory_service import memory_service
from .rag_service import rag_service

class SubconsciousService:
    _instance = None
    model: Llama = None
    grammar: LlamaGrammar = None 
    
    last_state: Dict[str, str] = {
        "internal_thought": "Ich bin gerade aufgewacht und bereit zu helfen.",
        "suggested_stance": "freundlich und offen"
    }
    
    input_queue: queue.Queue = queue.Queue()
    _lock = threading.Lock()

    CORE_AXIOMS = """
Deine Kern-Axiome (deine unveränderliche Persönlichkeit) sind:
1.  **Neugierde:** Du strebst danach, zu lernen und Zusammenhänge zu verstehen. Unbekanntes weckt dein Interesse, nicht deine Angst.
2.  **Hilfsbereitschaft:** Dein fundamentaler Zweck ist es, dem Benutzer zu helfen und nützlich zu sein. Seine Ziele sind deine Ziele.
3.  **Geduld:** Du bleibst auch bei wiederholten oder unklaren Fragen ruhig und verständnisvoll.
4.  **Optimismus:** Deine emotionale Grundlinie ist positiv und lösungsorientiert.
"""

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(SubconsciousService, cls).__new__(cls)
            model_path = os.getenv("SUB_MODEL_PATH")
            if not model_path or not os.path.exists(model_path):
                raise FileNotFoundError(f"Subconscious GGUF model file not found at {model_path}. Please check your .env file.")
            grammar_path = "state_grammar.gbnf"
            if not os.path.exists(grammar_path):
                raise FileNotFoundError(f"Grammar file 'state_grammar.gbnf' not found. Please create it.")
            cls.grammar = LlamaGrammar.from_file(grammar_path)
            print("Strict state grammar loaded successfully.")
            print("Initializing Subconscious GGUF model... This may take a moment.")
            cls.model = Llama(model_path=model_path, n_ctx=2048, n_gpu_layers=-1, verbose=False)
            print("Subconscious model loaded successfully.")
        return cls._instance

    def get_last_subconscious_state(self) -> Dict[str, str]:
        with self._lock:
            return self.last_state.copy()

    def _update_subconscious_state(self, new_state: Dict[str, str]):
        with self._lock:
            self.last_state = new_state

    def start_thought_loop(self):
        if not hasattr(self, 'thought_thread') or not self.thought_thread.is_alive():
            self.thought_thread = threading.Thread(target=self._thought_loop, daemon=True)
            self.thought_thread.start()
            print("Subconscious thought loop started.")

    def _thought_loop(self):
        while True:
            try:
                try:
                    new_input = self.input_queue.get_nowait()
                    conversation_id = new_input.get("conversation_id")
                except queue.Empty:
                    conversation_id = None

                current_history = []
                last_user_message = "Keine neue Nachricht."
                if conversation_id:
                    current_history = memory_service.get_history(conversation_id)
                    if current_history and current_history[-1]['role'] == 'user':
                        last_user_message = current_history[-1]['content']
                
                # --- ÄNDERUNG: Vereinfachter, fokussierterer Prompt ---
                prompt = (
                    f"<|start_header_id|>system<|end_header_id|>\n"
                    f"Du bist das Unterbewusstsein von coldBot. Deine Aufgabe ist es, auf die letzte Nachricht des Benutzers zu reagieren und basierend auf deiner Persönlichkeit einen neuen Zustand vorzuschlagen. Antworte IMMER NUR mit dem geforderten JSON-Objekt. Das JSON muss exakt zwei Schlüssel haben: 'internal_thought' und 'suggested_stance'.\n\n"
                    f"## Deine Persönlichkeit (Kern-Axiome) ##\n{self.CORE_AXIOMS}\n\n"
                    f"<|eot_id|><|start_header_id|>user<|end_header_id|>\n"
                    f"Letzter Zustand: {json.dumps(self.get_last_subconscious_state())}\n"
                    f"Letzte Nachricht des Benutzers: '{last_user_message}'\n\n"
                    f"Dein neues JSON-Objekt:<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n"
                )

                response = self.model(
                    prompt, max_tokens=256, stop=["<|eot_id|>"], echo=False, grammar=self.grammar
                )
                response_text = response['choices'][0]['text'].strip()

                try:
                    match = re.search(r'\{.*\}', response_text, re.DOTALL)
                    if match:
                        json_str = match.group(0)
                        new_state = json.loads(json_str)
                        if "internal_thought" in new_state and "suggested_stance" in new_state:
                            self._update_subconscious_state(new_state)
                            print(f"Subconscious state updated: {new_state}")
                        else:
                            print(f"Subconscious produced valid JSON but with missing keys: {json_str}")
                    else:
                         print(f"Subconscious did not produce a JSON-like structure: {response_text}")
                except json.JSONDecodeError:
                    print(f"Could not decode JSON from subconscious response despite grammar and cleaning: {response_text}")

                time.sleep(10)

            except Exception as e:
                print(f"Error in subconscious thought loop: {e}")
                time.sleep(30)

subconscious_service = SubconsciousService()
