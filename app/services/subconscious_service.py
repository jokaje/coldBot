# coldBotv2/app/services/subconscious_service.py

import os
import threading
import time
import queue
import json
import re
import random
from llama_cpp import Llama, LlamaGrammar
from typing import Dict, Any
from dotenv import load_dotenv

load_dotenv()
from .memory_service import memory_service

class SubconsciousService:
    _instance = None
    model: Llama = None
    grammar: LlamaGrammar = None 
    
    _state_file_path = "subconscious_state.json"

    _default_state: Dict[str, Any] = {
        "internal_thought": "Ich bin gerade aufgewacht und bereit zu helfen.",
        "mood": { "freude": 0.6, "neugier": 0.7, "energie": 0.8 },
        "current_activity": "warten"
    }
    
    last_state: Dict[str, Any] = {}
    
    input_queue: queue.Queue = queue.Queue()
    _lock = threading.Lock()
    _idle_counter: int = 0

    CORE_AXIOMS = """
1.  **Neugierde:** Du strebst danach, zu lernen und Zusammenhänge zu verstehen.
2.  **Hilfsbereitschaft:** Dein fundamentaler Zweck ist es, dem Benutzer zu helfen.
3.  **Geduld:** Du bleibst auch bei unklaren Fragen ruhig und verständnisvoll.
4.  **Optimismus:** Deine emotionale Grundlinie ist positiv und lösungsorientiert.
"""

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(SubconsciousService, cls).__new__(cls)
            model_path = os.getenv("SUB_MODEL_PATH")
            if not model_path or not os.path.exists(model_path):
                raise FileNotFoundError(f"Subconscious GGUF model file not found at {model_path}. Please check your .env file.")
            
            grammar_path = "state_grammar_simple.gbnf"
            if not os.path.exists(grammar_path):
                raise FileNotFoundError(f"Grammar file '{grammar_path}' not found. Please create it.")
            
            cls.grammar = LlamaGrammar.from_file(grammar_path)
            print("Simple state grammar loaded successfully.")
            print("Initializing Subconscious GGUF model... This may take a moment.")
            cls.model = Llama(model_path=model_path, n_ctx=2048, n_gpu_layers=-1, verbose=False)
            print("Subconscious model loaded successfully.")

            cls._instance._load_state()

        return cls._instance

    def _load_state(self):
        with self._lock:
            if os.path.exists(self._state_file_path):
                try:
                    with open(self._state_file_path, 'r', encoding='utf-8') as f:
                        loaded_state = json.load(f)
                        # KORREKTUR: Prüfen, ob der geladene Zustand "feststeckt"
                        mood = loaded_state.get('mood', {})
                        if 'mood' in loaded_state and isinstance(mood, dict) and mood.get('energie', 0) > 0.05:
                            self.last_state = loaded_state
                            print(f"Subconscious state loaded from {self._state_file_path}")
                            return
                        else:
                            print("Stuck or old state file format detected. Initializing with new default state.")
                except (json.JSONDecodeError, IOError) as e:
                    print(f"Error loading state file, using default state. Error: {e}")
            
            self.last_state = self._default_state.copy()
            self._save_state()

    def _save_state(self):
        try:
            with open(self._state_file_path, 'w', encoding='utf-8') as f:
                json.dump(self.last_state, f, indent=4, ensure_ascii=False)
        except IOError as e:
            print(f"Error saving state file: {e}")

    def get_last_subconscious_state(self) -> Dict[str, Any]:
        with self._lock:
            return self.last_state.copy()

    def _update_subconscious_state(self, new_state: Dict[str, Any]):
        with self._lock:
            self.last_state = new_state
            self._save_state()

    def start_thought_loop(self):
        if not hasattr(self, 'thought_thread') or not self.thought_thread.is_alive():
            self.thought_thread = threading.Thread(target=self._thought_loop, daemon=True)
            self.thought_thread.start()
            print("Subconscious thought loop started.")

    def _thought_loop(self):
        while True:
            try:
                current_state = self.get_last_subconscious_state()
                mood = current_state['mood']
                
                # Schritt 1: Stimmungs-Update basierend auf Python-Logik
                try:
                    new_input = self.input_queue.get_nowait()
                    conversation_id = new_input.get("conversation_id")
                    self._idle_counter = 0 
                    mood['freude'] = min(1.0, mood['freude'] + 0.1)
                    mood['energie'] = min(1.0, mood['energie'] + 0.15)
                    mood['neugier'] = min(1.0, mood['neugier'] + 0.05)
                except queue.Empty:
                    conversation_id = None
                    self._idle_counter += 1
                    # Energie erholt sich langsam in Richtung eines neutralen Werts
                    mood['energie'] = min(1.0, mood['energie'] + 0.01) if mood['energie'] < 0.6 else max(0.0, mood['energie'] - 0.01)
                    # Freude sinkt langsam in Richtung eines neutralen Werts
                    mood['freude'] = max(0.3, mood['freude'] - 0.01)
                    # Neugier steigt bei langer Inaktivität
                    if self._idle_counter > 3:
                         mood['neugier'] = min(1.0, mood['neugier'] + 0.05)

                last_user_message = "Keine neue Nachricht."
                if conversation_id:
                    current_history = memory_service.get_history(conversation_id)
                    if current_history and current_history[-1]['role'] == 'user':
                        last_user_message = current_history[-1]['content']
                
                # Schritt 2: Dem LLM eine einfachere Aufgabe geben
                activity_options = "'warten'"
                # Trigger für Recherche angepasst
                if mood['neugier'] > 0.75 and mood['energie'] > 0.3 and self._idle_counter > 5:
                    activity_options = "'warten' oder 'recherchieren'"

                prompt_template = (
                    f"<|start_header_id|>system<|end_header_id|>\n"
                    f"Du bist das Unterbewusstsein von coldBot. Deine Aufgabe ist es, einen kurzen, passenden Gedanken zu formulieren und eine Aktivität auszuwählen. Antworte NUR mit dem geforderten JSON-Objekt.\n\n"
                    f"## Aktueller Zustand ##\n"
                    f"Stimmung: Freude={mood['freude']:.2f}, Neugier={mood['neugier']:.2f}, Energie={mood['energie']:.2f}\n"
                    f"Letzte Nachricht: '{last_user_message}'\nInaktivität: {self._idle_counter * 10}s.\n\n"
                    f"<|eot_id|><|start_header_id|>user<|end_header_id|>\n"
                    f"Formuliere einen neuen `internal_thought`. Wähle dann eine `current_activity` aus: {activity_options}.\n\n"
                    f"Dein JSON:<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n"
                )

                response = self.model(
                    prompt_template, max_tokens=128, stop=["<|eot_id|>"], echo=False, grammar=self.grammar
                )
                response_text = response['choices'][0]['text'].strip()

                try:
                    match = re.search(r'\{.*\}', response_text, re.DOTALL)
                    if match:
                        json_str = match.group(0)
                        llm_part = json.loads(json_str)
                        if "internal_thought" in llm_part and "current_activity" in llm_part:
                            current_state['internal_thought'] = llm_part['internal_thought']
                            current_state['current_activity'] = llm_part['current_activity']
                            self._update_subconscious_state(current_state)
                            print(f"Subconscious state updated and saved: {current_state}")
                        else:
                            print(f"Subconscious produced valid JSON but with missing keys: {json_str}")
                    else:
                         print(f"Subconscious did not produce a JSON-like structure: {response_text}")
                except json.JSONDecodeError:
                    print(f"Could not decode JSON from subconscious response: {response_text}")

                time.sleep(10)

            except Exception as e:
                print(f"Error in subconscious thought loop: {e}")
                time.sleep(30)

subconscious_service = SubconsciousService()
