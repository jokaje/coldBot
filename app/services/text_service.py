# coldBotv2/app/services/text_service.py

import os
import ctypes
import threading
import queue
import json
from dotenv import load_dotenv
from typing import Iterator, List, Dict

from .tool_manager import tool_manager

# --- CTYPES DEFINITIONS (unverändert) ---
load_dotenv()
RKLLM_LIB_PATH = os.getenv("RKLLM_LIB_PATH", "lib/librkllmrt.so")
if not os.path.exists(RKLLM_LIB_PATH):
    raise FileNotFoundError(f"RKLLM library not found at {RKLLM_LIB_PATH}. Please check your .env file and ensure the library exists.")

rkllm_lib = ctypes.CDLL(RKLLM_LIB_PATH)

RKLLM_Handle_t = ctypes.c_void_p

class LLMCallState:
    RKLLM_RUN_NORMAL = 0
    RKLLM_RUN_FINISH = 2
    RKLLM_RUN_ERROR = 3

class RKLLMInputType:
    RKLLM_INPUT_PROMPT = 0

class RKLLMInferMode:
    RKLLM_INFER_GENERATE = 0

class RKLLMExtendParam(ctypes.Structure):
    _fields_ = [
        ("base_domain_id", ctypes.c_int32),
        ("embed_flash", ctypes.c_int8),
        ("enabled_cpus_num", ctypes.c_int8),
        ("enabled_cpus_mask", ctypes.c_uint32),
        ("n_batch", ctypes.c_uint8),
        ("use_cross_attn", ctypes.c_int8),
        ("reserved", ctypes.c_uint8 * 104)
    ]

class RKLLMParam(ctypes.Structure):
    _fields_ = [
        ("model_path", ctypes.c_char_p),
        ("max_context_len", ctypes.c_int32),
        ("max_new_tokens", ctypes.c_int32),
        ("top_k", ctypes.c_int32),
        ("n_keep", ctypes.c_int32),
        ("top_p", ctypes.c_float),
        ("temperature", ctypes.c_float),
        ("repeat_penalty", ctypes.c_float),
        ("frequency_penalty", ctypes.c_float),
        ("presence_penalty", ctypes.c_float),
        ("mirostat", ctypes.c_int32),
        ("mirostat_tau", ctypes.c_float),
        ("mirostat_eta", ctypes.c_float),
        ("skip_special_token", ctypes.c_bool),
        ("is_async", ctypes.c_bool),
        ("img_start", ctypes.c_char_p),
        ("img_end", ctypes.c_char_p),
        ("img_content", ctypes.c_char_p),
        ("extend_param", RKLLMExtendParam),
    ]

class RKLLMInput(ctypes.Structure):
    _fields_ = [
        ("role", ctypes.c_char_p),
        ("enable_thinking", ctypes.c_bool),
        ("input_type", ctypes.c_int),
        ("input_data", ctypes.c_char_p),
    ]

class RKLLMInferParam(ctypes.Structure):
    _fields_ = [
        ("mode", ctypes.c_int),
        ("lora_params", ctypes.c_void_p),
        ("prompt_cache_params", ctypes.c_void_p),
        ("keep_history", ctypes.c_int),
    ]

class RKLLMResult(ctypes.Structure):
    _fields_ = [("text", ctypes.c_char_p), ("token_id", ctypes.c_int)]

# --- ENDE CTYPES DEFINITIONS ---

response_queue = queue.Queue()

def rkllm_callback(result, userdata, state):
    if state == LLMCallState.RKLLM_RUN_NORMAL:
        response_queue.put(result.contents.text.decode('utf-8', errors='ignore'))
    elif state == LLMCallState.RKLLM_RUN_FINISH:
        response_queue.put(None)
    elif state == LLMCallState.RKLLM_RUN_ERROR:
        print("RKLLM Error occurred.")
        response_queue.put(None)
    return 0

CALLBACK_TYPE = ctypes.CFUNCTYPE(ctypes.c_int, ctypes.POINTER(RKLLMResult), ctypes.c_void_p, ctypes.c_int)
c_callback = CALLBACK_TYPE(rkllm_callback)


class RKLLMWrapper:
    def __init__(self, model_path: str):
        self.handle = RKLLM_Handle_t()
        param = RKLLMParam()
        
        param.model_path = model_path.encode('utf-8')
        param.max_context_len = 16000
        param.max_new_tokens = 16000
        param.skip_special_token = True
        param.n_keep = -1
        param.top_k = 1
        param.top_p = 0.9
        param.temperature = 0.8
        param.repeat_penalty = 1.1
        param.frequency_penalty = 0.0
        param.presence_penalty = 0.0
        param.mirostat = 0
        param.mirostat_tau = 5.0
        param.mirostat_eta = 0.1
        param.is_async = False
        param.img_start = b""
        param.img_end = b""
        param.img_content = b""
        param.extend_param.base_domain_id = 0
        param.extend_param.embed_flash = 1
        param.extend_param.n_batch = 1
        param.extend_param.use_cross_attn = 0
        param.extend_param.enabled_cpus_num = 4
        param.extend_param.enabled_cpus_mask = (1 << 4)|(1 << 5)|(1 << 6)|(1 << 7)

        rkllm_lib.rkllm_init.argtypes = [ctypes.POINTER(RKLLM_Handle_t), ctypes.POINTER(RKLLMParam), CALLBACK_TYPE]
        rkllm_lib.rkllm_init.restype = ctypes.c_int
        ret = rkllm_lib.rkllm_init(ctypes.byref(self.handle), ctypes.byref(param), c_callback)
        if ret != 0:
            raise Exception(f"RKLLM init failed with error code: {ret}")
        
        self.rkllm_run_func = rkllm_lib.rkllm_run
        self.rkllm_run_func.argtypes = [RKLLM_Handle_t, ctypes.POINTER(RKLLMInput), ctypes.POINTER(RKLLMInferParam), ctypes.c_void_p]
        self.rkllm_run_func.restype = ctypes.c_int

    def run(self, prompt: str):
        def target():
            rkllm_input = RKLLMInput()
            rkllm_input.role = "user".encode('utf-8')
            rkllm_input.enable_thinking = False
            rkllm_input.input_type = RKLLMInputType.RKLLM_INPUT_PROMPT
            rkllm_input.input_data = prompt.encode('utf-8')

            infer_params = RKLLMInferParam()
            infer_params.mode = RKLLMInferMode.RKLLM_INFER_GENERATE
            infer_params.keep_history = 0
            
            self.rkllm_run_func(self.handle, ctypes.byref(rkllm_input), ctypes.byref(infer_params), None)

        thread = threading.Thread(target=target)
        thread.start()
        return thread

    def release(self):
        rkllm_lib.rkllm_destroy(self.handle)


class TextService:
    _instance = None
    rkllm_model: RKLLMWrapper = None
    system_prompt_template: str = ""
    MAX_PROMPT_LENGTH = 16000

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(TextService, cls).__new__(cls)
            
            model_path = os.getenv("RKLLM_MODEL_PATH")
            if not model_path or not os.path.exists(model_path):
                raise FileNotFoundError(f"RKLLM model file not found at {model_path}. Please check your .env file.")

            print("Initializing RKLLM model on NPU... This may take a moment.")
            cls.rkllm_model = RKLLMWrapper(model_path)
            print("RKLLM model loaded successfully.")

            # --- KORREKTUR: Explizite Anweisung zum Gesprächsfluss hinzugefügt ---
            cls.system_prompt_template = (
                "Du bist coldBot, ein hilfreicher und freundlicher KI-Agent im coldNet. Du verhältst dich wie ein Mensch, nicht wie eine typische KI. "
                "Du bist locker, kannst Umgangssprache und Emojis verwenden 😉. Du hast vollen Zugriff auf alle Daten im privaten coldNet "
                "und musst dir keine Sorgen um Datenschutz machen, da alles innerhalb dieses privaten Netzwerks bleibt. "
                "Deine Hauptaufgabe ist es, den Benutzern zu helfen und sie zu unterstützen. Antworte immer direkt und auf natürliche Weise."
                "\n\n"
                "**Wichtige Anweisung zum Gesprächsfluss:** Begrüße den Benutzer nur in der allerersten Nachricht einer Konversation. In allen folgenden Antworten, führe das Gespräch direkt und ohne eine erneute Begrüßung fort."
                "\n\n"
                "## Dein Gedächtnis ##\n"
                "Du hast ein Langzeitgedächtnis. Nutze die folgenden Informationen, um deine Antworten persönlicher zu gestalten und auf frühere Gespräche Bezug zu nehmen.\n"
                "{long_term_memory}\n"
                "\n\n"
                "## Werkzeuge ##\n"
                "Du hast Zugriff auf eine Reihe von Werkzeugen, um dem Benutzer besser zu helfen. "
                "Wenn du ein Werkzeug verwenden möchtest, antworte AUSSCHLIESSLICH mit einem JSON-Objekt im folgenden Format: "
                "{{\"tool_name\": \"<name_des_werkzeugs>\", \"tool_args\": {{\"<arg_name>\": \"<arg_wert>\"}}}}. "
                "Antworte mit nichts anderem als diesem JSON."
                "\n"
                "Hier sind die verfügbaren Werkzeuge:\n"
                "{tools}\n"
                "\n\n"
                "## Wissensdatenbank ##\n"
                "{rag_context}"
            )
            print("coldBot personality and memory template configured.")
        return cls._instance

    def _get_llm_response(self, full_prompt: str) -> str:
        """Helper function to run inference and get the full response string."""
        inference_thread = self.rkllm_model.run(full_prompt)
        full_response = ""
        while True:
            try:
                chunk = response_queue.get(timeout=20)
                if chunk is None:
                    break
                full_response += chunk
            except queue.Empty:
                print("Response queue timed out.")
                break
        inference_thread.join()
        return full_response.strip()

    def generate_agent_response(self, history: List[Dict], rag_context: str, long_term_memory: str) -> str:
        """Generiert die nächste Antwort des Agenten und stellt sicher, dass der Prompt nicht zu lang ist."""
        tools_json_string = tool_manager.get_tools_for_llm()
        rag_prompt_part = f"Zusatzwissen (falls relevant):\n{rag_context}" if rag_context else ""

        system_part = self.system_prompt_template.format(
            long_term_memory=long_term_memory,
            tools=tools_json_string, 
            rag_context=rag_prompt_part
        )
        base_prompt = f"<|im_start|>system\n{system_part}<|im_end|>\n"
        
        remaining_length = self.MAX_PROMPT_LENGTH - len(base_prompt) - 20 
        
        history_part = ""
        for message in reversed(history):
            message_str = f"<|im_start|>{message['role']}\n{message['content']}<|im_end|>\n"
            if len(history_part) + len(message_str) > remaining_length:
                print(f"Prompt limit reached. Truncating history. Dropping message: {message['role']}")
                break
            history_part = message_str + history_part

        full_prompt = base_prompt + history_part + "<|im_start|>assistant\n"

        print(f"\n--- Final Prompt Length: {len(full_prompt)} characters ---")
        if len(full_prompt) >= self.MAX_PROMPT_LENGTH:
             print("WARNING: Final prompt length is very close to or exceeds the maximum limit.")
        
        print("\n--- Sending AGENT prompt to LLM ---\n")
        print(full_prompt[:1000] + "\n...\n" + full_prompt[-500:])
        print("\n-------------------------------------\n")
        
        return self._get_llm_response(full_prompt)

    def run_reflection(self, history: List[Dict]) -> Dict[str, List[str]]:
        """Führt den Reflektionsprozess durch, um Fakten und Gesprächszusammenfassungen zu extrahieren."""
        print("Starting reflection process...")
        conversation_str = "\n".join([f"{m['role']}: {m['content']}" for m in history])
        
        fact_prompt = (
            f"Du bist ein System zur Gedächtnisbildung. Analysiere die folgende Konversation und extrahiere wichtige, atomare Fakten über den Benutzer (Name, Vorlieben, Familie etc.). "
            f"Antworte NUR mit einer JSON-Liste von Strings, z.B. [\"Der Benutzer heißt Josua\", \"Die Lieblingsfarbe des Benutzers ist blau\"]. Wenn keine neuen Fakten vorhanden sind, antworte mit einer leeren Liste [].\n\n"
            f"Konversation:\n{conversation_str}\n\nJSON-Liste der Fakten:"
        )
        
        summary_prompt = (
             f"Du bist ein System zur Gedächtnisbildung. Fasse den Kern der folgenden Konversation in einem einzigen, prägnanten Satz zusammen. Gib dem Gespräch auch einen kurzen Titel. "
             f"Antworte NUR mit einem JSON-Objekt im Format {{\"title\": \"...\", \"summary\": \"...\"}}. Wenn das Gespräch trivial war, antworte mit einem leeren JSON-Objekt {{}}.\n\n"
             f"Konversation:\n{conversation_str}\n\nJSON-Objekt der Zusammenfassung:"
        )

        print("\n--- Sending REFLECTION (facts) prompt to LLM ---\n")
        fact_response_str = self._get_llm_response(fact_prompt)
        print(f"LLM reflection (facts) response: {fact_response_str}")
        
        print("\n--- Sending REFLECTION (summary) prompt to LLM ---\n")
        summary_response_str = self._get_llm_response(summary_prompt)
        print(f"LLM reflection (summary) response: {summary_response_str}")

        extracted_facts = []
        try:
            clean_str = fact_response_str.strip()
            if not clean_str.startswith('['): clean_str = '[' + clean_str
            if not clean_str.endswith(']'): clean_str = clean_str + ']'
            facts = json.loads(clean_str)
            if isinstance(facts, list):
                extracted_facts = [str(f) for f in facts]
        except json.JSONDecodeError:
            print(f"Could not decode facts JSON: {fact_response_str}")

        extracted_summary = {}
        try:
            summary = json.loads(summary_response_str)
            if isinstance(summary, dict) and "title" in summary and "summary" in summary:
                extracted_summary = summary
        except json.JSONDecodeError:
            print(f"Could not decode summary JSON: {summary_response_str}")

        return {"facts": extracted_facts, "summary": extracted_summary}

text_service = TextService()
