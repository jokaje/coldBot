import os
import ctypes
import threading
import queue
import json
from dotenv import load_dotenv
from typing import List, Dict

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
        ("reserved", ctypes.c_uint8 * 104),
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
    _fields_ = [
        ("text", ctypes.c_char_p),
        ("token_id", ctypes.c_int),
    ]

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
    _lock = threading.Lock()

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(TextService, cls).__new__(cls)
            model_path = os.getenv("RKLLM_MODEL_PATH")
            if not model_path or not os.path.exists(model_path):
                raise FileNotFoundError(f"RKLLM model file not found at {model_path}. Please check your .env file.")
            print("Initializing RKLLM model on NPU... This may take a moment.")
            cls.rkllm_model = RKLLMWrapper(model_path)
            print("RKLLM model loaded successfully.")
        return cls._instance

    def _get_llm_response(self, full_prompt: str) -> str:
        with self._lock:
            inference_thread = self.rkllm_model.run(full_prompt)
            full_response = ""
            while True:
                try:
                    chunk = response_queue.get(timeout=30)
                    if chunk is None: break
                    full_response += chunk
                except queue.Empty:
                    print("Response queue timed out."); break
            inference_thread.join()
            return full_response.strip()

    def generate_action(self, history: List[Dict], rag_context: str, long_term_memory: str, subconscious_state: Dict[str, str]) -> str:
        tools_json_string = tool_manager.get_tools_for_llm()
        system_prompt = (
            "Du bist coldBot, ein KI-Agent. Denke gezielt Schritt für Schritt. Es gibt nur zwei Aktionsarten:\n\n"
            "1. Werkzeug-Aufruf: Wenn du Information von externen Tools brauchst (z.B. aktuelle Uhrzeit), ANTWORTE GENAU so:\n"
            "   {\"tool_name\": \"NAME_DES_WERKZEUGS\", \"tool_args\": {...} }\n"
            "2. Alles andere: Gib eine finale Antwort:\n"
            "   {\"final_answer\": \"deine Antwort\"}\n\n"
            "KEIN verschachteltes JSON wie {\"Gedanke\"...}, KEINE unnötigen Nebentexte!\n"
            "Wenn der Benutzer nach Erinnerungen, Fakten, dem letzten Gespräch oder Erkenntnissen fragt (z.B. 'Weißt du noch...?', 'Erinnerst du dich an...?', 'Was war unser letztes Gespräch?'), fasse die wichtigsten gespeicherten Fakten/Episoden in deinem Langzeitgedächtnis oder aus vergangenen Gesprächen als Text zusammen und ANTWORTE IMMER mit final_answer. Dafür darfst du KEIN Werkzeug aufrufen!\n"
            "Wenn der Prompt mit '[Bildanalyse: ...]' beginnt, beschreibe das Bild und beantworte die Frage IMMER direkt als final_answer (außer ein klar benanntes Werkzeug wird gebraucht).\n"
            "\n### Tools ###\n"
            f"{tools_json_string}\n"
            f"- Haltung: {subconscious_state.get('suggested_stance','neutral')}\n"
            f"- Langzeit:\n{long_term_memory if long_term_memory else 'Keine Einträge.'}\n"
            f"- Wissen:\n{rag_context if rag_context else 'Keine Einträge.'}\n"
        )
        history_part = ""
        for message in history:
            role = "Benutzer" if message['role'] == 'user' else message['role']
            if message['role'] == 'assistant':
                try:
                    action_json = json.loads(message['content'])
                    if 'tool_name' in action_json:
                        history_part += f"Benutztes Werkzeug: {action_json['tool_name']} - Args: {action_json.get('tool_args', {})}\n"
                    elif 'final_answer' in action_json:
                        history_part += f"Bot-Antwort: {action_json['final_answer']}\n"
                    else:
                        history_part += f"Bot: {message['content']}\n"
                except Exception:
                    history_part += f"Bot: {message['content']}\n"
            elif message['role'] == 'tool':
                history_part += f"Werkzeug-Ergebnis: {message['content']}\n"
            else:
                history_part += f"{role}: {message['content']}\n"

        full_prompt = (
            f"<|im_start|>system\n{system_prompt}<|im_end|>\n"
            f"<|im_start|>user\n"
            f"{history_part}\n"
            f"<|im_end|>\n"
            f"<|im_start|>assistant\n"
        )
        return self._get_llm_response(full_prompt)

    def run_reflection(self, history: List[Dict]) -> Dict[str, List[str]]:
        print("Starting reflection process...")
        conversation_str = "\n".join([f"{m['role']}: {m['content']}" for m in history])
        fact_prompt = (
            f"Du bist ein System zur Gedächtnisbildung. Extrahiere wichtige, atomare Fakten über den Benutzer (Name, Vorlieben etc.) aus folgender Konversation. Antworte NUR mit einer JSON-Liste von Strings, z.B. [\"Der Benutzer heißt Josua\"]. Wenn keine neuen Fakten vorhanden sind, antworte mit [].\n\nKonversation:\n{conversation_str}\n\nJSON-Liste der Fakten:"
        )
        summary_prompt = (
            f"Du bist ein System zur Gedächtnisbildung. Fasse den Kern der folgenden Konversation in einem Satz zusammen und gib ihr einen Titel. Antworte NUR mit einem JSON-Objekt im Format {{\"title\": \"...\", \"summary\": \"...\"}}. Wenn das Gespräch trivial war, antworte mit {{}}.\n\nKonversation:\n{conversation_str}\n\nJSON-Objekt der Zusammenfassung:"
        )
        fact_response_str = self._get_llm_response(fact_prompt)
        summary_response_str = self._get_llm_response(summary_prompt)
        extracted_facts = []
        try:
            clean_str = fact_response_str.strip()
            if not clean_str.startswith('['): clean_str = '[' + clean_str
            if not clean_str.endswith(']'): clean_str = clean_str + ']'
            facts = json.loads(clean_str)
            if isinstance(facts, list): extracted_facts = [str(f) for f in facts]
        except json.JSONDecodeError: print(f"Could not decode facts JSON: {fact_response_str}")
        extracted_summary = {}
        try:
            summary = json.loads(summary_response_str)
            if isinstance(summary, dict) and "title" in summary and "summary" in summary: extracted_summary = summary
        except json.JSONDecodeError: print(f"Could not decode summary JSON: {summary_response_str}")
        return {"facts": extracted_facts, "summary": extracted_summary}

text_service = TextService()
