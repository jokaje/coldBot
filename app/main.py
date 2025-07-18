from fastapi import FastAPI, HTTPException, File, UploadFile, BackgroundTasks, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel
from typing import Optional, Dict, Any, List
import os
import json
from datetime import datetime
from apscheduler.schedulers.asyncio import AsyncIOScheduler
import re
import asyncio

from .services.text_service import text_service
from .services.image_service import image_service
from .services.memory_service import memory_service
from .services.rag_service import rag_service
from .services.tool_manager import tool_manager
from .services.subconscious_service import subconscious_service

class ConnectionManager:
    def __init__(self): self.active_connections: Dict[str, WebSocket] = {}
    async def connect(self, websocket: WebSocket, client_id: str): await websocket.accept(); self.active_connections[client_id] = websocket; print(f"Client {client_id} connected.")
    def disconnect(self, client_id: str):
        if client_id in self.active_connections: del self.active_connections[client_id]; print(f"Client {client_id} disconnected.")
    async def send_personal_message(self, message: str, client_id: str):
        if client_id in self.active_connections:
            websocket = self.active_connections[client_id]
            try: await websocket.send_text(message)
            except WebSocketDisconnect: self.disconnect(client_id)

manager = ConnectionManager()

app = FastAPI(
    title="coldBotv2 API",
    description="Modulare Offline KI-Agenten-API mit Unterbewusstsein",
    version="3.5.1 (Memory Fix)",
)

async def run_and_save_reflection(conversation_id: str, user: str = "default_user"):
    history = memory_service.get_history(conversation_id)
    if len(history) < 2: return
    results = await text_service.run_reflection(history)
    if results.get("facts"):
        unique_facts = list(set(results["facts"]))
        if unique_facts:
            metadatas = [{"source": "core_memory_fact", "user": user, "timestamp": datetime.now().isoformat()} for _ in unique_facts]
            rag_service.add_texts(texts=unique_facts, metadatas=metadatas)
    summary_data = results.get("summary")
    if summary_data and summary_data.get("summary"):
        summary_text = f"Titel: {summary_data['title']}. Inhalt: {summary_data['summary']}"
        metadata = {"source": "core_memory_episode", "user": user, "timestamp": datetime.now().isoformat()}
        rag_service.add_texts(texts=[summary_text], metadatas=[metadata])

async def proactive_check():
    print(f"\n--- Running Proactive Check at {datetime.now()} ---")
    user = "default_user"
    subconscious_state = subconscious_service.get_last_subconscious_state()
    
    if subconscious_state.get("current_activity") == "recherchieren":
        thought = subconscious_state.get("internal_thought", "")
        search_query_match = re.search(r"über (.*?) nachzudenken|über (.*?) recherchieren|frage mich, (.*)", thought, re.IGNORECASE)
        if search_query_match:
            query = next((item for item in search_query_match.groups() if item is not None), None)
            if query:
                print(f"Subconscious is curious about '{query}'. Starting proactive research.")
                search_result = tool_manager.execute_tool("web_search", {"query": query})
                fact_to_save = f"Recherche-Ergebnis zu '{query}': {search_result}"
                tool_manager.execute_tool("save_fact_to_memory", {"fact": fact_to_save, "user": user})
                
                subconscious_service._update_subconscious_state({
                    "internal_thought": f"Ich habe gerade etwas über '{query}' gelernt. Das war interessant.",
                    "suggested_stance": "nachdenklich",
                    "current_activity": "warten"
                })
                return

    all_facts = rag_service.get_all_by_meta(filter_metadata={"source": "core_memory_fact", "user": user})
    all_episodes = rag_service.get_all_by_meta(filter_metadata={"source": "core_memory_episode", "user": user})
    if not all_facts and not all_episodes: return
    
    memory_prompt_part = "Fakten über den Benutzer:\n" + "\n".join([f"- {res['document']}" for res in all_facts]) + "\n\nFrühere Gespräche:\n" + "\n".join([f"- {res['document']}" for res in all_episodes])
    proactive_prompt = (f"Du bist die proaktive Steuerungseinheit von coldBot. Analysiere das folgende Gedächtnis und die aktuelle Uhrzeit: {datetime.now().strftime('%Y-%m-%d %H:%M')}. Gibt es basierend darauf eine relevante, hilfreiche Nachricht, die du dem Benutzer senden solltest? Antworte NUR mit der Nachricht oder mit 'Keine Aktion erforderlich.'\n\n## Gedächtnis ##\n{memory_prompt_part}\n\nNachricht an den Benutzer (oder 'Keine Aktion erforderlich.'):")
    llm_decision = await text_service._get_llm_response(proactive_prompt)
    
    if llm_decision and "keine aktion" not in llm_decision.lower():
        for client_id in list(manager.active_connections.keys()):
             proactive_message = {"type": "proactive_message", "content": llm_decision}
             await manager.send_personal_message(json.dumps(proactive_message), client_id)

scheduler = AsyncIOScheduler()
scheduler.add_job(proactive_check, 'interval', hours=1)

@app.on_event("startup")
async def startup_event():
    scheduler.start(); print("Scheduler started.")
    subconscious_service.start_thought_loop()

@app.on_event("shutdown")
async def shutdown_event():
    scheduler.shutdown(); print("Scheduler shut down.")

@app.websocket("/ws/{client_id}")
async def websocket_endpoint(websocket: WebSocket, client_id: str):
    await manager.connect(websocket, client_id)
    try:
        while True: await websocket.receive_text()
    except WebSocketDisconnect: manager.disconnect(client_id)

def extract_action_json(llm_action_str):
    try:
        match = re.search(r'\{.*\}', llm_action_str, re.DOTALL)
        if match:
            clean_json_str = match.group(0)
            outer = json.loads(clean_json_str)
            if isinstance(outer, dict):
                if "tool_name" in outer or "final_answer" in outer:
                    return outer
    except Exception:
        pass
    return None

class MultimodalMessage(BaseModel):
    message: str
    image_description: Optional[str] = None
    conversation_id: str

@app.post("/chat/multimodal", tags=["Chat"])
async def multimodal_chat(message: MultimodalMessage, background_tasks: BackgroundTasks):
    client_id = message.conversation_id
    user_prompt = message.message
    if message.image_description:
        image_context = f"[Bildanalyse: {message.image_description}]"
        user_prompt = f"{image_context}\n{user_prompt}" if user_prompt else image_context
    if not user_prompt:
        user_prompt = "Beschreibe das Bild."

    memory_service.add_to_history(client_id, {"role": "user", "content": user_prompt})
    subconscious_service.input_queue.put({"conversation_id": client_id})

    MAX_TURNS = 5
    for turn in range(MAX_TURNS):
        history = memory_service.get_history(client_id)
        rag_docs = rag_service.search(user_prompt, n_results=2, filter_metadata={"source": "rag_document"})
        rag_context = "\n".join([res['document'] for res in rag_docs])
        fact_results = rag_service.search(query_text=user_prompt, n_results=3, filter_metadata={"source": "core_memory_fact"})
        episode_results = rag_service.search(query_text=user_prompt, n_results=2, filter_metadata={"source": "core_memory_episode"})
        long_term_memory = "Fakten:\n" + "\n".join([f"- {res['document']}" for res in fact_results]) + "\nEpisoden:\n" + "\n".join([res['document'] for res in episode_results])
        current_subconscious_state = subconscious_service.get_last_subconscious_state()

        # --- HIER IST DER WIEDEREINGEFÜGTE FIX FÜR ERINNERUNGSFRAGEN ---
        memory_keywords = ["erinner", "erinnerst", "erinnerung", "letzte", "letztes", "voriges", "gespräch", "woran", "weiß"]
        if any(word in user_prompt.lower() for word in memory_keywords):
            print("Memory keyword detected, overriding LLM to construct memory response.")
            facts = "\n".join([res['document'] for res in fact_results])
            episodes = "\n".join([res['document'] for res in episode_results])
            summary = ""
            if episodes:
                summary += "Ich erinnere mich an folgende Gesprächsthemen:\n" + episodes
            if facts:
                summary += "\n\nZudem habe ich mir folgende Fakten gemerkt:\n" + facts
            
            final_response_text = summary if summary else "Ich habe dazu leider keine spezifischen Erinnerungen gefunden."
            
            memory_service.add_to_history(client_id, {"role": "assistant", "content": json.dumps({"final_answer": final_response_text})})
            llm_chunk_message = {"type": "llm_chunk", "content": final_response_text}
            await manager.send_personal_message(json.dumps(llm_chunk_message), client_id)
            if final_response_text:
                background_tasks.add_task(run_and_save_reflection, client_id, user="default_user")
            break # Wichtig: Die Schleife hier beenden

        llm_action_str = await text_service.generate_action(history, rag_context, long_term_memory, current_subconscious_state)
        action_json = extract_action_json(llm_action_str)

        if action_json and "tool_name" in action_json:
            tool_name = action_json["tool_name"]
            tool_args = action_json.get("tool_args", {})
            tool_message = {"type": "tool_usage", "content": f"Benutze Werkzeug: {tool_name}..."}
            await manager.send_personal_message(json.dumps(tool_message), client_id)
            tool_result = tool_manager.execute_tool(tool_name, tool_args)
            memory_service.add_to_history(client_id, {"role": "assistant", "content": json.dumps(action_json)})
            memory_service.add_to_history(client_id, {"role": "tool", "content": str(tool_result)})
            continue

        elif action_json and "final_answer" in action_json:
            final_response_text = action_json["final_answer"]
            memory_service.add_to_history(client_id, {"role": "assistant", "content": json.dumps(action_json)})
            llm_chunk_message = {"type": "llm_chunk", "content": final_response_text}
            await manager.send_personal_message(json.dumps(llm_chunk_message), client_id)
            if final_response_text:
                background_tasks.add_task(run_and_save_reflection, client_id, user="default_user")
            break
        
        print(f"LLM produced invalid action, treating as final answer: {llm_action_str}")
        final_response_text = llm_action_str
        memory_service.add_to_history(client_id, {"role": "assistant", "content": final_response_text})
        llm_chunk_message = {"type": "llm_chunk", "content": final_response_text}
        await manager.send_personal_message(json.dumps(llm_chunk_message), client_id)
        if final_response_text:
            background_tasks.add_task(run_and_save_reflection, client_id, user="default_user")
        break

    return JSONResponse(content={"status": "success", "message": "Agent loop finished."})

@app.get("/", include_in_schema=False)
async def get_frontend():
    frontend_path = os.path.join(os.path.dirname(__file__), "..", "frontend", "index.html")
    if os.path.exists(frontend_path): return FileResponse(frontend_path)
    raise HTTPException(status_code=404, detail="Frontend not found.")

@app.post("/memory/add_document", tags=["Memory"])
async def add_document_to_memory(file: UploadFile = File(...)):
    try:
        content = await file.read(); text_chunks = content.decode('utf-8').split('\n\n')
        metadatas = [{"source": "rag_document", "filename": file.filename} for _ in text_chunks]
        rag_service.add_texts(texts=text_chunks, metadatas=metadatas)
        return {"status": "success", "filename": file.filename, "message": "Knowledge added."}
    except Exception as e: raise HTTPException(status_code=500, detail=f"Could not process file: {e}")

@app.post("/chat/image", tags=["Chat"])
async def image_analysis_only(file: UploadFile = File(...)):
    try:
        image_bytes = await file.read()
        return {"image_description": image_service.get_image_description(image_bytes)}
    except Exception as e: raise HTTPException(status_code=500, detail=f"Fehler bei der Bildanalyse: {e}")
