# coldBotv2/app/main.py

from fastapi import FastAPI, HTTPException, File, UploadFile, BackgroundTasks, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse
from pydantic import BaseModel
from typing import Optional, Dict, Any, List
import os
import uuid
import asyncio
import json
from datetime import datetime
from apscheduler.schedulers.asyncio import AsyncIOScheduler

# Importiere alle Services
from .services.text_service import text_service
from .services.image_service import image_service
from .services.memory_service import memory_service
from .services.rag_service import rag_service
from .services.tool_manager import tool_manager

# --- WebSocket Connection Manager ---
class ConnectionManager:
    def __init__(self):
        self.active_connections: Dict[str, WebSocket] = {}

    async def connect(self, websocket: WebSocket, client_id: str):
        await websocket.accept()
        self.active_connections[client_id] = websocket
        print(f"Client {client_id} connected.")

    def disconnect(self, client_id: str):
        if client_id in self.active_connections:
            del self.active_connections[client_id]
            print(f"Client {client_id} disconnected.")

    async def send_personal_message(self, message: str, client_id: str):
        if client_id in self.active_connections:
            websocket = self.active_connections[client_id]
            await websocket.send_text(message)

manager = ConnectionManager()

# --- App Initialisierung ---
app = FastAPI(
    title="coldBotv2 API",
    description="Modulare Offline KI-Agenten-API",
    version="2.0.1 (Proaktiver Agent - Stabilitäts-Fix)",
)

# --- KORREKTUR: Fehlende Funktion wieder eingefügt ---
def run_and_save_reflection(conversation_id: str, user: str = "default_user"):
    """
    Führt die Reflektion durch und speichert die Ergebnisse im Langzeitgedächtnis.
    Diese Funktion wird als Hintergrundaufgabe ausgeführt.
    """
    history = memory_service.get_history(conversation_id)
    if len(history) < 2:
        print("Skipping reflection for short conversation.")
        return

    results = text_service.run_reflection(history)
    
    if results.get("facts"):
        unique_facts = list(set(results["facts"]))
        if unique_facts:
            metadatas = [{
                "source": "core_memory_fact", 
                "user": user, 
                "timestamp": datetime.now().isoformat()
            } for _ in unique_facts]
            rag_service.add_texts(texts=unique_facts, metadatas=metadatas)

    summary_data = results.get("summary")
    if summary_data and summary_data.get("summary"):
        summary_text = f"Titel: {summary_data['title']}. Inhalt: {summary_data['summary']}"
        metadata = {
            "source": "core_memory_episode", 
            "user": user, 
            "timestamp": datetime.now().isoformat()
        }
        rag_service.add_texts(texts=[summary_text], metadatas=[metadata])


# --- Proaktiver Denkprozess ---
async def proactive_check():
    """
    Wird vom Scheduler aufgerufen, um proaktiv zu prüfen, ob der Bot etwas tun sollte.
    """
    print(f"\n--- Running Proactive Check at {datetime.now()} ---")
    
    user = "default_user"
    
    all_facts = rag_service.get_all_by_meta(filter_metadata={"source": "core_memory_fact", "user": user})
    all_episodes = rag_service.get_all_by_meta(filter_metadata={"source": "core_memory_episode", "user": user})

    if not all_facts and not all_episodes:
        print("Proactive check: No memories found for user. Nothing to do.")
        return

    memory_prompt_part = "Fakten über den Benutzer:\n" + "\n".join([f"- {res['document']}" for res in all_facts])
    memory_prompt_part += "\n\nFrühere Gespräche:\n" + "\n".join([f"- {res['document']}" for res in all_episodes])

    proactive_prompt = (
        f"Du bist die proaktive Steuerungseinheit von coldBot. Analysiere das folgende Gedächtnis und die aktuelle Uhrzeit: {datetime.now().strftime('%Y-%m-%d %H:%M')}. "
        f"Gibt es basierend darauf eine relevante, nicht aufdringliche und hilfreiche Nachricht, die du dem Benutzer senden solltest? (z.B. eine Erinnerung, eine nette Frage basierend auf einem früheren Gespräch). "
        f"Antworte NUR mit der Nachricht, die an den Benutzer gesendet werden soll. Wenn es nichts Wichtiges gibt, antworte mit dem exakten Text 'Keine Aktion erforderlich.'\n\n"
        f"## Gedächtnis ##\n{memory_prompt_part}\n\n"
        f"Nachricht an den Benutzer (oder 'Keine Aktion erforderlich.'):"
    )

    print("Asking LLM for proactive message...")
    llm_decision = text_service._get_llm_response(proactive_prompt)
    print(f"LLM proactive decision: {llm_decision}")

    if llm_decision != "Keine Aktion erforderlich.":
        for client_id in manager.active_connections:
             proactive_message = {"type": "proactive_message", "content": llm_decision}
             await manager.send_personal_message(json.dumps(proactive_message), client_id)
             print(f"Sent proactive message to {client_id}: {llm_decision}")


# --- Scheduler Setup ---
scheduler = AsyncIOScheduler()
scheduler.add_job(proactive_check, 'interval', hours=1)

@app.on_event("startup")
async def startup_event():
    scheduler.start()
    print("Scheduler started.")

@app.on_event("shutdown")
async def shutdown_event():
    scheduler.shutdown()
    print("Scheduler shut down.")


# --- WebSocket Endpunkt ---
@app.websocket("/ws/{client_id}")
async def websocket_endpoint(websocket: WebSocket, client_id: str):
    await manager.connect(websocket, client_id)
    try:
        while True:
            await websocket.receive_text()
    except WebSocketDisconnect:
        manager.disconnect(client_id)


# --- HTTP Endpunkte ---

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

    memory_service.add_to_history(client_id, {"role": "user", "content": user_prompt})

    fact_results = rag_service.search(query_text=user_prompt, n_results=3, filter_metadata={"source": "core_memory_fact"})
    relevant_episode_results = rag_service.search(query_text=user_prompt, n_results=2, filter_metadata={"source": "core_memory_episode"})
    all_episodes = rag_service.get_all_by_meta(filter_metadata={"source": "core_memory_episode"})
    last_episode = all_episodes[-1] if all_episodes else None
    
    final_episodes = {}
    if last_episode: final_episodes[last_episode['document']] = last_episode
    for res in relevant_episode_results: final_episodes[res['document']] = res

    unique_facts = list(set([res['document'] for res in fact_results]))
    
    memory_prompt_part = ""
    if unique_facts: memory_prompt_part += f"Fakten über den Benutzer:\n" + "\n".join([f"- {fact}" for fact in unique_facts]) + "\n"
    if final_episodes: memory_prompt_part += f"Relevante frühere Gespräche:\n" + "\n".join([f"- {res['document']}" for res in sorted(final_episodes.values(), key=lambda x: x['metadata']['timestamp'], reverse=True)]) + "\n"

    final_response_text = ""
    MAX_TURNS = 5
    for turn in range(MAX_TURNS):
        history = memory_service.get_history(client_id)
        rag_context = ""
        if turn == 0 and user_prompt:
            rag_docs = rag_service.search(user_prompt, n_results=2, filter_metadata={"source": "rag_document"})
            rag_context = "\n".join([res['document'] for res in rag_docs])

        llm_response = text_service.generate_agent_response(history, rag_context, memory_prompt_part)
        
        try:
            tool_call_data = json.loads(llm_response)
            if "tool_name" in tool_call_data:
                tool_name = tool_call_data["tool_name"]
                tool_args = tool_call_data.get("tool_args", {})
                
                tool_message = {"type": "tool_usage", "content": f"Benutze Werkzeug: {tool_name}..."}
                await manager.send_personal_message(json.dumps(tool_message), client_id)
                
                tool_result = tool_manager.execute_tool(tool_name, tool_args)
                memory_service.add_to_history(client_id, {"role": "assistant", "content": llm_response})
                memory_service.add_to_history(client_id, {"role": "tool", "content": str(tool_result)})
                continue
        except (json.JSONDecodeError, TypeError):
            pass

        final_response_text = llm_response
        memory_service.add_to_history(client_id, {"role": "assistant", "content": final_response_text})
        
        for chunk in final_response_text:
            llm_chunk_message = {"type": "llm_chunk", "content": chunk}
            await manager.send_personal_message(json.dumps(llm_chunk_message), client_id)
            await asyncio.sleep(0.02)
        break
    
    if final_response_text:
        background_tasks.add_task(run_and_save_reflection, client_id)

    return {"status": "success", "message": "Response sent via WebSocket."}


@app.get("/", include_in_schema=False)
async def get_frontend():
    frontend_path = os.path.join(os.path.dirname(__file__), "..", "frontend", "index.html")
    if os.path.exists(frontend_path):
        return FileResponse(frontend_path)
    raise HTTPException(status_code=404, detail="Frontend not found.")

# --- KORREKTUR: Die helper Endpunkte wurden repariert ---
@app.post("/memory/add_document", tags=["Memory"])
async def add_document_to_memory(file: UploadFile = File(...)):
    try:
        content = await file.read()
        text_chunks = content.decode('utf-8').split('\n\n')
        metadatas = [{"source": "rag_document", "filename": file.filename} for _ in text_chunks]
        rag_service.add_texts(texts=text_chunks, metadatas=metadatas)
        return {"status": "success", "filename": file.filename, "message": "Knowledge added."}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Could not process file: {e}")

@app.post("/chat/image", tags=["Chat"], deprecated=True)
async def image_analysis_only(file: UploadFile = File(...)):
    try:
        image_bytes = await file.read()
        return {"image_description": image_service.get_image_description(image_bytes)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Fehler bei der Bildanalyse: {e}")
