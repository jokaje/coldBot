# coldBotv2/app/main.py

from fastapi import FastAPI, HTTPException, File, UploadFile, BackgroundTasks
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel
from typing import Optional, Dict, Any, List
import os
import uuid
import asyncio
import json
from datetime import datetime

from .services.text_service import text_service
from .services.image_service import image_service
from .services.memory_service import memory_service
from .services.rag_service import rag_service
from .services.tool_manager import tool_manager

class MultimodalMessage(BaseModel):
    message: str
    image_description: Optional[str] = None
    conversation_id: str

app = FastAPI(
    title="coldBotv2 API",
    description="Modulare Offline KI-Agenten-API",
    version="1.4.0 (Intelligentes Gedächtnis)", # Versionsupdate
)

# --- Hintergrund-Reflektion ---
def run_and_save_reflection(conversation_id: str, user: str = "default_user"):
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

# --- API Endpunkte ---

@app.get("/", include_in_schema=False)
async def get_frontend():
    frontend_path = os.path.join(os.path.dirname(__file__), "..", "frontend", "index.html")
    if os.path.exists(frontend_path):
        return FileResponse(frontend_path)
    raise HTTPException(status_code=404, detail="Frontend not found.")

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

@app.post("/chat/multimodal", tags=["Chat"])
async def multimodal_chat(message: MultimodalMessage, background_tasks: BackgroundTasks):
    conversation_id = message.conversation_id
    
    user_prompt = message.message
    if message.image_description:
        image_context = f"[Bildanalyse: {message.image_description}]"
        user_prompt = f"{image_context}\n{user_prompt}" if user_prompt else image_context

    memory_service.add_to_history(conversation_id, {"role": "user", "content": user_prompt})

    # --- NEU: Intelligente Gedächtnis-Logik ---
    
    # 1. Hole thematisch relevante Fakten
    fact_results = rag_service.search(query_text=user_prompt, n_results=3, filter_metadata={"source": "core_memory_fact"})
    
    # 2. Hole thematisch relevante, ältere Gespräche
    relevant_episode_results = rag_service.search(query_text=user_prompt, n_results=2, filter_metadata={"source": "core_memory_episode"})
    
    # 3. Hole IMMER das letzte Gespräch
    last_episode = None
    all_episodes = rag_service.get_all_by_meta(filter_metadata={"source": "core_memory_episode"})
    if all_episodes:
        # Sortiere alle Episoden nach Zeitstempel, um die jüngste zu finden
        all_episodes.sort(key=lambda x: x['metadata']['timestamp'], reverse=True)
        last_episode = all_episodes[0]

    # 4. Kombiniere und dedupliziere die Erinnerungen
    final_episodes = {}
    if last_episode:
        final_episodes[last_episode['document']] = last_episode
    for res in relevant_episode_results:
        final_episodes[res['document']] = res

    # 5. Formatiere den Gedächtnis-Teil für den Prompt
    unique_facts = list(set([res['document'] for res in fact_results]))
    
    memory_prompt_part = ""
    if unique_facts:
        facts_str = "\n".join([f"- {fact}" for fact in unique_facts])
        memory_prompt_part += f"Fakten über den Benutzer:\n{facts_str}\n"
    if final_episodes:
        # Sortiere die finalen Episoden wieder nach Zeit für eine logische Anzeige im Prompt
        sorted_episodes = sorted(final_episodes.values(), key=lambda x: x['metadata']['timestamp'], reverse=True)
        episodes_str = "\n".join([f"- {res['document']} (Erinnert am {datetime.fromisoformat(res['metadata']['timestamp']).strftime('%d.%m.%Y')})" for res in sorted_episodes])
        memory_prompt_part += f"Relevante frühere Gespräche:\n{episodes_str}\n"


    async def response_generator():
        final_response_text = ""
        MAX_TURNS = 5
        for turn in range(MAX_TURNS):
            print(f"\n--- Agent Turn {turn + 1}/{MAX_TURNS} ---")
            
            history = memory_service.get_history(conversation_id)
            
            rag_context = ""
            if turn == 0 and user_prompt:
                rag_docs = rag_service.search(user_prompt, n_results=2, filter_metadata={"source": "rag_document"})
                rag_context = "\n".join([res['document'] for res in rag_docs])

            llm_response = text_service.generate_agent_response(history, rag_context, memory_prompt_part)
            print(f"LLM raw response: {llm_response}")

            try:
                tool_call_data = json.loads(llm_response)
                if "tool_name" in tool_call_data and "tool_args" in tool_call_data:
                    tool_name = tool_call_data["tool_name"]
                    tool_args = tool_call_data["tool_args"]
                    
                    tool_message = {"type": "tool_usage", "content": f"Benutze Werkzeug: {tool_name}..."}
                    yield json.dumps(tool_message) + "\n"
                    await asyncio.sleep(0.1) 

                    tool_result = tool_manager.execute_tool(tool_name, tool_args)
                    print(f"Tool '{tool_name}' result: {tool_result}")

                    memory_service.add_to_history(conversation_id, {"role": "assistant", "content": llm_response})
                    memory_service.add_to_history(conversation_id, {"role": "tool", "content": str(tool_result)})
                    
                    continue
                
            except (json.JSONDecodeError, TypeError):
                pass

            final_response_text = llm_response
            memory_service.add_to_history(conversation_id, {"role": "assistant", "content": final_response_text})
            
            for chunk in final_response_text:
                llm_chunk_message = {"type": "llm_chunk", "content": chunk}
                yield json.dumps(llm_chunk_message) + "\n"
                await asyncio.sleep(0.02)
            break
        else:
            final_error_message = {"type": "llm_chunk", "content": "Der Agent konnte nach mehreren Versuchen keine endgültige Antwort finden."}
            yield json.dumps(final_error_message) + "\n"
        
        if final_response_text:
             background_tasks.add_task(run_and_save_reflection, conversation_id)


    return StreamingResponse(response_generator(), media_type="text/plain; charset=utf-8")
