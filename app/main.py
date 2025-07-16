# coldBotv2/app/main.py

from fastapi import FastAPI, HTTPException, File, UploadFile
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel
from typing import Optional, Dict, Any
import os
import uuid
import asyncio
import json

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
    version="1.1.0 (Agent mit strukturiertem Stream)", # Versionsupdate
)

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
        text = content.decode('utf-8')
        rag_service.add_text(text, source_name=file.filename)
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
async def multimodal_chat(message: MultimodalMessage):
    conversation_id = message.conversation_id
    
    user_prompt = message.message
    if message.image_description:
        image_context = f"[Bildanalyse: {message.image_description}]"
        user_prompt = f"{image_context}\n{user_prompt}" if user_prompt else image_context

    memory_service.add_to_history(conversation_id, {"role": "user", "content": user_prompt})

    async def response_generator():
        MAX_TURNS = 5
        for turn in range(MAX_TURNS):
            print(f"\n--- Agent Turn {turn + 1}/{MAX_TURNS} ---")
            
            history = memory_service.get_history(conversation_id)
            
            rag_context = ""
            # RAG-Suche nur bei der ersten Runde für die initiale Frage durchführen
            if turn == 0 and user_prompt:
                rag_context = rag_service.search(user_prompt)

            llm_response = text_service.generate_agent_response(history, rag_context)
            print(f"LLM raw response: {llm_response}")

            try:
                tool_call_data = json.loads(llm_response)
                if "tool_name" in tool_call_data and "tool_args" in tool_call_data:
                    tool_name = tool_call_data["tool_name"]
                    tool_args = tool_call_data["tool_args"]
                    
                    # --- KORREKTUR: Sende eine strukturierte JSON-Nachricht für den Werkzeugaufruf ---
                    tool_message = {"type": "tool_usage", "content": f"Benutze Werkzeug: {tool_name}..."}
                    yield json.dumps(tool_message) + "\n" # Zeilenumbruch als Trennzeichen
                    await asyncio.sleep(0.1) 

                    tool_result = tool_manager.execute_tool(tool_name, tool_args)
                    print(f"Tool '{tool_name}' result: {tool_result}")

                    memory_service.add_to_history(conversation_id, {"role": "assistant", "content": llm_response})
                    memory_service.add_to_history(conversation_id, {"role": "tool", "content": str(tool_result)})
                    
                    continue
                
            except (json.JSONDecodeError, TypeError):
                pass

            memory_service.add_to_history(conversation_id, {"role": "assistant", "content": llm_response})
            
            # --- KORREKTUR: Sende die finale Antwort als strukturierte JSON-Nachrichten ---
            for chunk in llm_response:
                llm_chunk_message = {"type": "llm_chunk", "content": chunk}
                yield json.dumps(llm_chunk_message) + "\n"
                await asyncio.sleep(0.02)
            break
        else:
            final_error_message = {"type": "llm_chunk", "content": "Der Agent konnte nach mehreren Versuchen keine endgültige Antwort finden."}
            yield json.dumps(final_error_message) + "\n"

    return StreamingResponse(response_generator(), media_type="text/plain; charset=utf-8")
