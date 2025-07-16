# coldBotv2/app/main.py

from fastapi import FastAPI, HTTPException, File, UploadFile
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel
from typing import Optional
import os
import uuid
import asyncio

from .services.text_service import text_service
from .services.image_service import image_service
from .services.memory_service import memory_service
from .services.rag_service import rag_service # NEU

class MultimodalMessage(BaseModel):
    message: str
    image_description: Optional[str] = None
    conversation_id: Optional[str] = None

app = FastAPI(
    title="coldBotv2 API",
    description="Modulare Offline KI-Agenten-API",
    version="0.7.0",
)

@app.get("/", include_in_schema=False)
async def get_frontend():
    # ... (bleibt unverändert)
    frontend_path = os.path.join(os.path.dirname(__file__), "..", "frontend", "index.html")
    if os.path.exists(frontend_path):
        return FileResponse(frontend_path)
    raise HTTPException(status_code=404, detail="Frontend not found.")

# --- NEUER ENDPUNKT ZUM HINZUFÜGEN VON WISSEN ---
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
    # ... (bleibt unverändert)
    try:
        image_bytes = await file.read()
        return {"image_description": image_service.get_image_description(image_bytes)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Fehler bei der Bildanalyse: {e}")

@app.post("/chat/multimodal", tags=["Chat"])
async def multimodal_chat(message: MultimodalMessage):
    conversation_id = message.conversation_id or str(uuid.uuid4())
    history = memory_service.get_history(conversation_id)
    
    user_question = message.message
    
    # --- NEUE LOGIK: RAG-Suche ---
    # Suche nur nach Kontext, wenn es eine echte Frage gibt und kein Bild im Fokus ist.
    rag_context = ""
    if user_question and not message.image_description:
        print(f"Searching knowledge base for: '{user_question}'")
        rag_context = rag_service.search(user_question)

    prompt = ""
    if message.image_description:
        final_user_question = user_question if user_question else "Was siehst du auf diesem Bild?"
        prompt = f"Der Benutzer hat ein Bild hochgeladen. Deine Analyse des Bildes lautet: '{message.image_description}'. Die Frage des Benutzers dazu lautet: '{final_user_question}'"
    else:
        prompt = user_question

    async def response_generator():
        full_response = ""
        try:
            # Übergebe den RAG-Kontext an den TextService
            stream = text_service.generate_response_stream(prompt, history, rag_context)
            for chunk in stream:
                full_response += chunk
                yield chunk
                await asyncio.sleep(0)
            
            user_message_for_history = {"role": "user", "content": prompt}
            bot_message_for_history = {"role": "assistant", "content": full_response}
            memory_service.add_to_history(conversation_id, user_message_for_history, bot_message_for_history)
        except Exception as e:
            print(f"Error during stream generation: {e}")
            yield "Sorry, da ist etwas schiefgelaufen."

    return StreamingResponse(response_generator(), media_type="text/plain; charset=utf-8")
