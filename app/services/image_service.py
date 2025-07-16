# coldBotv2/app/services/image_service.py

import torch
from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration
import io # Wichtig: Dieses Modul importieren

class ImageService:
    _instance = None
    processor = None
    model = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(ImageService, cls).__new__(cls)
            
            # Definiere das Modell, das wir von Hugging Face laden wollen
            model_id = "Salesforce/blip-image-captioning-base" 
            print(f"Loading image captioning model: {model_id}. This may take some time...")

            # Lade den Prozessor und das Modell
            cls.processor = BlipProcessor.from_pretrained(model_id)
            cls.model = BlipForConditionalGeneration.from_pretrained(model_id)
            
            print("Image captioning model loaded successfully.")
        return cls._instance

    def get_image_description(self, image_bytes: bytes) -> str:
        """
        Generiert eine Textbeschreibung für ein gegebenes Bild.
        """
        if self.processor is None or self.model is None:
            raise Exception("Image model is not initialized.")

        try:
            # --- KORREKTUR ---
            # Die rohen Bytes werden in einen In-Memory-Stream umgewandelt.
            # Das stellt sicher, dass Pillow die Daten korrekt lesen kann.
            image_stream = io.BytesIO(image_bytes)
            raw_image = Image.open(image_stream).convert('RGB')
            # --- ENDE DER KORREKTUR ---

            # Bereite das Bild für das Modell vor
            inputs = self.processor(raw_image, return_tensors="pt")

            # Generiere die Bildbeschreibung
            out = self.model.generate(**inputs, max_new_tokens=50)
            
            # Dekodiere die Beschreibung zu einem lesbaren Text
            caption = self.processor.decode(out[0], skip_special_tokens=True)
            
            return caption.strip()
        except Exception as e:
            print(f"Error generating image description: {e}")
            return "Ich konnte das Bild leider nicht analysieren."


# Erstelle eine globale Instanz des Services
image_service = ImageService()
