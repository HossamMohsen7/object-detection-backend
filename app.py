from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
from transformers import (
    BlipProcessor,
    BlipForConditionalGeneration,
    MarianMTModel,
    MarianTokenizer,
)
from base64 import b64decode
from io import BytesIO
import time


app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-large")
model = BlipForConditionalGeneration.from_pretrained(
    "Salesforce/blip-image-captioning-large"
)
# Initialize the MarianMT model and tokenizer for translation
translation_model_name = "Helsinki-NLP/opus-mt-en-ar"
translation_model = MarianMTModel.from_pretrained(translation_model_name)
translation_tokenizer = MarianTokenizer.from_pretrained(translation_model_name)


def generate_caption(img):
    raw_image = img.convert("RGB")

    inputs = processor(raw_image, return_tensors="pt", max_new_tokens=100)

    start_time = time.time()
    out = model.generate(**inputs)
    generation_time = time.time() - start_time

    caption = processor.decode(out[0], skip_special_tokens=True)

    # Translate the caption into Arabic
    translated_inputs = translation_tokenizer(caption, return_tensors="pt")
    translated_out = translation_model.generate(**translated_inputs)
    translated_caption = translation_tokenizer.decode(
        translated_out[0], skip_special_tokens=True
    )

    return translated_caption


@app.get("/")
async def main():
    return {"message": "Hello World"}


# json data that has "image" as base64 string
@app.post("/predict")
async def predict(data: dict):
    image = data["image"]
    # convert the base64 string to an image
    image = Image.open(BytesIO(b64decode(image)))
    caption = generate_caption(image)
    return {"message": caption}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
