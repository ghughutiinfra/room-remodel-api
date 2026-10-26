from fastapi import FastAPI, File, Form, UploadFile
from fastapi.responses import Response, JSONResponse
from huggingface_hub import InferenceClient
from PIL import Image
import io
import os

app = FastAPI()

@app.get("/")
def home():
    return {"status": "AI Interior Remodel Backend is Ready!"}

@app.post("/generate")
async def generate_remodel(
    style: str = Form(...),
    image: UploadFile = File(...)
):
    try:
        image_bytes = await image.read()
        prompt = f"Transform this room into a luxurious {style} interior design, highly detailed, photorealistic architectural photography"

        hf_token = os.environ.get("HF_TOKEN", "")
        if not hf_token:
            return JSONResponse(status_code=500, content={"error": "HF_TOKEN secret is missing."})

        client = InferenceClient("timbrooks/instruct-pix2pix", token=hf_token)
        pil_image = Image.open(io.BytesIO(image_bytes)).convert("RGB")

        result_image = client.image_to_image(pil_image, prompt=prompt)

        if result_image is None:
            return JSONResponse(status_code=500, content={"error": "AI model failed to generate image."})

        img_byte_arr = io.BytesIO()
        result_image.save(img_byte_arr, format='JPEG')

        return Response(content=img_byte_arr.getvalue(), media_type="image/jpeg")

    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})
