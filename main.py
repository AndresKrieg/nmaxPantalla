from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
import requests

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/api/generar-imagen")
async def generar_imagen(request: Request):
    try:
        data = await request.json()
        print("📥 Recibido del frontend:", data)
        model_version = data.get("model_version")
        replicate_token = data.get("replicate_token")
        prompt = data.get("prompt")
        image_url = data.get("image_url")
        mask = data.get("mask")
        creativity = float(data.get("creativity", 0.2))

        if not 0 <= creativity <= 1:
            raise ValueError("El valor de 'creativity' debe estar entre 0 y 1.")

        if not all([replicate_token, model_version, prompt, image_url, mask]):
            return {"error": "Faltan campos obligatorios en la solicitud."}

        if not replicate_token:
            return {"error": "No se recibió la API key"}
        print(f"🔑 Token que se está usando para Replicate: {replicate_token}") 

        if not model_version:
            return {"error": "No se recibió la la version del modelo"}
        print(f"🔑 version del modelo que se está usando para Replicate: {model_version}") 

        if not prompt or not image_url:
            return {"error": "Faltan parámetros (prompt o image_url)"}

        image_mask = mask
        negative_prompt = "blurry, two riders, respect the mask, distorted, extra limbs, modify mask, unrealistic lighting, low quality, wrong colors, vehicle flying, deformed rider, shadows missing, duplicated wheels, glitch, no tire tracks"

        # Enviar solicitud a Replicate
        response = requests.post(
            "https://api.replicate.com/v1/predictions",
            headers={
                "Authorization": f"Token {replicate_token}",
                "Content-Type": "application/json"
            },
            json={
                "version": model_version,
                "input": {
                    "hdr": 0,
                    "mask": image_mask,
                    "image": image_url,
                    "steps": 25,
                    "width": 1024,
                    "height": 512,
                    "prompt": prompt,
                    "negative_prompt": negative_prompt,
                    "scheduler": "DPMSolverMultistep",
                    "creativity": creativity,
                    "resolution": "original",
                    "resemblance": 0.5,
                    "guidance_scale": 7.5
                }
            }
        ) 

        print("📤 Enviando a Replicate:", {
            "prompt": prompt,
            "image": image_url,
            "mask": mask,
            "negative_prompt": negative_prompt,
            "creativity": creativity,
        })

        prediction = response.json()
        prediction_url = prediction.get("urls", {}).get("get")

        if not prediction_url:
            return {"error": "No se pudo obtener la URL de seguimiento del modelo"}

        # Esperar resultado
        while True:
            result = requests.get(
                prediction_url,
                headers={"Authorization": f"Token {replicate_token}"}
            ).json()

            print("⌛ Estado actual:", result["status"])

            if result["status"] == "succeeded":
                return {"imagen_generada": result["output"][0]}
            elif result["status"] == "failed":
                return {"error": "Fallo en la generación de imagen"}

    except Exception as e:
        print("❌ Error inesperado:", str(e))
        return {"error": f"Error en el backend: {str(e)}"}  
