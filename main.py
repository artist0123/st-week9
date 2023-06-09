import uvicorn
from fastapi import FastAPI, File, UploadFile, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
import numpy as np
import cv2
import base64


app = FastAPI()

templates = Jinja2Templates(directory="templates")

class ImageRequest(BaseModel):
    image: str


# encode image as base64 string
def encode_image(image):
    _, encoded_image = cv2.imencode(".jpg", image)
    return "data:image/jpeg;base64," + base64.b64encode(encoded_image).decode()

# decode base64 string to image
def decode_image(image_string):
    encoded_data = image_string.split(',')[1]
    nparr = np.frombuffer(base64.b64decode(encoded_data), np.uint8)
    return cv2.imdecode(nparr, cv2.IMREAD_COLOR)

def apply_canny(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 100, 200)
    return edges


@app.get("/", response_class=HTMLResponse)
async def homepage(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


@app.post("/process-image")
async def process_image(request: Request, file: UploadFile = File()):
    #Read uploaded file as byte
    data = file.file.read()
    file.file.close()

    #Encode byte to base64
    image = "data:image/jpeg;base64," + base64.b64encode(data).decode("utf-8")
    decoded = decode_image(image)
    edges = apply_canny(decoded)
    processed_image = encode_image(edges)

    return templates.TemplateResponse("index.html", {"request": request, "image": image, "processed_image": processed_image})







