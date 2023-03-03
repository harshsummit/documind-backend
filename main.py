from typing import Union
from fastapi import FastAPI, Response
import json
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from script import run_yolo, get_ocr_result, get_doc_class, runDocUMind
from fastapi.middleware.cors import CORSMiddleware


origins = [
    "http://localhost",
    "http://localhost:3000",
    "*"
]


class Image(BaseModel):
    filename: str
    b64: str


class Pdf(BaseModel):
    filename: str
    b64: str


app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
def index():
    return {
        "Fast": "Vibe",
        "hello": "Welcome to the fastapi server",
        "services": {
            "extract text from image": "/extract/image",
            "extract text from pdf": "/extract/pdf",
            "classify the file": "/classify/"
        },
        "Thank you": "for using this service"
    }


@app.get("/yolo")
async def index():
    return {"text": run_yolo()}

@app.get("/ocr")
async def index():
    return get_ocr_result()


@app.get("/doc-class")
async def index():
    return get_doc_class()

@app.get("/documind")
async def index():
    data = runDocUMind("PAN Card", 80, ["logo-stamp","profile-image"], ["Piyush Bansal"])
    return JSONResponse(content=data)

# @app.post("/extract/pdf")
# async def index(pdf: Pdf):
#     return {"text": extract_from_pdf(pdf.b64, pdf.filename)}


# @app.post("/classify/image")
# async def index(image: Image):
#     return {"class": classify_image(image.b64, image.filename)}


# @app.post("/classify/pdf")
# async def index(pdf: Pdf):
#     return {"text": classify_pdf(pdf.b64, pdf.filename)}