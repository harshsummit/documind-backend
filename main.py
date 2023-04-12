from typing import Union
from fastapi import FastAPI, UploadFile, File
import time
from typing import List
from fastapi.responses import JSONResponse, FileResponse
from pydantic import BaseModel
from script import run_yolo, get_ocr_result, get_doc_class, runDocUMind, multiDoc, multiRelation,classifyFromZipFile
from fastapi.middleware.cors import CORSMiddleware


origins = [
    "http://localhost",
    "http://localhost:3000",
    "*"
]

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.middleware("http")
async def add_process_time_header(request, call_next):
    start_time = time.time()
    response = await call_next(request)
    process_time = time.time() - start_time
    response.headers["X-Process-Time"] = str(f'{process_time:0.4f} sec')
    return response

@app.get("/")
def index():
    return {
        "Documind": "Server",
        "Hello": "Welcome to the fastapi server",
        "services": {
            "extract featuers from image": "/yolo",
            "extract text from file": "/ocr",
            "classify the file": "/doc-class",
            "run the whole documind": "/documind"
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

class Payload(BaseModel):
    doclabel: str
    classificationThreshold: int
    idChecks: List[str] = []
    detailCheck: List[str] = []

class Document(BaseModel):
    docid: str
    filename: str
    fileb64: str
    payload: Payload

# class Documents(BaseModel):
#     documents: List[Document]

class Relations(BaseModel):
    relationId: int
    relationName: str
    relationImage: str
    documents: List[Document]

class Application(BaseModel):
    applicationId: int
    relations: List[Relations]

@app.get("/documind")
async def index():
    data = runDocUMind("1","PAN Card", 80, ["logo-stamp","profile-image"], ["Piyush Bansal"], 'test/test2.jpg')
    return JSONResponse(content=data)

@app.post("/documind")
async def index(application: Application):
    return JSONResponse(content=multiRelation(application))

@app.get("/test")
async def index(document: Document):
    return {"text": document}


# @app.post("/classify/image")
# async def index(image: Image):
#     return {"class": classify_image(image.b64, image.filename)}


# @app.post("/classify/pdf")
# async def index(pdf: Pdf):
#     return {"text": classify_pdf(pdf.b64, pdf.filename)}

@app.post("/upload-zip/")
async def create_upload_file(file: UploadFile = File(...)):
    contents = await file.read()
    print("_____________________",type(contents))
    classifyFromZipFile(contents, 1)
    return FileResponse('result.zip')

@app.post("/upload-zip-2/")
async def create_upload_file(file: UploadFile = File(...)):
    contents = await file.read()
    print("_____________________",type(contents))
    classifyFromZipFile(contents, 2)
    print("we reached here")
    return FileResponse('result.zip')