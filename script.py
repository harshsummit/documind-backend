# imports for yolo
import torch
import base64
import numpy as np
from PIL import Image
import io

# imports for doc classification
from doc_classification_test import predict_document_image
from transformers import LayoutLMv3FeatureExtractor, LayoutLMv3TokenizerFast, LayoutLMv3Processor, LayoutLMv3ForSequenceClassification

# imports for paddle ocr
import os
from paddleocr import PaddleOCR
os.environ["KMP_DUPLICATE_LIB_OK"]="True"

# Import for blur check
import cv2

# For profile matching

from deepface import DeepFace
# from deepface.commons import functions
from sklearn.cluster import DBSCAN
from sklearn.metrics import pairwise_distances
##################

ocr = PaddleOCR(use_angle_cls=True, lang='en')

yolo_model = torch.hub.load('ultralytics/yolov5', 'custom', path='models/yolo.pt')

feature_extractor = LayoutLMv3FeatureExtractor(apply_ocr=False)
tokenizer = LayoutLMv3TokenizerFast.from_pretrained("models/layoutlmv3-base", local_files_only=True)
processor = LayoutLMv3Processor(feature_extractor, tokenizer)
model = LayoutLMv3ForSequenceClassification.from_pretrained("models/layoutlmv3", local_files_only=True)
model = model.eval().to("cpu")

img = 'test/test2.jpg'

def converB64tofile(b64):
  image = np.array(Image.open(io.BytesIO(base64.b64decode(b64))).convert("RGB"))
  return image

def run_yolo(image_path = img):
    try:
      results = yolo_model(image_path)
      # results.crop()
      # results.show()  # display results
      results.print()  # print results to screen
      # results.pandas().xyxy[0] shows all the coordinates and class of the object detected
      count_row = results.pandas().xyxy[0].shape[0]

      formattedResult = []
      for i in range(count_row):
        formattedResult.append(results.pandas().xyxy[0].iloc[i].to_dict())

      print('\n', formattedResult)
      # results.save(save_dir='results')
      return formattedResult
    except:
       return "Yolo Failed"

def get_ocr_result(image_path = img):
  result = ocr.ocr(image_path, cls=True)
  results_dict = []

  for text in result[0]:
    top = max(text[0][0][1],text[0][1][1],text[0][2][1],text[0][3][1])
    bottom = min(text[0][0][1],text[0][1][1],text[0][2][1],text[0][3][1])
    left = min(text[0][0][0],text[0][1][0],text[0][2][0],text[0][3][0])
    right = max(text[0][0][0],text[0][1][0],text[0][2][0],text[0][3][0])
    results_dict.append({
        'word':text[1][0],
        'bounding_box':[left, top, right, bottom]
    })
  return results_dict

def get_doc_class(ocr_result = [], image_path=img):
  if(len(ocr_result)==0):
   ocr_result =get_ocr_result()
  print(ocr_result)
  return predict_document_image(image_path, model, processor, ocr_result)

def runDocUMind(docid,doc_label,filename, classification_threshold, idChecks, detailCheck, image_path=img, ppimages=[]):
  docType = "NON ID"
  if doc_label in ["PAN Card", "Aadhar", "Driving"]:
    docType = "ID Proof"
  response = { "docid": docid,"name":filename, "label": doc_label, "docType": docType, "uploadedDate": "26/02/2023", "status": "Auto Approved"}
  
  flags = []
  ocr_result =get_ocr_result(image_path)
  result = get_doc_class(ocr_result, image_path)
  document_class = result["class"]
  document_score = result["score"]
  if(document_class!=doc_label):
    flags.append({ "name": "Label Check", "predictedValue": document_class, "inputValue": doc_label, "status": "Not Matched","probability": document_score, "coordinates": [], "code": 404})
  elif(document_score<classification_threshold):
    temp_res = "Label Check - We are not sure if the document is of type: " + doc_label
    flags.append({ "name": temp_res, "predictedValue": "", "inputValue": "", "status": "Threshold Not Met","probability": document_score, "coordinates": [] , "code": 402})
  else:
    flags.append({ "name": "Label Check", "predictedValue": document_class, "inputValue": doc_label, "status": "Matched","probability": document_score, "coordinates": [], "code": 200})
  

  # Second model check

  if docType == "ID Proof":
    print("Its an ID, you may run YOLO")
    yolo_results = run_yolo(image_path)

    if isinstance(yolo_results, list):
      entities_found = {"profile-image":[]}


      for i in range(len(yolo_results)):
        featureType = yolo_results[i]["name"]
        if(featureType in entities_found):
          entities_found[featureType].append(i)
        else:
          entities_found[featureType] = [i]

      for x in idChecks:
        if x not in entities_found:
          flags.append({ "name": x, "predictedValue": "", "inputValue": "", "status": "Not Found","probability": "", "coordinates": [], "code": 404})
        else:
          for feature in entities_found[x]:
            currentFeature = yolo_results[feature]
            coordinates = [currentFeature["xmin"], currentFeature["ymin"], currentFeature["xmax"], currentFeature["ymax"]]
            flags.append({ "name": currentFeature["name"], "predictedValue": "", "inputValue": "", "status": "Feature Found","probability": currentFeature["confidence"], "coordinates": coordinates, "code": 200})

      # code for adding profile images to ppimages array

      for idx in entities_found["profile-image"]:
        xi = int(yolo_results[idx]["xmin"])
        yi = int(yolo_results[idx]["ymin"])
        xj = int(yolo_results[idx]["xmax"])
        yj = int(yolo_results[idx]["ymax"])
        crop_img = image_path[yi:yj, xi:xj]
        crop_img = cv2.cvtColor(crop_img, cv2.COLOR_BGR2RGB)
        ppimages.append([crop_img,filename + " " + doc_label])

  # Third model check

  for x in detailCheck:
    if len(x)>0:
      findResult = findInfo(x, ocr_result)
      print(findResult)
      if(len(findResult[0]) > 0):
        coordinates = findResult[0]["bounding_box"]
        coordinates = [ coordinates[0], coordinates[3], coordinates[2], coordinates[1]]
        flags.append({ "name": "Info Check", "predictedValue": findResult[0]["word"], "inputValue": x, "status": "Info Found","probability": (findResult[1]/len(x))*100, "coordinates": coordinates, "code": 200})
      else:
        flags.append({ "name": "Info Check", "predictedValue": "", "inputValue": x, "status": "Info Not Found","probability": "", "coordinates": [], "code": 404})

    
  # img = cv2.imread(image_path)
  lap_var = cv2.Laplacian(image_path, cv2.CV_64F).var()
  if lap_var < 100:
      print('Poor Image Quality (Blurry)')
      flags.append({ "name": "Image is Blur", "predictedValue": "", "inputValue": "", "status": "Image is poor in quality","probability": "", "coordinates": [], "code": 402})
  else:
      print('Good Image Quality')
  
  response["features"] = flags


  # Profile Image Checks

  return response

def multiDoc(documents):
  result = []
  ppimages = []
  for document in documents:
    docid = document.docid
    filename = document.filename
    doclabel = document.payload.doclabel
    classificationThreshold = document.payload.classificationThreshold
    idChecks = document.payload.idChecks
    detailCheck = document.payload.detailCheck
    fileObject = converB64tofile(document.fileb64)
    data = runDocUMind(docid,doclabel,filename, classificationThreshold, idChecks, detailCheck, fileObject, ppimages)
    result.append(data)
  if len(ppimages)>=2:
    result.append(clusterProfiles(ppimages))
  else:
     result.append([])
  return result

def findInfo(s1, ocrResults = []):
  s1 = s1.lower()
  s2 = s1.replace(" ", "").lower()
  res = ["", 0.9]
  for tokens in ocrResults:
    text = tokens["word"].lower()
    if text.find(s1) >= 0 or text.find(s2) >= 0 :
      return [tokens, len(s1)]
  return res

# def longestCommonSubsequence(text1: str, text2: str) -> int:
#         dp = [[0] * (len(text2) + 1) for _ in range(len(text1) + 1)]
#         for i, c in enumerate(text1):
#             for j, d in enumerate(text2):
#                 dp[i + 1][j + 1] = 1 + dp[i][j] if c == d else max(dp[i][j + 1], dp[i + 1][j])
#         return dp[-1][-1]

def clusterProfiles(ppimages):
  # with open('dummyimg.txt', 'r') as file:
  #   file_contents = file.read()
  # file_contents = converB64tofile(file_contents)
  embeddings = []
  # ppimages.append([file_contents,-1])
  imglist = []
  for img_idx in range(len(ppimages)):
      img = ppimages[img_idx]
      try:
          embedding = DeepFace.represent(img[0], model_name='Facenet512')
          embeddings.append(embedding[0]['embedding'])
          imglist.append(img)
      except:
          # ppimages.pop(img_idx)
          print(f"Error extracting embedding for {img_idx}")

  embeddings = np.array(embeddings)


  # scores = []
  # for k in range(2, len(imglist)):
  #     kmeans = KMeans(n_clusters=k)
  #     kmeans.fit(embeddings)
  #     score = silhouette_score(embeddings, kmeans.labels_)  
  #     scores.append(score)

  # num_clusters = np.argmax(scores) + 2

  # kmeans = KMeans(n_clusters=num_clusters)
  # kmeans.fit(embeddings)



  for i in range(len(imglist)):
    arr = imglist[i][0]  
    retval, buffer = cv2.imencode('.jpg', arr)
    img_base64 = base64.b64encode(buffer).decode('utf-8')
    imglist[i][0] = img_base64

  distances = pairwise_distances(embeddings, metric='euclidean')

  dbscan = DBSCAN(eps=19, min_samples=1, metric='precomputed')

  dbscan.fit(distances)

  labels = dbscan.labels_
  num_clusters = len(set(labels)) - (1 if -1 in labels else 0)
  image_clusters = {i: [] for i in range(num_clusters)}
  for i, label in enumerate(labels):
      image_clusters[label].append(imglist[i])
    
  # print(imglist)

  result = []

  print('==============================================================================')
  for key in image_clusters:
    for x in image_clusters[key]:
      print(key , x[1])
  print('==============================================================================')

  for key in image_clusters:
     if image_clusters[key][0][1] != -1:
        result.append(image_clusters[key])

  return result
