# imports for yolo
import torch

# imports for doc classification
from doc_classification_test import predict_document_image
from transformers import LayoutLMv3FeatureExtractor, LayoutLMv3TokenizerFast, LayoutLMv3Processor, LayoutLMv3ForSequenceClassification

# imports for paddle ocr
import os
import json
from paddleocr import PaddleOCR
os.environ["KMP_DUPLICATE_LIB_OK"]="True"

ocr = PaddleOCR(use_angle_cls=True, lang='en')

yolo_model = torch.hub.load('ultralytics/yolov5', 'custom', path='models/yolo.pt')

feature_extractor = LayoutLMv3FeatureExtractor(apply_ocr=False)
tokenizer = LayoutLMv3TokenizerFast.from_pretrained("models/layoutlmv3-base", local_files_only=True)
processor = LayoutLMv3Processor(feature_extractor, tokenizer)
model = LayoutLMv3ForSequenceClassification.from_pretrained("models/layoutlmv3-2", local_files_only=True)
model = model.eval().to("cpu")

image_path = 'test/test4.jpg'

def run_yolo():
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

def get_ocr_result():
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

def get_doc_class():
  ocr_result =get_ocr_result()
  print(ocr_result)
  return predict_document_image(image_path, model, processor, ocr_result)

def runDocUMind(doc_label, classification_threshold, idChecks):
  flags = []
  result = get_doc_class()
  document_class = result["class"]
  document_score = result["score"]
  if(document_class!=doc_label):
    flags.append({ "name": "Label doesn't match the document type", "expectedValue": document_class, "receivedValue": doc_label, "status": "Not Matched","probability": document_score, "coordinates": []})
  elif(document_score<classification_threshold):
    temp_res = "We are not sure if the document is of type: " + doc_label
    flags.append({ "name": temp_res, "expectedValue": "", "receivedValue": "", "status": "Threshold Not Met","probability": document_score, "coordinates": []})
  
  id_types = {"Driving", "PAN Card", "Aadhar"}

  if doc_label in id_types:
    print("Its an ID, you may run YOLO")
    yolo_results = run_yolo()

    if not isinstance(yolo_results, list):
      return flags

    entities_found = {}
    for i in range(len(yolo_results)):
      featureType = yolo_results[i]["name"]
      if(featureType in entities_found):
        entities_found[featureType].append(i)
      else:
        entities_found[featureType] = [i]

    for x in idChecks:
      if x not in entities_found:
        flags.append({ "name": x, "expectedValue": "", "receivedValue": "", "status": "Not Found","probability": "", "coordinates": []})
      else:
        for feature in entities_found[x]:
          currentFeature = yolo_results[feature]
          coordinates = [currentFeature["xmin"], currentFeature["ymin"], currentFeature["xmax"], currentFeature["ymax"]]
          flags.append({ "name": currentFeature["name"], "expectedValue": "", "receivedValue": "", "status": "Feature Found","probability": currentFeature["confidence"], "coordinates": coordinates})

  return flags
