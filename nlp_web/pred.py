import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AdamW

def prediction(text):
  model_name = "beomi/kcbert-base"
  model_save_path = "/home/t24117/nlp/model/kcbert_korean_emotion_classifier.pth"
  tokenizer = AutoTokenizer.from_pretrained(model_name)
  # 모델 아키텍처 생성
  loaded_model = AutoModelForSequenceClassification.from_pretrained("beomi/kcbert-base", num_labels=11)

  # 저장된 가중치 불러오기
  loaded_model.load_state_dict(torch.load(model_save_path))

  # 모델을 평가 모드로 설정
  loaded_model.eval()

  def valid_label(label):
    if label == 0:
      return '여성/가족'
    elif label == 1:
      return '남성'
    elif label == 2:
      return '성소수자'
    elif label == 3:
      return '인종/국적'
    elif label == 4:
      return '연령'
    elif label == 5:
      return '지역'
    elif label == 6:
      return '종교'
    elif label == 7:
      return '기타 혐오'
    elif label == 8:
      return '악플/욕설'
    elif label == 9:
      return '개인지칭'
    else:
      return '일반문장'
  input_encodings = tokenizer(text, padding=True, truncation=True, return_tensors="pt")

  # 모델에 입력 데이터 전달
  with torch.no_grad():
      output = loaded_model(**input_encodings)

  # 예측 결과 확인
  logits = output.logits
  predicted_labels = logits.argmax(dim=1)
  return valid_label(predicted_labels)