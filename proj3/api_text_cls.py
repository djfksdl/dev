from fastapi import FastAPI, Form

# STET1 - 추론기를 만들기위한 패키지를 가져옴
from transformers import pipeline

# STET2 파이프라인이라는 추론기 만들기. 
classifier = pipeline("sentiment-analysis", model="snunlp/KR-FinBert-SC")

app = FastAPI()

@app.post("/textClassification/")
async def login(text: str = Form()):

    # STET3 - 추론시킬 데이터 가져오기
    text = "샤오미의 폴더블 폰의 점유율이 삼성전자 보다 높아졌다" # 삼성에서는 부정인데 샤오미입장에서는 긍정/ 이런게 어렵다. 비정형

    # STET4 - 추론
    result = classifier(text)

    # STET5 - 추론 결과
    print(result)
    return {"result": text}