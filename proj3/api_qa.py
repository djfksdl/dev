from fastapi import FastAPI, Form

# STET1
from transformers import pipeline
 
# STET2 
question_answerer = pipeline("question-answering", model="bespin-global/klue-bert-base-aihub-mrc")


app = FastAPI()


@app.post("/qa/")
async def login(text: str = Form()):

    # STET3 
    context = '''애플 M2(Apple M2)는 애플이 설계한 중앙 처리 장치(CPU)와 그래픽 처리 장치(GPU)의 ARM 기반 시스템이다. 
    인텔 코어(Intel Core)에서 맥킨토시 컴퓨터용으로 설계된 2세대 ARM 아키텍처이다. 애플은 2022년 6월 6일 WWDC에서 맥북 에어, 13인치 맥북 프로와 함께 M2를 발표했다. 
    애플 M1의 후속작이다. M2는 TSMC의 '향상된 5나노미터 기술' N5P 공정으로 만들어졌으며, 이전 세대 M1보다 25% 증가한 200억개의 트랜지스터를 포함하고 있으며, 최대 24기가바이트의 RAM과 2테라바이트의 저장공간으로 구성할 수 있다. 
    8개의 CPU 코어(성능 4개, 효율성 4개)와 최대 10개의 GPU 코어를 가지고 있다. M2는 또한 메모리 대역폭을 100 GB/s로 증가시킨다. 
    애플은 기존 M1 대비 CPU가 최대 18%, GPU가 최대 35% 향상됐다고 주장하고 있으며,[1] 블룸버그통신은 M2맥스에 CPU 코어 12개와 GPU 코어 38개가 포함될 것이라고 보도했다.'''
    question = "m2가 m1에 비해 얼마나 좋아졌어?"

    # STET4
    result =question_answerer(question=question, context=context)

    # STET5 
    print(result)

    return {"text": result}