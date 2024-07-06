# from typing import Union

# from fastapi import FastAPI

# app = FastAPI()


# @app.get("/")
# def read_root():
#     return {"Hello": "World"}


# @app.get("/items/{item_id}")
# def read_item(item_id: int, q: Union[str, None] = None):
#     return {"item_id": item_id, "q": q}


# # uvicorn api_cls:app 이렇게 실행한다.
# # 유비콘이 톰캣같은 녀석이다. 
# # 서버뒤에 docs를 치면 쓰레드로 넘어간다. 거기서 id치고 

from fastapi import FastAPI, File, UploadFile

# STEP 1
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python.components import processors
from mediapipe.tasks.python import vision

# STEP 2 - 추론기를 미리 띄워놓고 서버연동해서 계속 불러다 씀.
base_options = python.BaseOptions(model_asset_path='models/cls/efficientnet_lite2.tflite') # base_option: 기본 옵션: 모델 경로를 넣어줘야함. -> 넣을 파일 오른쪽 상대경로 copy하고 복붙하면 됨
options = vision.ImageClassifierOptions( # 클래시피케이션에 필요한 옵션
base_options=base_options, max_results=1) # 베이스 넣어주고,  max_results= 결과가 나오는 갯수
classifier = vision.ImageClassifier.create_from_options(options)

app = FastAPI()
from PIL import Image
import numpy as np
import io

# @app.post("/files/")
# async def create_file(file: bytes = File()): # 파일 바이츠랑 업로드파일이랑은 같은게 아님... 비동기성 동작하지 않음?
#     return {"file_size": len(file)}

# fast api하려면 이게 나음
@app.post("/uploadfile/")
async def create_upload_file(file: UploadFile):

    content = await file.read() # 비동기로 컨텐츠 읽을게
    # return {"filename": file.filename,
    #         "filesize ": len(content)
    #         }
    # content -> jpg파일인데 http 통신에서는 파일이 character type왔다갔다 함.
    # content를 밑에 이미지로 바꿔줘야함
    # 1) text-> binery : io.BytesIO(text) : 바이너리가 : 기계가 이해하는 형태로 바꿈
    # 2) binery -> pil_image 파이썬 이미지로 먼저 비트맵을 만들고 mp이미지로 변환한다.(왜냐? 한번에 바뀌지 않기 때문)
    # 3) 미디어파이프이미지 형태로 바꾼다.
    

    # STEP 3 - 외우기
    binary = io.BytesIO(content) # 1)과정
    pil_img = Image.open(binary) # 2)과정
    image = mp.Image(image_format=mp.ImageFormat.SRGB, data=np.asarray(pil_img)) # 3)과정

    # STEP 4
    classification_result = classifier.classify(image) # 미디어파이프를 이용한 추론기 : classifier 

    # STEP 5
    top_category = classification_result.classifications[0].categories[0]
    result = f"{top_category.category_name} ({top_category.score:.2f})"

    return {"result": result} # 서버에 쏴줌

# 파일도 쉽게 주고받을 수 있게 해줌
# 서버 들어가서 docs들어가서 파일 올리면 결과가 나옴. 
# 그럼 이제 api서버를 통해서 파일을 통신할 수 있다