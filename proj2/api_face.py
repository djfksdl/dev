from fastapi import FastAPI, File, UploadFile

# STEP 1
import cv2
import numpy as np
import insightface
from insightface.app import FaceAnalysis

# STEP 2 
face = FaceAnalysis() #app으로 하면 밑에 값이랑 겹쳐서 오류날 수 있어서 다른 변수로 바꿔주기
face.prepare(ctx_id=0, det_size=(640,640))

target_face = []

app = FastAPI()


@app.post("/registFace/")
async def registFace(file: UploadFile):
    content = await file.read()

    # STEP 3
    # img = cv2.imread("iu1.jpg")
    # --> buf = file.open("iu1.jpg")
    # --> img = cv2.imdecode(buf)
    nparr = np.fromstring(content, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
  
    # STEP 4
    faces1 = face.get(img)
    assert len(faces1)==1

    # STEP 5
    target_face.append(np.array(faces1[0].normed_embedding, dtype=np.float32))
    print(target_face)
    return {"result":len(faces1)}
    # 여기는 왜 normed냐? 파이썬이 제공하는 자체 데이터 타입을 써야함. 임베딩값도 내보낼 수 있다. 

@app.post("/compareFace/")
async def compareFace(file: UploadFile):
    content = await file.read()
    
    # STEP 3
    # img = cv2.imread("iu1.jpg")
    # --> buf = file.open("iu1.jpg")
    # --> img = cv2.imdecode(buf)
    nparr = np.fromstring(content, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
  
    # STEP 4
    faces1 = face.get(img)
    assert len(faces1)==1

    # STEP 5
    test_face = np.array(faces1[0].normed_embedding, dtype=np.float32)

    sim = np.dot(target_face[0], test_face.T)
    return {"result":sim.item()}



    
