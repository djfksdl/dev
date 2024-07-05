# example > person_detection > demo_analysis.py 붙여서 setp정리하기
# pip install onnxruntime 이후 실행하면 오류남 -> 터미널에서 오류 난줄에 84line어쩌고가 있음. 그거 컨트롤로 클릭해서 들어가서 init뒤에 _ 붙여주기 *2 -> 다시 실행하면 jpg들어가면 네모 생겨있음.

# STEP 1
import cv2
import numpy as np
import insightface
from insightface.app import FaceAnalysis

# STEP 2 -> 어디에 다운받아지냐면 컨트롤로 들어가면 있음
app = FaceAnalysis()
app.prepare(ctx_id=0, det_size=(640,640))

# STEP 3
from insightface.data import get_image as ins_get_image
img = ins_get_image('t1')

# STEP 4 - 4개 task를 다 돌린 값이 들어와있다.
faces = app.get(img)
assert len(faces)==6
print(faces[0])

# STEP 5
rimg = app.draw_on(img, faces)
cv2.imwrite("./t1_output.jpg", rimg)
# then print all-to-all face similarity
feats = []
for face in faces:
    feats.append(face.normed_embedding)
feats = np.array(feats, dtype=np.float32)
sims = np.dot(feats, feats.T)
print(sims)

# tlf