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
# from insightface.data import get_image as ins_get_image 샘플구간임. 안쓸거
# img = ins_get_image('t1')
img1 = cv2.imread("go.jpg")
img2 = cv2.imread("go2.jpg")

# STEP 4 - 4개 task를 다 돌린 값이 들어와있다.
faces1 = app.get(img1)
faces2 = app.get(img2)
assert len(faces1)==1
assert len(faces2)==1
# print(faces1[0])

# STEP 5
rimg = app.draw_on(img1, faces1)
cv2.imwrite("./go_output.jpg", rimg)
rimg = app.draw_on(img2, faces2)
cv2.imwrite("./go2_output.jpg", rimg)

# # then print all-to-all face similarity
# feats = []
# for face in faces:
#     feats.append(face.normed_embedding)

# 여기는 거의 모든 임베딩에서 쓰인다.! 구조 외우기
feat1 = np.array(faces1[0].normed_embedding , dtype=np.float32) # np.dot 은 2개의 임배딩 곱?을 행렬곱-> 행렬곱은 코사인 유사도를 나타낸다. 
feat2 = np.array(faces2[0].normed_embedding , dtype=np.float32) 
# 유사도 비교 부분 행렬곱 진행 코사인 시밀러리티로 2개의 데이터간의 유사도를 측정할수 있습니다.
# -1 ~ 1 사이에 값이 나온다.
# 동일인기준으로 threshold를 0.4정도로 잡는다.
# 얼굴특징인 feats로 나타낸다.
sim = np.dot(feat1, feat2.T) # 뒤에 있는 행렬은 세로 배열해야해서 .T로 트랜스폼 해준다. 이래서 나온 값이 '유사도'
print(sim)

# 얼굴을 찍었을때 5개의 값이 추출되고 내가 쓸 수 있다. 
