# example > person_detection > demo_analysis.py 붙여서 setp정리하기

# STEP 1
import cv2
import numpy as np
import insightface
from insightface.app import FaceAnalysis

# STEP 2
app = FaceAnalysis()
app.prepare(ctx_id=0, det_size=(640,640))

# STEP 3
from insightface.data import get_image as ins_get_image
img = ins_get_image('t1')

# STEP 4
faces = app.get(img)
assert len(faces)==6

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