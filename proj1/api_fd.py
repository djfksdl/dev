from fastapi import FastAPI, File, UploadFile

# STEP 1: Import the necessary modules.
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# STEP 2: Create an FaceDetector object.
base_options = python.BaseOptions(model_asset_path='models\\fd\\blaze_face_short_range.tflite')
options = vision.FaceDetectorOptions(base_options=base_options)
detector = vision.FaceDetector.create_from_options(options)

app = FastAPI()
from PIL import Image
import io

@app.post("/uploadfile/")
async def create_upload_file(file: UploadFile):

    content = await file.read()

    # STEP 3: Load the input image.
    binary = io.BytesIO(content) # 1)과정
    pil_img = Image.open(binary) # 2)과정
    image = mp.Image(image_format=mp.ImageFormat.SRGB, data=np.asarray(pil_img))

    # STEP 4: Detect faces in the input image.
    detection_result = detector.detect(image)

    # print(detection_result)
    counts = len(detection_result.detections)

    isExist = False
    if counts > 0:
        isExist = True

    # STEP 5: Process the detection result. In this case, visualize it.
    # image_copy = np.copy(image.numpy_view())
    # annotated_image = visualize(image_copy, detection_result)
    # rgb_annotated_image = cv2.cvtColor(annotated_image, cv2.COLOR_BGR2RGB)
    # cv2.imshow("test",rgb_annotated_image)
    # cv2.waitKey(0)
    return {"isExist": isExist,
            "counts": counts
            }