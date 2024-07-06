from fastapi import FastAPI, File, UploadFile

# STEP 1: Import the necessary modules. : 추론기를 만들기위해 필요한 패키지 가져오는 것
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# STEP 2: Create an ObjectDetector object. : 추론기 만들기 (베이스옵션에 모델정보 넣어주기)
base_options = python.BaseOptions(model_asset_path='models\det\efficientdet_lite0.tflite') # 경로 바꿔주기. 오른쪽버튼에서 relative copy해서 붙여넣기
options = vision.ObjectDetectorOptions(base_options=base_options,
                                       score_threshold=0.5) # 박스를 찾았을때 물체가 있을 확률값이다. 0.5가 기본이다. 그 이전값은 버린다.
detector = vision.ObjectDetector.create_from_options(options)

app = FastAPI()
from PIL import Image
import numpy as np
import io


@app.post("/uploadfile/")
async def create_upload_file(file: UploadFile):
    content = await file.read()

    # STEP 3: Load the input image.
    binary = io.BytesIO(content) # 1)과정
    pil_img = Image.open(binary) # 2)과정
    image = mp.Image(image_format=mp.ImageFormat.SRGB, data=np.asarray(pil_img)) # 3)과정


    # STEP 4: Detect objects in the input image.
    detection_result = detector.detect(image) # 추론한다

    # 터미널에 나온 값 복붙해서 분석하기 쉽게 붙여둔다.
    # DetectionResult(
    # detections=[
    #     Detection(bounding_box=BoundingBox(origin_x=72, origin_y=162, width=252, height=191), 
    #               categories=[Category(index=None, score=0.7803766131401062, display_name=None, category_name='cat')], 
    #               keypoints=[]), 
    #     Detection(bounding_box=BoundingBox(origin_x=303, origin_y=27, width=249, height=345), 
    #               categories=[Category(index=None, score=0.7627291083335876, display_name=None, category_name='dog')], 
    #               keypoints=[])
    #             ]
    # )

    counts = len(detection_result.detections)
    object_list = []
    for detection in detection_result.detections:
        object_category = detection.categories[0].category_name
        object_list.append(object_category)
    
    # print(detection_result)

    # STEP 5: Process the detection result. In this case, visualize it. : 결과 보여주기
    # image_copy = np.copy(image.numpy_view())
    # annotated_image = visualize(image_copy, detection_result) # detection_result를 넣으면 어노태이션을 그려줌
    # rgb_annotated_image = cv2.cvtColor(annotated_image, cv2.COLOR_BGR2RGB) # rgb-> bgr을 쓰기때문에 바꿔주는거임
    # # cv2_imshow(rgb_annotated_image) 
    # cv2.imshow("test", rgb_annotated_image)
    # cv2.waitKey(0)

    return {"counts": counts,
            "objet_list": object_list
            }