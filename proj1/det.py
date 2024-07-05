### Visualization utilities 
import cv2
import numpy as np

MARGIN = 10  # pixels
ROW_SIZE = 10  # pixels
FONT_SIZE = 1
FONT_THICKNESS = 1
TEXT_COLOR = (255, 0, 0)  # red


def visualize(
    image,
    detection_result
) -> np.ndarray:
  """Draws bounding boxes on the input image and return it.
  Args:
    image: The input RGB image.
    detection_result: The list of all "Detection" entities to be visualize.
  Returns:
    Image with bounding boxes.
  """
  for detection in detection_result.detections:
    # Draw bounding_box
    bbox = detection.bounding_box
    start_point = bbox.origin_x, bbox.origin_y
    end_point = bbox.origin_x + bbox.width, bbox.origin_y + bbox.height
    cv2.rectangle(image, start_point, end_point, TEXT_COLOR, 3)

    # Draw label and score
    category = detection.categories[0]
    category_name = category.category_name
    probability = round(category.score, 2)
    result_text = category_name + ' (' + str(probability) + ')'
    text_location = (MARGIN + bbox.origin_x,
                     MARGIN + ROW_SIZE + bbox.origin_y)
    cv2.putText(image, result_text, text_location, cv2.FONT_HERSHEY_PLAIN,
                FONT_SIZE, TEXT_COLOR, FONT_THICKNESS)

  return image

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

# STEP 3: Load the input image.
image = mp.Image.create_from_file("cat_and_dog.jpg") # 이미지 넣기

# STEP 4: Detect objects in the input image.
detection_result = detector.detect(image) # 추론한다

# STEP 5: Process the detection result. In this case, visualize it. : 결과 보여주기
image_copy = np.copy(image.numpy_view())
annotated_image = visualize(image_copy, detection_result) # detection_result를 넣으면 어노태이션을 그려줌
rgb_annotated_image = cv2.cvtColor(annotated_image, cv2.COLOR_BGR2RGB) # rgb-> bgr을 쓰기때문에 바꿔주는거임
# cv2_imshow(rgb_annotated_image) 
cv2.imshow("test", rgb_annotated_image)
cv2.waitKey(0)