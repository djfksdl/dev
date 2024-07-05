# import urllib # 코드 돌려서 이미지를 다운받을거냐 아님 링크로 들어가서 이미지 다운받을거냐인데 굳이 필요없긴함. 그냥 예시 / 윈도우즈에서는 뭐 하나 수정해야함
import urllib.request # .request를 써주면 실행할때 이미지가 다운받아진다.

IMAGE_FILENAMES = ['burger.jpg', 'cat.jpg', 'cat-551554_1280.jpg','dw.jfif']

### test image 코드 긁어오기
# for name in IMAGE_FILENAMES:
#   url = f'https://storage.googleapis.com/mediapipe-tasks/image_classifier/{name}'
#   urllib.request.urlretrieve(url, name)

#   https://storage.googleapis.com/mediapipe-tasks/image_classifier/burger.jpg
#   https://storage.googleapis.com/mediapipe-tasks/image_classifier/cat.jpg

# import cv2
# # from google.colab.patches import cv2_imshow
# import math

# DESIRED_HEIGHT = 480
# DESIRED_WIDTH = 480

# def resize_and_show(image):
#   h, w = image.shape[:2]
#   if h < w:
#     img = cv2.resize(image, (DESIRED_WIDTH, math.floor(h/(w/DESIRED_WIDTH))))
#   else:
#     img = cv2.resize(image, (math.floor(w/(h/DESIRED_HEIGHT)), DESIRED_HEIGHT))
# #   cv2_imshow(img) : 화면에 이미지 보여달라는 뜻
# # space 두번 누르기 (tab X)
#   cv2.imshow("test", img) # 어떤 창의 이름을 "test"로 지정하고,  img파일을 띄울거다
#   cv2.waitKey(0) # 0을 집어 넣으면 key입력이 될때까지 무한대로 기다리기 (키는 스페이스바든 아무 입력이나 상관없음)/ 원래는 숫자만큼 기다리는건데 0은 무한대!

# # Preview the images. 우리가 다운받은 이미지가 잘 있는지 확인하기!

# images = {name: cv2.imread(name) for name in IMAGE_FILENAMES}
# for name, image in images.items():
#   print(name)
#   resize_and_show(image)


# STEP 1: Import the necessary modules. : 패키지를 가져오는것
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python.components import processors
from mediapipe.tasks.python import vision

# STEP 2: Create an ImageClassifier object. : 추론기(객체) 만드는것
base_options = python.BaseOptions(model_asset_path='models/cls/efficientnet_lite2.tflite') # base_option: 기본 옵션: 모델 경로를 넣어줘야함. -> 넣을 파일 오른쪽 상대경로 copy하고 복붙하면 됨
options = vision.ImageClassifierOptions( # 클래시피케이션에 필요한 옵션
base_options=base_options, max_results=1) # 베이스 넣어주고,  max_results= 결과가 나오는 갯수
classifier = vision.ImageClassifier.create_from_options(options)


# STEP 3: Load the input image. : 추론할 이미지(데이터)를 가져오는 것
image = mp.Image.create_from_file(IMAGE_FILENAMES[3]) # 버거를 가져옴!
# image = mp.Image.create_from_file('burger.jpg') 이렇게 써도 됨

# STEP 4: Classify the input image. : 손댈일이 없다. 가져온 데이터 추론하고 추론결과를 가져오는것! 
classification_result = classifier.classify(image)
# print(classification_result) 디버깅용

# STEP 5: Process the classification result. In this case, visualize it. : 응용! 사용자에게 어떻게 보여줄건지
top_category = classification_result.classifications[0].categories[0]
print(f"{top_category.category_name} ({top_category.score:.2f})") 
