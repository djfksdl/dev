# 문장에 대해서 긍정/부정 
# STET1 - 추론기를 만들기위한 패키지를 가져옴
from transformers import pipeline

# STET2 파이프라인이라는 추론기 만들기. 추론기에는 모델정보가 들어가야함. 어떤 테스크의 어떤 모델을 쓸거라고 알려줘야함. 
# model 누가만든/모델
# classifier = pipeline("sentiment-analysis", model="stevhliu/my_awesome_model")
classifier = pipeline("sentiment-analysis", model="snunlp/KR-FinBert-SC")


# STET3 - 추론시킬 데이터 가져오기
# text = "This was a masterpiece. Not completely faithful to the books, but enthralling from beginning to end. Might be my favorite of the three."
text = "샤오미의 폴더블 폰의 점유율이 삼성전자 보다 높아졌다" # 삼성에서는 부정인데 샤오미입장에서는 긍정/ 이런게 어렵다. 비정형
# 긍정인데 삼전의 입장에서 보려면 qa도 돌리면 됨
# lnm의 복합추론을 기반으로 해서 한번에 할 수 있어서 핫하다. 강아지 고양이 있으면 고양이 라고할수도있고 강아지라고할 수 있는데 비젼에서는.그리고 줄여나가야하는데 

# STET4 - 추론
result = classifier(text)

# STET5 - 추론 결과
print(result)


# 이후에 실행시키면 자동으로 모델을 받아줌.
