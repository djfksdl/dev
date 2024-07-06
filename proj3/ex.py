# STET1 - 추론기를 만들기위한 패키지를 가져옴
from transformers import pipeline

# STET2 파이프라인이라는 추론기 만들기. 추론기에는 모델정보가 들어가야함. 어떤 테스크의 어떤 모델을 쓸거라고 알려줘야함. 
summarizer = pipeline("summarization", model="csebuetnlp/mT5_multilingual_XLSum")

# STET3 - 추론시킬 데이터 가져오기
text = '''
6일 오전 9시 40분쯤 서울역 인근에 있는 한국철도공사(코레일) 서울본부 지하 전산실에서 원인을 알 수 없는 화재가 발생했다.
연합뉴스에 따르면 불은 덕트(배관)를 타고 올라 상층부까지 번졌다가 발생 1시간 20여분 뒤인 오전 11시 3분쯤 큰 불길이 잡혀 초진이 완료됐다.
신고를 받고 현장에 도착한 소방당국은 차량 46대와 인력 177명을 동원해 진화 작업을 벌였다. 현재까지 인명피해는 없는 것으로 파악됐다.화재 여파로 열차 이용에 불편이 빚어졌다. 열차는 정상 운행 중이지만 전산 장애가 발생해 역창구에서 승차권 조회 및 발매, 환불 작업이 정상적으로 이뤄지지 못하고 있는 상황이다. 코레일 측은 역 창구 대신 코레일톡(모바일앱)을 이용해달라고 당부했다.
'''

# STET4 - 추론
summary_result = summarizer(text)

# STET5 - 추론 결과
print(summary_result)

# 번역
translator = pipeline("translation_ko_to_en", model="facebook/nllb-200-distilled-600M")
print(translator(summary_result[0]['summary_text']))
