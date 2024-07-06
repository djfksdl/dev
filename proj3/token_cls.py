# STET1
from transformers import pipeline

# STET2 
classifier = pipeline("ner", model="KoichiYasuoka/roberta-large-korean-upos")

# STET3 
text = "홍시 맛이 나서 홍시라 생각한다."

# STET4
result = classifier(text)

# STET5 
print(result)