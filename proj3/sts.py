# hugging에서 senetence transformer 가 있다.
# step1
from sentence_transformers import SentenceTransformer

# step2
model = SentenceTransformer("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")

# step3
sentences1 =  "탕이 시원하다"
sentences2 =  "물 온도가 뜨겁다"

# step4
embedding1 = model.encode(sentences1)
embedding2 = model.encode(sentences2)
print(embedding1.shape)
# [3, 384]

# step5
similarities = model.similarity(embedding1, embedding2)
print(similarities)
# tensor([[1.0000, 0.6660, 0.1046],
#         [0.6660, 1.0000, 0.1411],
#         [0.1046, 0.1411, 1.0000]])