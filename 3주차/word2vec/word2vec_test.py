import numpy as np
from gensim.models import Word2Vec

model = Word2Vec.load('word2vec.model')

# print(model.wv.vocab.keys())

def is_oov(token):
    if token not in model.wv.vocab.keys():
        return True
    else:
        return False

# # OOV 테스트
print(is_oov('대한민국'))
print(is_oov('서울'))
print(is_oov('일본'))
print(is_oov('도쿄'))

# # 유사도 테스트
print(model.similarity("대한민국", "일본"))
print(model.similarity("서울", "도쿄"))
print(model.similarity("강아지", "고양이"))
print(model.similarity("강아지", "서울"))

# most similar
print(model.most_similar("강아지"))

# analogy test
print(model.most_similar(positive=["대한민국", "일본"], negative=["서울"]))