import gensim
from gensim.models.fasttext import FastText

path='wiki_preprocessed.txt'
sentences=gensim.models.word2vec.Text8Corpus(path)
model=FastText(sentences,min_count=5,size=100,window=5)
model.save('fasttext_model')

saved_model=FastText.load('fasttext_model')

word_vector=saved_model['이순신']
print(word_vector)
print(saved_model.similarity('이순신','이명박'))
print(saved_model.similarity('이순신','원군'))

print(saved_model.similarity_by_word('이순신'))
print(saved_model.similarity_by_word('조선'))

saved_model.most_similar(positive=['대한민국','도쿄'],negative=['서울'])
