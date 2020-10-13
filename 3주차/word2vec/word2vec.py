from konlpy.tag import Okt

from gensim.models import Word2Vec

okt = Okt()

data = []

with open('wiki_preprocessed.txt', 'r', encoding='utf-8') as fr:
    for line in fr.readlines():
        morphs = okt.pos(line.strip())
        data.append([morph[0] for morph in morphs])

print('Word2vec training start')
model = Word2Vec(data, size=100, window=5, min_count=5, workers=4, sg=1) #size: vector size, data: should 2D list.(list of sentences)

model.save('word2vec_morph.model')

print('done')
