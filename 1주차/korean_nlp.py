from konlpy.tag import Hannanum

hannanum = Hannanum()

hannanum.morphs  # 형태소 분석

print("형태소 분리")
print(hannanum.morphs(u'K9전차가 전방에 포를 발사하였습니다.'))

print("형태소 분리 + 형태소 태그 부착")
print(hannanum.pos(u'코로나19 바이러스 확산이 진정 국면으로 접어들고 있습니다.'))

print("명사만 추출")
print(hannanum.nouns(u'코로나19 바이러스 확산이 진정 국면으로 접어들고 있습니다.'))
