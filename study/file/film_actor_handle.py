import pandas as pd

banFile = pd.read_excel('banactor.xlsx')
banNames = banFile.艺人.values

filmFile = pd.read_csv('fee_film_all.csv', header=0, index_col='album_id')
directors = filmFile.director.values
actors = filmFile.actor.values

# sensitive_find = []
# for director in  directors:


# for name in banNames:
#     count = 0
#     if directorAndActor.count(name) > 0:
#         sensitive_find.append(name + ':' + str(directorAndActor.count(name)) + '次')
#         directorAndActor = directorAndActor.replace(name, ' \033[1;31m' + name + '\033[0m')
# print('发现敏感词如下：')
# for item in sensitive_find:
#     print(item)
# print('敏感词位置已用星号进行标注：\n' + directorAndActor)
