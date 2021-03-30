import pandas as pd

banFile = pd.read_excel('banactor.xlsx')
banNameData = banFile.values.tolist()
banNames = []
for banName in banNameData:
    banNames.append(banName[1])
# print(result)
# for actor_name in result[0:]:
# print(actor_name)

filmFile = pd.read_csv('fee_film_all.csv', header=0, index_col='album_id')
directorAndActor = filmFile.loc[:, ['director', 'actor']]
print(directorAndActor)

sensitive_find = []
for name in banNames:
    if directorAndActor.count(name) > 0:
        sensitive_find.append(name + ':' + str(directorAndActor.count(name)) + '次')
        directorAndActor = directorAndActor.replace(name, ' \033[1;31m' + name + '\033[0m')
print('发现敏感词如下：')
for item in sensitive_find:
    print(item)
print('敏感词位置已用星号进行标注：\n' + directorAndActor)
