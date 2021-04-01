import xlsxwriter, xlrd
import re

films = xlrd.open_workbook('test.xlsx')
tableFilm = films.sheet_by_index(0)
rowNum = tableFilm.nrows
colNum = tableFilm.ncols
actors = xlrd.open_workbook('banactor.xlsx')
tableActor = actors.sheet_by_index(0)
actor_names = tableActor.col_values(1, 1)

workbook = xlsxwriter.Workbook('write_xlsx1.xlsx')
bold_red = workbook.add_format({'bold': True, 'color': 'red'})
worksheet = workbook.add_worksheet(name='荣风鑫')

for row in range(1, rowNum):
    matchDirector = 0
    matchActor = 0
    director = ''.join(tableFilm.row_values(row, 0, 1))
    actor = ''.join(tableFilm.row_values(row, 1, 2))
    for name in actor_names:
        matchDirector = max(director.count(name), matchDirector)
        matchActor = max(actor.count(name), matchActor)
    if matchDirector > 0:
        print('match director', director)
        worksheet.write(row, 0, director, bold_red)
    else:
        worksheet.write(row, 0, director)
    if matchActor > 0:
        print('match actor', actor)
        worksheet.write(row, 1, actor, bold_red)
    else:
        worksheet.write(row, 1, actor)
workbook.close()
