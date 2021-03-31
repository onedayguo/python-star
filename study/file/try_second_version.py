import pandas as pd
import xlsxwriter

#读出违禁艺人名单并且存入list-->ban_Name
df_xls_ban = pd.read_excel('banactor.xlsx').values.tolist()
ban_Name = []
for n in df_xls_ban:
    ban_Name.append(n[1])
#读出需要筛选文件的director和actor项并且存入list-->select_actor
df_xls_select = pd.read_excel('test.xlsx').values.tolist()
select_actor = []
for s in df_xls_select:
    select_actor.append(s[1])
#匹配两个list将含有违规艺人的字符标红
#list转str
str_select_actor = ",".join(select_actor)
select_file = str_select_actor
ban_word = ban_Name
ban_find = []
new_word = select_file
for item in ban_word:
    if str_select_actor.count(item) > 0:
        ban_find.append(item+':'+str(str_select_actor.count(item))+'次')
        new_word = new_word.replace(item, '\033[1;31m'+item+'\033[0m')
# print('发现违禁艺人如下：')
# for item in ban_find:
#     print(item)
# print('违禁艺人名字已标红：'+new_word)
#标红写回excel
new_list = new_word.split(",")
#print(new_list)
workbook = xlsxwriter.Workbook('test1.xlsx')
worksheet = workbook.add_worksheet()
for i in (1, len(new_list) + 1):
    worksheet.write(i, new_list[i+1])
workbook.close()