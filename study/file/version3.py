import re
import pandas as pd
from xlsxwriter.workbook import Workbook
#创建一个工作薄用于写入新的单元格数据
workbook = Workbook('test1.xlsx')
worksheet = workbook.add_worksheet()
red = workbook.add_format({'color': 'red'})
#读出违禁艺人名单并且存入list-->ban_Name
df_xls_ban = pd.read_excel('banactor.xlsx').values.tolist()
ban_Name = []
for n in df_xls_ban:
    ban_Name.append(n[1])
#读出需要筛选文件的actor项并且存入list-->select_actor
df_xls_select = pd.read_excel('test.xlsx').values.tolist()
select_list = []
for s in df_xls_select:
    select_list.append(s[1])

for item in ban_Name:
    final = []
    for base in select_list:
        matchObj = re.search(item, base, re.M | re.I)
        if matchObj:
            final.extend((red, base))
        else:
            final.append(base)
    #worksheet.write(row_num, 1, final)
print(final)
#workbook.close()