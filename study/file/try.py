import pandas as pd
import re
#读出违禁艺人名单并且存入list-->ban_actorName
from pandas import DataFrame

df_xls1 = pd.read_excel('banactor.xlsx')
ban_actorName =df_xls1.艺人.values
#读出需要筛选文件的director和actor项
df_xls2 = pd.read_excel('test.xlsx')

for item in ban_actorName:
    df_xls2.replace(item, '荣凤鑫',inplace = True)
print('发现违禁艺人如下：')

DataFrame.to_excel(df_xls2,'out1.xlsx')