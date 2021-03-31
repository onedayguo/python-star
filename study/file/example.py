# word = input("请输入含有违规词的段落：")
sensitive = ['第一','国家级','最高级','最佳','独一无二','一流','仅此一次','顶级','顶尖','尖端','极品','极佳','绝佳','绝对','终极','极致','首个','首选', '独家','首发','首次']
sensitive_find = []
new_word = '第一;开始;独家'
for item in sensitive:
    if new_word.count(item)>0:
        sensitive_find.append(item+':'+str(new_word.count(item))+'次')
        new_word = new_word.replace(item, '\033[1;31m'+item+'\033[0m')
print('发现敏感词如下：')
for item in sensitive_find:
    print(item)
print('敏感词位置已标红：'+new_word)