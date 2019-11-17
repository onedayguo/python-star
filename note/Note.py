#  python 学习
'''
这里写笔记
int 整型
python是一门弱类型的语言，在使用数据的时候不需要声明数据类型
list 列表
tuple 元组
dict 字典
set 集合
输出
    格式化输出
    %g 输出整数或者小数
    %d 输出整数部分
    %f 输出保留六位小数    %.10f 保留小数点10位
输入
    input()
类型转换
    int()
    float()
汉子--组词--造句--作文
变量--语句--函数--类
数学运算符
    + 1.求和 2.正号 3.连接字符串
    - 1.求差 2.负号 3.
    * 1.乘积 2.重复
    / 2.求商 2.
列表中元素的排序
sort（）
    按自然顺序从小到大进行排序
    对原数据直接进行排序操作，不会生成新的备份
列表的嵌套使用
元组
    tuple
    比小时形式：（元素1，元素2....）
    不可变：存储的元素的值，不可修改，不可删除
    列表和元组之间可以相互转换
字典
    dict
    1.字典的定义形式 dict1{key1:value1,key2:value2...}
        :key具有唯一性，一般情况下使用字符串表示
    2.元素的访问
    3.元素的常用方法
    4.元素的遍历

函数
    单独的一个功能，经常使用，将实现该功能的代码放在一起，起个别名

'''

import numpy as np
# 定义数据集
data_set = np.array([1, 2, 5, 9, 7])
print(f'data_set={data_set}')
# 均值
mean = np.mean(data_set)
print(f'mean of data_set = {mean}')
# 方差 & 标准差
variance = np.var(data_set)
standard_deviation = np.std(data_set)
print(f'variance of data_set = {variance}')
print(f'standard deviation of data_set = {standard_deviation}')

# 样本方差 & 样本标准差,ddof代表无偏差
variance1 = np.var(data_set, ddof=1)
standard_deviation1 = np.std(data_set,ddof=1)
print(f'variance1 of data_set = {variance1}')
print(f'standard deviation of data_set = {standard_deviation1}')






















