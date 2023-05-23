import numpy as np
import pandas as pd
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori, association_rules
import matplotlib.pyplot as plt
from collections import defaultdict


# 读入数据并打印信息
def get_data(data_path):
    # 读入原始数据
    data = open(data_path).readlines()
    A_count = 0  # 记录属性数
    C_count = 0  # 记录case数
    V_count = 0  # 记录投票数
    Attributes = {}  # 属性与编号映射字典
    Cases = []  # 案例号
    Votes = []  # 各案例投票

    # 统计投票数据
    for i in range(len(data)):
        item = data[i].split(',')
        if item[0] == 'A':
            A_count += 1
            Attributes[int(item[1])] = item[3]
        if item[0] == 'C':
            C_count += 1
            Cases.append(int(item[1].split('"')[1]))
            vote = []
            while (i < len(data) - 1):
                i += 1
                if data[i][0] != 'V':
                    break
                V_count += 1
                vote.append(int(data[i].split(',')[1]))
            Votes.append(vote)
    # 打印基本信息
    print('Attribute count:', A_count)
    print('Case count:', C_count)
    print('Vote count:', V_count)
    # 返回属性字典和投票数据
    return Attributes, Votes


# 训练集和测试集数据路径
train_path = 'anonymous-msweb.data'
test_path = 'anonymous-msweb.test'

# 读入原始数据
print("Trainset:")
Attributes, train_Votes = get_data(train_path)
print("Testset:")
_, test_Votes = get_data(test_path)

# 合并训练集数据集
Votes = [*train_Votes, *test_Votes]

# 统计投票属性种数
vote_unique = set()
for Vote in Votes:
    for item in Vote:
        vote_unique.add(item)
print('所有投票的属性有 %d 种' % len(vote_unique))
# 数据可视化
votes_every_attr = defaultdict(int)  # 每个属性获得的投票数
votes_every_case = []  # 每个用户的投票数
# 遍历数据项
for item in Votes:
    votes_every_case.append(len(item))
    for attr in item:
        votes_every_attr[attr] += 1
# 画出不同投票数的用户的分布
lens, counts = np.unique(np.array(votes_every_case), return_counts=True)
plt.bar(lens, counts / len(Votes) * 100)
plt.title('Votes every case')
plt.xlabel('Length of votes')
plt.ylabel('Cases Pencentage %')
# 画出不同属性的投票数分布
plt.figure(2)
attrs = list(votes_every_attr.keys())
vote_counts = list(votes_every_attr.values())
plt.bar(attrs, np.array(vote_counts) / len(Votes) * 100)
plt.title('Votes every attribute')
plt.xlabel('Attribute Number')
plt.ylabel('Cases Pencentage %')
plt.show(block=True)

TE = TransactionEncoder()  # onehot编码器
one_hot_records = TE.fit_transform(Votes)  # 对投票数进行编码
# 将编码后的数据转为数据帧并加上属性列名
columns = list()
for item in TE.columns_:
    columns.append(Attributes[item])
Votes_df = pd.DataFrame(one_hot_records)
Votes_df.columns = columns

# 计算得出频繁项集
min_support = 1000 / len(Votes)  # 出现500次就定义为频繁
freq_items = apriori(Votes_df, min_support=min_support, use_colnames=True)
print(freq_items)

# 关联规则发掘
Votes_df_1 = pd.DataFrame(one_hot_records)
Votes_df_1.columns = TE.columns_  # 为了直观显示规则，用属性编号代替属性
min_support = 50 / len(Votes)  # 降低置信度阈值
freq_items_1 = apriori(Votes_df_1, min_support=min_support, use_colnames=True)
association_rule = association_rules(freq_items_1, metric='confidence', min_threshold=0.95)  # 关联规则发掘，置信度阈值为0.95
# 打印关联规则
for i in range(len(association_rule)):
    rule = association_rule.iloc[i, :]
    antecedents = list(rule[0])
    consequents = list(rule[1])
    confidence = rule['confidence']
    lift = rule['lift']
    print('(', end='')
    for j, item in enumerate(antecedents):
        print(item, end='')
        if j != len(antecedents) - 1:
            print(' ,', end='')
        else:
            print(')', end='')
    print(" -> ", end='')
    print('(', end='')
    for j, item in enumerate(consequents):
        print(item, end='')
        if j != len(consequents) - 1:
            print(' ,', end='')
        else:
            print(')', end='')
    print(' confifence: %.3f' % confidence, end='')
    print(' lift: %.3f' % lift)
