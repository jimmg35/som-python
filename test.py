import pandas as pd
import json

data = pd.read_csv(r'./cluster-result/1703688301_som/cluster.csv', encoding="utf-8")

categories = data['cluster'].unique()
group = {}

for category in categories:
  x1 = data[data['cluster'] == category]['x1'].tolist()
  x2 = data[data['cluster'] == category]['x2'].tolist()
  combined = list(map(list, zip(x1, x2)))
  group[int(category)]=combined


with open('x1_x2.json', 'w') as json_file:
  json.dump(group, json_file, indent=2)  # indent参数用于美化输出，可选
aa = {0:[1,2,3],1:[8,9,4], 2:[8,8,8,8,8]}
print(len(aa))