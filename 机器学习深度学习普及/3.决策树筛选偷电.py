import pandas  as pd  #数据处理
import random #随机数
data=pd.read_excel("model.xls")
data=data.as_matrix()#表格转化数组
#random.shuffle(data)#打乱顺序
print(data) #读取数据
p=0.8 #定义概率
train=data[:int(len(data)*p),:] #80%用于训练
test=data[:int(len(data)*p),:] #20%用于预测

from  sklearn.tree import DecisionTreeClassifier #分类器
tree=DecisionTreeClassifier()
tree.fit(train[:,:3],train[:,3]) #训练
print(tree.score(train[:,:3],train[:,3])) #训练之后的结果
print(tree.score(test[:,:3],test[:,3]))#训练之后的结果

