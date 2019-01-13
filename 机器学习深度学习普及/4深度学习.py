import pandas  as pd  #数据处理
import random #随机数
data=pd.read_excel("model.xls")
data=data.as_matrix()#表格转化数组
#random.shuffle(data)#打乱顺序
print(data) #读取数据
p=0.8 #定义概率
train=data[:int(len(data)*p),:] #80%用于训练
test=data[:int(len(data)*p),:] #20%用于预测

from keras.models import Sequential #神经网络初始化
from  keras.layers.core import  Dense,Activation#层次，激活函数
net=Sequential()#构建神经网络
net.add(Dense(input_dim=3,output_dim=14)) #输入3个节点，输出14个节点
net.add(Activation("relu"))#激活神经网络

net.add(Dense(input_dim=14,output_dim=18)) #输入3个节点，输出14个节点
net.add(Activation("relu"))#激活神经网络

net.add(Dense(input_dim=18,output_dim=1)) #输入3个节点，输出14个节点
net.add(Activation("sigmoid"))#神经网络外层输出

net.compile(loss="binary_crossentropy", #损失函数
            optimizer="adam",
           )#编译神经网络，加速，降低误差
net.fit(train[:,:3],train[:,3],epochs=10,batch_size=1)
last=net.predict_classes(train[:,:3]).reshape(len(train))#预测数据
print((last==train[:,3]).sum()/len(last)) #准确率