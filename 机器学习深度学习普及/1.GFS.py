from  sklearn.neighbors import KNeighborsClassifier #分类器
knn=KNeighborsClassifier(n_neighbors=2) #2个核心，2个类

x=[[180,180,180],[181,182,183],
   [160,20,100], [162,21,103]]
#180厘米，180平方，180毫米，160厘米，20平方，100毫米，
y=["高富帅","高富帅","屌丝","屌丝"]
knn.fit(x,y)#训练数据模型


print("孙宇晨是",knn.predict([[191,197,172]]))
print("杨杰是",knn.predict([[161,17,112]]))
