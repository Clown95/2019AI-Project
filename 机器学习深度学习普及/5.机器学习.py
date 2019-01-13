from  sklearn.neighbors import KNeighborsClassifier #分类器
knn=KNeighborsClassifier(n_neighbors=2) #2个核心，2个类
x=[[180,180,44],[181,176,42],
   [160,100,35], [158,98,21]]
y=["男人","男人","女人","女人"]

knn.fit(x,y)#训练数据模型
print("书生是",knn.predict([[163,88,36]]))