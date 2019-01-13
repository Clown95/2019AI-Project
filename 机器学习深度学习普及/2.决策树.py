from  sklearn.tree import DecisionTreeClassifier #分类器
knn=DecisionTreeClassifier() #2个核心，2个类



x=[[180,180,180],[181,182,183],
   [160,20,100], [162,21,103]]
#180厘米，180平方，180毫米，160厘米，20平方，100毫米，
y=["高富帅","高富帅","屌丝","屌丝"]
knn.fit(x,y)#训练数据模型


print("书生是",knn.predict([[181,187,192]]))
print("52小牛是",knn.predict([[153,17,92]]))
print("追梦少年是",knn.predict([[165,27,122]]))
print("yincheng是",knn.predict([[169,17,130]]))
print("越南邻国宰相",knn.predict([[178,230,170]]))
print("乘风is" ,knn.predict([[178,120,155]]))