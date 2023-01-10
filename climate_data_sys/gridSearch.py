import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.model_selection import KFold
from sklearn.model_selection import GridSearchCV

iris_data = pd.read_csv("",encoding="utf-8")


y =iris_data.loc[:,"Name"]
x = iris_data.loc[:,["SepalLength","SepalWidth","PetalLength","PetalWidth"]]

#deviding train and test
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size =0.2,train_size = 0.6,shuffle=True)

#パラメータを指定する
parameters = [
    {"C":[],"kernel":["linear"]},
    {"C":[],"kernel":["rbf"],"gamma":[0.001,0.0001]},
    {"C":[],"kernel":["sigmoid"],"gamma":[0.001,0.0001]}
]

Kfold_cs =KFold(n_splits=5,shuffle=True)
clf =GridSearchCV(SVC(),parameters,cv=kfold_cv)
clf.fit(x_train,y_tarain)
print("最適なパラメーター=",clf.best_estimator_)

y_pred =clf.predict(x_test)

print("評価正解率=",accuracy_score(y_test,y_pred))

