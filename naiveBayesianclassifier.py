import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn import metrics
df = pd.read_csv("C:/Users/AMAR S/Desktop/pima_indian.csv")
feature_col_names = ['num_preg','glucose_conc','diastolic_bp','thickness','insulin','bmi','diab_pred','age']
predicated_class_names = ['diabetes']
x = df[feature_col_names].values
y = df[predicated_class_names].values
xtrain,xtest,ytrain,ytest = train_test_split(x,y,test_size=0.33)
print('\n The total number of Training data : ',ytrain.shape)
print('\n the total unmber of Test Data : ',ytest.shape)
clf = GaussianNB().fit(xtrain,ytrain.ravel())
predicted = clf.predict(xtest)
predictTestData = clf.predict([[1,189,60,23,846,30.1,0.398,59]])
print('\n Confusion matrix')
print(metrics.confusion_matrix(ytest,predicted))
print('Accuracy of the classifier is ',metrics.accuracy_score(ytest,predicted))
print('The value of Precision ',metrics.precision_score(ytest,predicted))
print('The value Recall ',metrics.recall_score(ytest,predicted))
print('Predicted value for individual Test Data : ',predictTestData)