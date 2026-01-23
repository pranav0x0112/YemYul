import pandas as pd
from sklearn import datasets
from sklearn.model_selection import train_test_split
# load the iris dataset
iris = datasets.load_iris()
df = pd.DataFrame(iris.data)
label = pd.DataFrame(iris.target)
# Let us split 70% for Training and 30% for Testing
X_train, X_test, y_train, y_test = train_test_split(df,
label,test_size=0.3,shuffle = True)
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train,
test_size=0.3,shuffle= True)
print(" Totoal number of samples in the Dataset ", df.shape[0])
print(" Total number of labels in the Dataset ", label.shape[0])
print(" Total number of samples in training set X_train ",
X_train.shape[0])
print(" Total number of labels in training set y_train " ,
y_train.shape[0])
print(" Total number of samples in testing set X_test ",
X_test.shape[0])
print(" Total number of labels in testing set y_test " ,
y_test.shape[0])
print(" Total number of samples in validation set X_test ",
X_val.shape[0])
print(" Total number of labels in validation set y_test " ,
y_val.shape[0])
print('The 3 samples of training set')
print(X_train.head(3))
print('The random 3 samples of training set')
print(X_train.sample(3))
print('The 3 samples of testing set')
print(X_test.head(3))
print('The random 3 samples of testing set')
print(X_test.sample(3))
print('The 3 samples of validation set')
print(X_val.head(3))
print('The random 3 samples of validation set')
print(X_val.sample(3))
