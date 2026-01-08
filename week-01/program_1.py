from sklearn import datasets
import pandas as pd 

print('THIS IS THE PROGRAM TO ACCESS IRIS DATASET')
iris = datasets.load_iris()
print("To print the description of Iris Dataset")
print(iris.DESCR)
print('\n\n\n\n')

# df will fold dataset as a table 

df = pd.DataFrame(iris.data, columns=iris.feature_names)
df['target'] = pd.Series(iris.target)

df['target_names'] = df['target'].apply(lambda y:iris.target_names[y])

print('To display Last 5 samples')

# df.head(5) will return the first five samples in the dataset
print(df.tail(5))
print('To display randomply 5 samples')
#df.sample(5) will return randomly five samples from the dataset
print(df.sample(5))

from sklearn.model_selection import train_test_split
df_train,df_test=train_test_split(df,test_size=0.3) # For 70: 30 Split
print('The total number of samples in the dataset = ',df.shape[0])
print('The number of samples in training set = ',df_train.shape[0])
print('The number of samples in testing set = ',df_test.shape[0])
print('The first five samples of training set')
print(df_train.head(5))
print('\n\nThe first five samples of testing set')
print(df_test.head(5))