from sklearn import datasets
import pandas as pd
print('THIS IS THE PROGRAM TO ACCESS DIABETES DATASET')
dbt=datasets.load_diabetes()
print(" To print the description of Digits Dataset")

print(dbt.DESCR)
print('\n\n\n\n')
# df will fold dataset as a table
df=pd.DataFrame(
dbt.data,
columns=dbt.feature_names
)
#labels are assigned to df[target] table or array
df['target']=pd.Series(
dbt.target
)
print('To display First 5 samples')
# df.head(5) will return the first five samples in the dataset
print(df.head(5))
print('To display randomply 5 samples')
#df.sample(5) will return randomly five samples from the dataset
print(df.sample(5))
# Train Test Split Ratio
from sklearn.model_selection import train_test_split
df_train,df_test=train_test_split(df,test_size=0.3) # For 70: 30 Split
print('The total number of samples in the dataset = ',df.shape[0])
print('The number of samples in training set = ',df_train.shape[0])
print('The number of samples in testing set = ',df_test.shape[0])
print('The first five samples of training set')
print(df_train.head(5))
print('\n\nThe first five samples of testing set')
print(df_test.head(5))

# Built-in pandas DataFrame functions for exploration
print("\n--- DataFrame describe() ---\n", df.describe())
print("\n--- DataFrame corr() ---\n", df.corr())
print("\n--- Missing values per column ---\n", df.isnull().sum())
print("\n--- Number of duplicate rows ---\n", df.duplicated().sum())
print("\n--- Value counts for target column ---\n", df['target'].value_counts())