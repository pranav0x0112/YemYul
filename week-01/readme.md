# Iris Dataset: ML Lab Q&A

**1. What are the features (columns) present in the Iris dataset, and what do they represent?**
- sepal length (cm): Length of the sepal
- sepal width (cm): Width of the sepal
- petal length (cm): Length of the petal
- petal width (cm): Width of the petal

**2. How many samples and species are included in the Iris dataset?**
- 150 samples
- 3 species: setosa, versicolor, virginica

**3. What is the distribution of each species in the dataset? Is it balanced?**
- Each species has 50 samples (balanced dataset)

**4. Are there any missing values or duplicates in the Iris dataset?**
- There are no missing values, but there is 1 duplicate row in the Iris dataset as loaded by scikit-learn.

**5. What are the minimum, maximum, mean, and standard deviation values for each feature?**
- Use `df.describe()` in pandas to get these statistics

**6. Which features are numeric and which are categorical?**
- All four features are numeric; the target/species is categorical

**7. What is the unit of measurement for the sepal and petal attributes?**
- Centimeters (cm)

**8. How can you load the Iris dataset using Python libraries such as pandas or scikit-learn?**
```python
from sklearn import datasets
import pandas as pd
iris = datasets.load_iris()
df = pd.DataFrame(iris.data, columns=iris.feature_names)
df['target'] = iris.target
df['species'] = pd.Series(iris.target).apply(lambda y: iris.target_names[y])
```

**9. What is the correlation between different features in the Iris dataset?**
- Use `df.corr()` to compute correlations

**10. Are there any outliers in the sepal width or other features? How can they be detected and removed?**
- Use boxplots or `df[(df['sepal width (cm)'] > upper) | (df['sepal width (cm)'] < lower)]` to detect outliers

**11. Which feature(s) best separate the three Iris species?**
- Petal length and petal width best separate the species

**12. What visualizations can help in understanding the distribution of features across species?**
- Boxplots, scatter plots, pair plots (e.g., seaborn's `pairplot`)

**13. How can you check for and handle missing or duplicate data in the dataset using pandas?**
```python
df.isnull().sum()  # Check missing
df.duplicated().sum()  # Check duplicates
df.dropna()  # Remove missing
df.drop_duplicates()  # Remove duplicates
```

**14. How can you visualize the data using boxplots, scatter plots, or pair plots?**
```python
import seaborn as sns
sns.boxplot(x='species', y='sepal length (cm)', data=df)
sns.pairplot(df, hue='species')
```

**15. How do you convert the Iris dataset into a pandas DataFrame for analysis?**
- See code in (8) above