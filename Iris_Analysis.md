# Purpose & Understanding of Dataset

#  Iris Dataset Analysis

The **Iris dataset** is one of the most well-known datasets in machine learning, first introduced by **Ronald Fisher in 1936**.  
It is commonly used for **pattern recognition** and **classification problems**.

## 🔹 Dataset Information:
- It contains **149 samples** (normally it's 150, but one is missing).
- It has **3 species of iris flowers**:
  - **Setosa**
  - **Versicolor**
  - **Virginica**
- Each sample has **4 numerical features**:
  - **Sepal Length (cm)**
  - **Sepal Width (cm)**
  - **Petal Length (cm)**
  - **Petal Width (cm)**

##  Goals of this Analysis:
 **Understand the relationships** between different flower features.  
 **Visualize patterns** between species.  
 **Identify any trends or clusters** in the dataset.  


```python
# Import necessary libraries
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset (Fixing the path issue)
df = pd.read_csv(r"C:\Users\ashna\Downloads\iris\iris_.csv")  

# Assign column names (since the dataset doesn't have them)
df.columns = ["sepal_length", "sepal_width", "petal_length", "petal_width", "species"]

# Display first few rows
df.head()

```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>sepal_length</th>
      <th>sepal_width</th>
      <th>petal_length</th>
      <th>petal_width</th>
      <th>species</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>4.9</td>
      <td>3.0</td>
      <td>1.4</td>
      <td>0.2</td>
      <td>Iris-setosa</td>
    </tr>
    <tr>
      <th>1</th>
      <td>4.7</td>
      <td>3.2</td>
      <td>1.3</td>
      <td>0.2</td>
      <td>Iris-setosa</td>
    </tr>
    <tr>
      <th>2</th>
      <td>4.6</td>
      <td>3.1</td>
      <td>1.5</td>
      <td>0.2</td>
      <td>Iris-setosa</td>
    </tr>
    <tr>
      <th>3</th>
      <td>5.0</td>
      <td>3.6</td>
      <td>1.4</td>
      <td>0.2</td>
      <td>Iris-setosa</td>
    </tr>
    <tr>
      <th>4</th>
      <td>5.4</td>
      <td>3.9</td>
      <td>1.7</td>
      <td>0.4</td>
      <td>Iris-setosa</td>
    </tr>
  </tbody>
</table>
</div>



# Check for Missing Values & Basic Stats


```python
# Check for missing values and data types
df.info()

# Show basic statistics
df.describe()

```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 149 entries, 0 to 148
    Data columns (total 5 columns):
     #   Column        Non-Null Count  Dtype  
    ---  ------        --------------  -----  
     0   sepal_length  149 non-null    float64
     1   sepal_width   149 non-null    float64
     2   petal_length  149 non-null    float64
     3   petal_width   149 non-null    float64
     4   species       149 non-null    object 
    dtypes: float64(4), object(1)
    memory usage: 5.9+ KB
    




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>sepal_length</th>
      <th>sepal_width</th>
      <th>petal_length</th>
      <th>petal_width</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>149.000000</td>
      <td>149.000000</td>
      <td>149.000000</td>
      <td>149.000000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>5.848322</td>
      <td>3.051007</td>
      <td>3.774497</td>
      <td>1.205369</td>
    </tr>
    <tr>
      <th>std</th>
      <td>0.828594</td>
      <td>0.433499</td>
      <td>1.759651</td>
      <td>0.761292</td>
    </tr>
    <tr>
      <th>min</th>
      <td>4.300000</td>
      <td>2.000000</td>
      <td>1.000000</td>
      <td>0.100000</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>5.100000</td>
      <td>2.800000</td>
      <td>1.600000</td>
      <td>0.300000</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>5.800000</td>
      <td>3.000000</td>
      <td>4.400000</td>
      <td>1.300000</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>6.400000</td>
      <td>3.300000</td>
      <td>5.100000</td>
      <td>1.800000</td>
    </tr>
    <tr>
      <th>max</th>
      <td>7.900000</td>
      <td>4.400000</td>
      <td>6.900000</td>
      <td>2.500000</td>
    </tr>
  </tbody>
</table>
</div>



# 🔹 Summary Statistics of the Dataset

The table below provides **descriptive statistics** for each numerical column.

###  Key Observations:
 **Sepal Length** ranges from **4.3 cm to 7.9 cm**, with an average of **5.85 cm**.  
 **Petal Width** has a **high standard deviation** (0.76), meaning more variation in this feature.  
 **Petal length and petal width have the highest correlation** (indicating flowers with longer petals also tend to have wider petals).  
 The dataset appears **well-balanced across species**.



```python
sns.pairplot(df, hue="species")
plt.show()

```


    
![png](output_6_0.png)
    


# Pairplot: Feature Relationships Between Species

The scatterplot below shows the relationships between **sepal length, sepal width, petal length, and petal width** for all three iris species.

###  Key Observations:
 Setosa is completely separated** from Versicolor and Virginica, meaning it’s easy to classify.  
 Versicolor and Virginica have overlapping areas**, making them harder to distinguish.  
 Petal length and petal width are strongly correlated**, meaning flowers with longer petals also tend to have wider petals.



```python
plt.hist(df["sepal_length"], bins=20, color='skyblue', edgecolor='black')
plt.xlabel("Sepal Length (cm)")
plt.ylabel("Frequency")
plt.title("Distribution of Sepal Length in Iris Dataset")
plt.show()

```


    
![png](output_8_0.png)
    


#  Sepal Length Distribution

This histogram shows the **distribution of sepal lengths** for all three iris species.

###  Key Observations:
 Most sepal lengths range between **5.0 cm and 6.5 cm**.  
 **Setosa has smaller sepals** compared to Versicolor & Virginica.  
 The dataset appears **well-balanced** across all species.



```python
plt.figure(figsize=(10, 6))
df.boxplot()
plt.title("Boxplot of Iris Dataset Features")
plt.show()

```


    
![png](output_10_0.png)
    


#  Boxplot: Identifying Outliers

The boxplot below shows the **distribution of each feature**.

###  Key Observations:
 There are **no major outliers** in the dataset.  
 Setosa has **the smallest petal width** compared to other species.  
 Petal length & petal width have **wider distributions** in Virginica.


#  Conclusion

This analysis helped us **visualize relationships** in the Iris dataset.  

###  Key Takeaways:
**Setosa is the easiest species to classify**, as it is clearly separated.  
**Versicolor and Virginica overlap**, making them harder to distinguish.  
**Petal length & petal width have the highest correlation**, meaning they grow together.  
The dataset is **well-balanced across species**.  





```python

```
