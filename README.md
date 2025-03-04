#  Iris Dataset Analysis with Synthetic Data

##  Introduction
This project explores the **Iris dataset**, a well-known dataset in machine learning, originally introduced by Ronald Fisher in 1936.  
In addition to analyzing the **Iris dataset**, we also **created and merged a synthetic flower dataset** to compare classification models.

## ** Dataset Information**
We worked with **two datasets**:
1. **Original Iris Dataset (`iris_.csv`)**  
   - 149 samples across **three species** (Setosa, Versicolor, Virginica).  
   - Features: Sepal & Petal Length/Width.

2. **Synthetic Flower Dataset (`Synthetic_Flower_2ndDataset.csv`)**  
   - 149 samples across **three new species** (Lily, Orchid, Rose).  
   - Generated with unique feature distributions.

## ** Objectives**
✔ Understand relationships between flower features in different datasets.  
✔ Merge and analyze data from **two different datasets**.  
✔ Compare patterns between species in the Iris and synthetic datasets.  

---

## ** Methodology**

### **1️ Data Preparation**
- Loaded **Iris dataset (`iris_.csv`)** and **Synthetic Flower dataset (`Synthetic_Flower_2ndDataset.csv`)**.
- Ensured **"Flower_ID"** existed in both datasets for merging.

### **2️ Merging Process**
- Used **Flower_ID** as a unique identifier for merging.
- Applied a **left join** to merge datasets.
- Removed duplicate columns to maintain a clean structure.
- Saved the **final merged dataset (`Final_Merged_Dataset.csv`)**.

### **3️ Data Analysis & Visualization**
- Created **pairplots, histograms, and boxplots** to compare original and synthetic datasets.
- Analyzed **differences in feature distributions between real and synthetic flowers**.

---

## ** Key Findings**

### **1️ Comparison Between Original & Synthetic Data**
- **Setosa is well-separated** in the original dataset, while **Lily, Orchid, and Rose** have overlapping feature distributions.
- **Synthetic dataset shows more variability in Sepal and Petal Width**.
- **Petal Length & Petal Width remain strongly correlated** in both datasets.

### **2️ Merging & Data Integration Insights**
- The **final dataset combines both original and synthetic data**, allowing for cross-dataset comparisons.
- The merging process **highlighted differences in feature distributions** across datasets.

---

## ** Conclusion**
This project demonstrates **how different datasets can be merged, compared, and analyzed** for classification tasks.

 **Comparison between two different flower datasets.**  
 **Successful data merging process with minimal loss.**  
 **Key insights from cross-dataset visualization.**  






