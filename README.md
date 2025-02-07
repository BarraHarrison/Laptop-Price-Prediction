# Laptop Price Prediction using ML

## 📌 Introduction
The **Laptop Price Prediction** project utilizes a dataset from Kaggle, specifically `laptop_price.csv`. This dataset contains information about various laptop models, including details like company, product type, CPU, RAM, screen resolution, storage, and price. The objective of this project is to train a machine learning model to predict laptop prices based on these features.

## 🛠️ Data Preprocessing Commands

### Loading the Dataset
```python
data_frame = pd.read_csv("laptop_price.csv", encoding="latin-1")
data_frame
```
- Reads the `laptop_price.csv` dataset into a pandas DataFrame, allowing further manipulation.

### Exploring Unique Values
```python
data_frame.Product.value_counts()
data_frame.Company.value_counts()
```
- Displays the count of unique values in the `Product` and `Company` columns, giving insight into the distribution of laptop brands.

### Dropping Unnecessary Columns
```python
data_frame = data_frame.drop("Product", axis=1)
data_frame
```
- Removes the `Product` column, which is unlikely to contribute significantly to predicting laptop prices.

### Converting Categorical Data to Numerical Data
```python
pd.get_dummies(data_frame.Company)
```
- Converts the `Company` column into numerical values using one-hot encoding.
```python
data_frame = data_frame.join(pd.get_dummies(data_frame.Company))
data_frame = data_frame.drop("Company", axis=1)
data_frame
```
- Adds the one-hot encoded `Company` features to the dataset and removes the original `Company` column.
```python
data_frame = data_frame.join(pd.get_dummies(data_frame.TypeName))
data_frame = data_frame.drop("TypeName", axis=1)
data_frame
```
- Performs the same encoding process for the `TypeName` column.

### Extracting Screen Resolution Details
```python
data_frame.ScreenResolution.str.split(" ")
data_frame.ScreenResolution.str.split(" ").apply(lambda x: x[-1])
```
- Splits the `ScreenResolution` column and extracts the resolution values.
```python
data_frame["ScreenResolution"] = data_frame.ScreenResolution.str.split(" ").apply(lambda x: x[-1])
data_frame["Screen Width"] = data_frame.ScreenResolution.str.split("x").apply(lambda x: x[0])
data_frame["Screen Height"] = data_frame.ScreenResolution.str.split("x").apply(lambda x: x[1])
data_frame
```
- Stores screen width and height as separate numerical columns.
```python
data_frame = data_frame.drop("ScreenResolution", axis=1)
```
- Removes the original `ScreenResolution` column after extracting relevant details.

### Extracting CPU Details
```python
data_frame.Cpu.str.split(" ").apply(lambda x: x[0])
data_frame.Cpu.str.split(" ").apply(lambda x: x[-1])
```
- Extracts the first word (CPU brand) and last word (CPU frequency) from the `Cpu` column.
```python
data_frame["CPU Brand"] = data_frame.Cpu.str.split(" ").apply(lambda x: x[0])
data_frame["CPU Frequency"] = data_frame.Cpu.str.split(" ").apply(lambda x: x[-1])
data_frame = data_frame.drop("Cpu", axis=1)
data_frame
```
- Adds these extracted features to the dataset and removes the original `Cpu` column.

### Cleaning RAM Data
```python
data_frame["Ram"] = data_frame["Ram"].str[:-2]
```
- Removes the 'GB' suffix from the `Ram` column to keep only numerical values.

### Converting Data Types
```python
data_frame["Screen Width"] = data_frame["Screen Width"].astype("int")
data_frame["Screen Height"] = data_frame["Screen Height"].astype("int")
```
- Converts screen dimensions to integers.

### Visualizing Data Distribution
```python
data_frame.hist()
```
- Displays histograms of the numerical features to analyze data distribution.

### Extracting and Converting Memory Information
```python
data_frame["Memory Amount"] = data_frame.Memory.str.split(" ").apply(lambda x: x[0])
data_frame["Memory Type"] = data_frame.Memory.str.split(" ").apply(lambda x: x[1])
```
- Extracts the memory amount and type from the `Memory` column.
```python
def turn_memory_into_MB(value):
    if "GB" in value:
        return float(value[:value.find("GB")]) * 1000
    elif "TB" in value:
        return float(value[:value.find("TB")]) * 1000000

data_frame["Memory Amount"] = data_frame["Memory Amount"].apply(turn_memory_into_MB)
```
- Converts memory capacity from GB/TB to MB for consistency.

### Converting Weight Column
```python
data_frame["Weight"] = data_frame["Weight"].astype("float")
```
- Converts `Weight` column values to floating-point numbers for numerical analysis.

### Encoding Operating System Data
```python
data_frame = data_frame.join(pd.get_dummies(data_frame.OpSys))
data_frame = data_frame.drop("OpSys", axis=1)
```
- One-hot encodes the `OpSys` column and removes the original column.

### Encoding CPU Brands
```python
cpu_categories = pd.get_dummies(data_frame["CPU Brand"])
cpu_categories.columns = [col + "_CPU" for col in cpu_categories.columns]
data_frame = data_frame.join(cpu_categories)
data_frame = data_frame.drop("CPU Brand", axis=1)
```
- One-hot encodes the `CPU Brand` column and adds it to the dataset.

### Correlation Analysis
```python
import seaborn as sns
import matplotlib.pyplot as plt

plt.figure(figsize=(18,15))
sns.heatmap(data_frame.corr(numeric_only=True), annot=True, cmap="YlGnBu")
```
- Displays a heatmap of correlations between numerical features to identify strong relationships.
```python
target_correlations = data_frame.corr(numeric_only=True)["Price_euros"].apply(abs).sort_values()
```
- Sorts feature correlations with the target variable (`Price_euros`).

### Feature Selection
```python
selected_features = target_correlations[-20:].index
plt.figure(figsize=(18,15))
sns.heatmap(data_frame[selected_features].corr(), annot=True, cmap="YlGnBu")
```
- Selects the most relevant features for price prediction and visualizes their correlations.

## 🤖 Model Training
```python
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor

x, y = data_frame.drop("Price_euros", axis=1), data_frame["Price_euros"]
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.15)
```
- Splits the dataset into training and testing sets.
```python
scaler = StandardScaler()
x_train_scaled = scaler.fit_transform(x_train)
x_test_scaled = scaler.transform(x_test)
```
- Standardizes the feature values.

### Model Evaluation
```python
plt.figure(figsize=(12,8))
plt.scatter(y_prediction, y_test)
plt.plot(range(0, 6000), range(0, 6000), c="red")
```
- Visualizes the predicted vs. actual prices.

### Making Predictions
```python
x_new_scaled = scaler.transform([x_test.iloc[0]])
forest.predict(x_new_scaled)
```
- Predicts the price of a new laptop based on trained features.

## 🎯 Conclusion
- This project demonstrates a complete machine learning pipeline for predicting laptop prices.
- The dataset was preprocessed by encoding categorical variables and selecting key features.
- A Random Forest model was trained and evaluated, achieving high accuracy.
- The model was able to make precise price predictions, even on real-world test data.
