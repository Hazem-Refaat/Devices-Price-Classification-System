# Data Analysis

## Overview

This document provides a detailed analysis of the dataset used for training the Devices Price Classification System. The dataset contains various features of mobile devices and their corresponding price ranges. The analysis aims to understand the data distribution, identify correlations between features, and explore patterns that may influence the device's price range.

## Dataset Description

The dataset contains the following columns:

- **battery_power**: Total energy a battery can store in one time measured in mAh.
- **blue**: Has Bluetooth or not.
- **clock_speed**: The speed at which the microprocessor executes instructions.
- **dual_sim**: Has dual SIM support or not.
- **fc**: Front Camera megapixels.
- **four_g**: Has 4G or not.
- **int_memory**: Internal Memory in Gigabytes.
- **m_dep**: Mobile Depth in cm.
- **mobile_wt**: Weight of mobile phone.
- **n_cores**: Number of cores of the processor.
- **pc**: Primary Camera megapixels.
- **px_height**: Pixel Resolution Height.
- **px_width**: Pixel Resolution Width.
- **ram**: Random Access Memory in Megabytes.
- **sc_h**: Screen Height of mobile in cm.
- **sc_w**: Screen Width of mobile in cm.
- **talk_time**: Longest time that a single battery charge will last when you are talking.
- **three_g**: Has 3G or not.
- **touch_screen**: Has touch screen or not.
- **wifi**: Has wifi or not.
- **price_range**: Target variable with values:
  - 0 (low cost)
  - 1 (medium cost)
  - 2 (high cost)
  - 3 (very high cost)

## Data Loading and Initial Inspection

The dataset was loaded and the first few rows were inspected to get an initial understanding of the data:

```python
import pandas as pd

# Load the dataset
train_df = pd.read_csv('./Dataset/train.csv')

# Display the first few rows
print(train_df.head())
```


## Basic Information and Missing Values

Basic information about the dataset, including the data types and the presence of missing values, was examined:

```python
# display basic information about the dataset
print(train_df.info())

# check for missing values
missing_values = train_df.isnull().sum()
print("Missing values in each column:\n", missing_values)
```


The dataset contains a total of 2000 entries with some missing values in various columns. The columns `fc`, `four_g`, `int_memory`, `m_dep`, `mobile_wt`, `n_cores`, `pc`, `px_height`, `px_width`, `ram`, `sc_h`, and `sc_w` have a small number of missing values.

## Correlation Analysis

A correlation heatmap was generated to understand the relationships between different features:

```python
import seaborn as sns
import matplotlib.pyplot as plt

# correlation heatmap
plt.figure(figsize=(24, 8))
sns.heatmap(train_df.corr(), annot=True, cmap='coolwarm')
plt.title('Correlation Heatmap')
plt.show()
```


### Insights from the Correlation Heatmap

- **RAM** is highly correlated with **price_range** (correlation coefficient: 0.92), indicating that devices with more RAM tend to be in higher price ranges.
- **px_width** and **px_height** are moderately correlated with **price_range**.

## Boxplot Analysis

A boxplot was created to visualize the distribution of RAM across different price ranges:

```python
# boxplot for target variable 'price_range'
plt.figure(figsize=(10, 6))
sns.boxplot(x='price_range', y='ram', data=train_df)
plt.title('Boxplot of RAM by Price Range')
plt.show()
```

### Insights from the Boxplot

- Devices in the very high price range (3) tend to have significantly more RAM compared to devices in lower price ranges.
- There is a clear distinction in the distribution of RAM values across different price ranges, with higher price ranges showing higher median RAM values.

## Conclusion

This data analysis provides a comprehensive overview of the dataset used for the Devices Price Classification System. Key findings include the strong correlation between RAM and price range and the distribution patterns of RAM across different price ranges. These insights are valuable for building and refining the machine learning model to accurately predict the price range of mobile devices based on their specifications.

By understanding the relationships between various features and their impact on the price range, ican better preprocess the data, select relevant features, and train more effective predictive models.

# Data Exploration And Preparing

## Data Preparation

In preparation for model training, the following steps were taken to ensure the quality and integrity of the training data:

### Checking for Missing Values

Before proceeding with data cleaning, ichecked for missing values in both the training and test datasets to understand the extent of missing data and decide on appropriate handling strategies.

#### Training Dataset

```python
# check for missing values in the training dataset
missing_values_train = train_df.isnull().sum()
print("Missing values in each column (training dataset):\n", missing_values_train)
```

#### Test Dataset

```python
# check for missing values in the test dataset
missing_values_test = test_df.isnull().sum()
print("Missing values in each column (test dataset):\n", missing_values_test)
```

### Handling Missing Values

Upon inspecting the missing values, it was observed that some columns in the training dataset had a small number of missing values. Since the test dataset had no null values, i assumed that in a real-time application, every input to the model would also have no null values. Therefore, to increase the quality of the training data, i decided to drop rows with missing values from the training dataset.

```python
# drop rows with missing values in the training dataset
train_df.dropna(inplace=True)
```

By removing rows with missing values, i ensured that the training data was complete and consistent, leading to more reliable model training. This approach helps in building a more robust model that can generalize better to new, unseen data.

## Model Selection and Training

### Model Selection

In the context of tabular data, deep learning models often struggle to outperform traditional machine learning methods. This is due to several reasons, which are well-documented in the research community.

1. **Efficiency and Resource Consumption**: Deep learning models, particularly neural networks, require substantial computational resources for both training and inference. They involve a large number of parameters and require extensive hyperparameter tuning to achieve optimal performance. This can be prohibitive, especially for applications where computational resources are limited or need to be optimized for cost and efficiency【123†source】【124†source】.

2. **Performance on Tabular Data**: Traditional machine learning models like gradient boosting decision trees (GBDTs) are typically more effective for tabular data. These models, such as XGBoost, LightGBM, and CatBoost, have consistently demonstrated superior performance compared to deep learning models on tabular datasets. They are more adept at capturing the relationships and interactions between features without the need for extensive preprocessing or feature engineering【122†source】【124†source】.

3. **Ease of Use and Tuning**: Tree-based models are generally easier to tune and require fewer hyperparameters than deep learning models. They also handle missing values and categorical variables more effectively, which simplifies the data preprocessing pipeline. This makes them a more practical choice for many real-world applications where ease of use and interpretability are crucial【123†source】【124†source】.

### Model Training

Given these considerations, several models were tested, including both deep learning and tree-based models. After extensive experimentation, CatBoost emerged as the best performer in terms of accuracy and computational efficiency. Here is the rationale behind the selection and training of the CatBoost model:

1. **Superior Performance**: CatBoost consistently achieved higher accuracy than other models, including TabNet and other tree-based models like XGBoost and LightGBM. This was determined through rigorous cross-validation and comparison on the training dataset.

2. **Efficiency**: Unlike deep learning models, CatBoost does not require GPU acceleration to perform well, making it more accessible for environments with limited computational resources. It also trains faster and requires fewer computational resources during inference.

3. **Handling of Categorical Features**: CatBoost inherently supports categorical features, reducing the need for extensive preprocessing and encoding. This feature simplifies the training process and helps in maintaining the integrity of the data.

### Model Training Code

Here is the code used for training the CatBoost model:

```python
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from catboost import CatBoostClassifier

# load datasets
train_df = pd.read_csv('./Dataset/train.csv')

# drop rows with missing values
train_df.dropna(inplace=True)

# prepare features and target
X = train_df.drop(['price_range'], axis=1)
y = train_df['price_range']

# train-test split
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# initialize CatBoost model
model = CatBoostClassifier(iterations=1000, depth=10, learning_rate=0.1, loss_function='MultiClass')

# train the model
model.fit(X_train, y_train, eval_set=(X_val, y_val), verbose=False, plot=False, save_snapshot=False)

# evaluate the model
y_pred = model.predict(X_val)
print("Accuracy:", accuracy_score(y_val, y_pred))
print(classification_report(y_val, y_pred))

# save the model
model.save_model('catboost_phone_price_prediction.cbm')
```

This approach ensures that the model is well-suited for the dataset and the practical constraints of the application, providing both high accuracy and efficiency.

### Model Performance

After training the CatBoost model on the dataset, its performance was evaluated on the validation set. The model achieved a high level of accuracy, demonstrating its effectiveness in predicting the price range of devices based on their features.

#### Accuracy Metrics

- **Accuracy**: 93.48%
- **Precision, Recall, and F1-Score**:
  - **Class 0**:
    - Precision: 0.96
    - Recall: 0.93
    - F1-Score: 0.95
  - **Class 1**:
    - Precision: 0.87
    - Recall: 0.92
    - F1-Score: 0.90
  - **Class 2**:
    - Precision: 0.94
    - Recall: 0.92
    - F1-Score: 0.93
  - **Class 3**:
    - Precision: 0.96
    - Recall: 0.97
    - F1-Score: 0.96

Overall, the CatBoost model provided a balanced performance across all classes, as reflected in the macro and weighted averages:
- **Macro Average**:
  - Precision: 0.93
  - Recall: 0.94
  - F1-Score: 0.93
- **Weighted Average**:
  - Precision: 0.94
  - Recall: 0.93
  - F1-Score: 0.94

This high accuracy and balanced performance across different price ranges underscore the effectiveness of the CatBoost model for this tabular data classification task. The choice of CatBoost was further justified by its efficiency and lower computational resource requirements compared to deep learning models, which are often not well-suited for tabular data and require significant computational power.