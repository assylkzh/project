# Traffic Data Preprocessing and Analysis

This README provides an overview of the Python script used to preprocess and analyze traffic volume data. The script is designed to clean, visualize, and transform raw data into a format suitable for further analysis or modeling.

## Project Description

The goal of this project is to analyze traffic volume data and preprocess it for machine learning models or other statistical analyses. The dataset includes information about traffic volume and various environmental and temporal factors such as weather, holidays, and time of day.

## Features

- **Data Loading**: Load traffic volume data from a CSV file.
- **Data Cleaning**: Handle missing values and outliers.
- **Feature Engineering**: Transform and encode categorical and numerical features.
- **Data Scaling**: Scale numerical features to a uniform range using MinMaxScaler.
- **Visualization**: Generate histograms, scatter plots, boxplots, and correlation heatmaps.
- **Data Export**: Save the preprocessed data to a new CSV file.
- **Modeling**: Train a Random Forest model to predict traffic volume.

## File Structure

- **`traffic-volume.csv`**: Raw dataset containing traffic and environmental data.
- **`traffic-volume_processed_data.csv`**: Processed dataset ready for analysis.
- **Python Script**: The main script performs data preprocessing and visualization.

## Prerequisites

- Python 3.8+
- Required libraries:
  - `pandas`
  - `numpy`
  - `matplotlib`
  - `seaborn`
  - `sklearn`

Install the required libraries using:
```bash
pip install pandas numpy matplotlib seaborn scikit-learn
```

## Script Workflow

1. **Load Data**:
   - The script reads the dataset from the specified file path using `pandas`.

2. **Exploratory Data Analysis (EDA)**:
   - Display summary statistics using `data.describe()` and check for missing values.
   - Visualize data distributions and relationships using histograms, scatter plots, and boxplots.

3. **Data Cleaning**:
   - Handle missing values by filling or imputing them.
   - Remove outliers using the IQR method and Isolation Forest.

4. **Feature Engineering**:
   - Extract new features such as `year`, `month`, `day`, `hour`, and `day_of_week` from the `date_time` column.
   - Encode categorical features using one-hot encoding and frequency encoding.

5. **Data Scaling**:
   - Normalize numerical features to a range of [0, 1] using `MinMaxScaler`.

6. **Model Training and Evaluation**:
   - Split the data into training and testing sets.
   - Train a Random Forest model using the training data.
   - Evaluate the model's performance using RMSE, MAE, and R^2 metrics.

7. **Visualization**:
   - Generate plots to compare actual and predicted traffic volume.

8. **Save Processed Data**:
   - Save the cleaned and transformed data to a new CSV file.

## Key Functions

### Data Loading
```python
def load_dataset(file_path):
    return pd.read_csv(file_path)
```

### Handling Missing Values
```python
def handle_null_values(df):
    data = df.copy()
    data['is_holiday'] = data['is_holiday'].fillna('no').apply(lambda x: 'yes' if x != 'no' else 'no')
    return data
```

### Outlier Detection and Removal
```python
def apply_iqr(df, iqr_columns):
    ...

def apply_isolation_forest(df, if_columns, contamination=0.01):
    ...
```

### Feature Engineering
```python
def transform_data(df):
    ...

def encode_data(df):
    ...
```

### Data Scaling
```python
def scale_data(df, is_train):
    ...
```

### Preprocessing Pipeline
```python
def preprocess_data(df, is_train):
    ...
```

### Model Training and Evaluation
```python
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.ensemble import RandomForestRegressor
import matplotlib.pyplot as plt

def train_model(X_train, y_train):
    rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
    rf_model.fit(X_train, y_train)
    return rf_model

def display_prediction(model, X_train, y_train, X_test, y_test):
    train_predictions = model.predict(X_train)
    print("-" * 30)
    print("Training RMSE:", mean_squared_error(y_train, train_predictions, squared=False))
    print("Training MAE:", mean_absolute_error(y_train, train_predictions))
    print("Training R^2:", r2_score(y_train, train_predictions))
    print("-" * 30)

    test_predictions = model.predict(X_test)
    print("Test RMSE:", mean_squared_error(y_test, test_predictions, squared=False))
    print("Test MAE:", mean_absolute_error(y_test, test_predictions))
    print("Test R^2:", r2_score(y_test, test_predictions))
    print("-" * 30)

    return train_predictions, test_predictions

def plot_results(y_test, test_predictions):
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.scatter(y_test, test_predictions, alpha=0.6)
    plt.plot([y_test.min(), y_test.max()], 
             [y_test.min(), y_test.max()],
             'r--', lw=2)
    plt.title('Actual vs Predicted Traffic Volume')
    plt.xlabel('Actual Traffic Volume')
    plt.ylabel('Predicted Traffic Volume')
    plt.grid(True)
    plt.show()
```

### Main Process
```python
def process():
    file_path = '/path/to/traffic-volume_processed_data.csv'
    data = pd.read_csv(file_path)

    X = data.drop(columns=['traffic_volume'])
    y = data['traffic_volume']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    rf_model = train_model(X_train, y_train)

    train_preds, test_preds = display_prediction(rf_model, X_train, y_train, X_test, y_test)

    plot_results(y_test, test_preds)
```

## Usage

1. Update the `file_path` variable in the script to the location of your dataset.
2. Run the script using:
```bash
python script_name.py
```
3. The preprocessed dataset will be saved to the specified output file path, and the model evaluation results will be displayed.

## Visualizations

- **Correlation Heatmap**: Displays the correlation between numerical features.
- **Histograms**: Visualize the distribution of numerical features.
- **Boxplots**: Compare traffic volume on holidays vs. non-holidays.
- **Scatter Plots**: Explore relationships between traffic volume and key features.
- **Actual vs. Predicted Plot**: Visualize the Random Forest model predictions against actual values.

