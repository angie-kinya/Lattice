import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset
try:
    df = pd.read_csv('data/lattice_physics_dataset.csv', header=None)
    print("Dataset loaded successfully!")
except FileNotFoundError:
    print("Error: The file was not found. Please check the file path.")
except Exception as e:
    print(f"An error occurred: {e}")

# Check the number of columns
num_columns = df.shape[1]
print(f"Number of columns: {num_columns}")

# Assign column names
column_names = [f'enrichment_{i}' for i in range(1, num_columns - 1)] + ['k-inf', 'PPPF']
df.columns = column_names

# Display the first 5 rows of the dataset
print(df.head())

# Verify the column names
print(f"Columns in DataFrame: {df.columns}")

# Explore the structure of the dataset
print(df.info())

# Check the shape of the dataset
print(f"Dataset shape: {df.shape}")

# Check the data types present
print(df.dtypes)

# Check for missing values
print(f"Missing values: {df.isnull().sum()}")
assert df.isnull().sum().sum() == 0, "There are missing values in the dataset!"


# Summary statistics
print(df.describe())

# Compute the mean, median, and SD for the target variables
target_stats = df[['k-inf', 'PPPF']].agg(['mean', 'median', 'std'])
print(target_stats)

# Create enrichment ranges (bins) using quantile binning
df['enrichment_range'] = pd.qcut(df['enrichment_1'], q=4, labels=False)

# Group by enrichment ranges and compute the mean of k-inf and PPPF
grouped = df.groupby('enrichment_range')[['k-inf', 'PPPF']].mean()
print(grouped)

# Check correlation between enrichments and target variables
correlation_matrix = df.corr()
print(correlation_matrix[['k-inf', 'PPPF']].sort_values(by='k-inf', ascending=False))


# Data Visualization
# Line cahrt showing k-inf vs enrichment for a specific fuel rod
plt.figure(figsize=(10, 6))
plt.plot(df['enrichment_1'], df['k-inf'], marker='o', linestyle='-', color='b')
plt.title('Trend of k-inf vs. Fuel Rod Enrichment')
plt.xlabel('Enrichment (w/o U-235)')
plt.ylabel('k-inf')
plt.grid(True)
plt.show()

sns.set_style('whitegrid')
# Bar Chart showing mean PPPF by enrichment range
plt.figure(figsize=(10, 6))
sns.barplot(x='enrichment_range', y='PPPF', data=df, ci=None)
plt.title('Mean PPPF by Enrichment Range')
plt.xlabel('Enrichment Range (w/o U-235)')
plt.ylabel('Mean PPPF')
plt.show()

# Histogram distribution of k-inf
plt.figure(figsize=(10, 6))
sns.histplot(df['k-inf'], kde=True, color='g')
plt.title('Distribution of k-inf')
plt.xlabel('k-inf')
plt.ylabel('Frequency')
plt.show()

# Scatter plot showing the relationship between k-inf and PPPF
plt.figure(figsize=(10, 6))
sns.scatterplot(x='k-inf', y='PPPF', data=df, alpha=0.6)
plt.title('Relationship between k-inf and PPPF')
plt.xlabel('k-inf')
plt.ylabel('PPPF')
plt.show()