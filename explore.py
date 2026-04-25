import pandas as pd

# Load the dataset
df = pd.read_csv('heart.csv')

# Show first 5 rows
print("=== First 5 rows ===")
print(df.head())

# Show shape
print("\n=== Dataset size ===")
print(f"Rows: {df.shape[0]}, Columns: {df.shape[1]}")

# Show column names
print("\n=== Column names ===")
print(df.columns.tolist())

# Check missing values
print("\n=== Missing values ===")
print(df.isnull().sum())

# Show basic stats
print("\n=== Basic statistics ===")
print(df.describe())