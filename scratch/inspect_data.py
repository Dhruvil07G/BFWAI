"""Quick inspection of the customer dataset."""
import pandas as pd
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

df = pd.read_csv("data/raw/customer_inventory_upload.csv")
print(f"Rows: {len(df)}")
print(f"Columns ({len(df.columns)}): {list(df.columns)}")
print(f"Unique products: {df['product_id'].nunique()}")
print(f"Product IDs: {sorted(df['product_id'].unique().tolist())}")
if 'category' in df.columns:
    print(f"Categories ({df['category'].nunique()}): {sorted(df['category'].unique().tolist())}")
print(f"Date range: {df['date'].min()} -> {df['date'].max()}")
print(f"\nDtypes:\n{df.dtypes}")
print(f"\nNull counts:\n{df.isnull().sum()}")
print(f"\nFirst 3 rows:")
print(df.head(3).to_string())
print(f"\nRecords per product:")
print(df.groupby('product_id').size().to_string())
