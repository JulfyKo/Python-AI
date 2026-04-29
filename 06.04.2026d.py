import pandas as pd
import matplotlib.pyplot as plt

data = {
    'OrderID': [1001, 1002, 1003],
    'Customer': ['Alice', 'Bob', 'Alice'],
    'Product': ['Laptop', 'Chair', 'Mouse'],
    'Category': ['Electronics', 'Furniture', 'Electronics'],
    'Quantity': [1, 2, 3],
    'Price': [1500, 180, 25],
    'OrderDate': ['2023-06-01', '2023-06-03', '2023-06-05']
}

df = pd.DataFrame(data)
df['OrderDate'] = pd.to_datetime(df['OrderDate'])
df['TotalAmount'] = df['Quantity'] * df['Price']

print("--- 3 ---")
print(df['TotalAmount'].sum())
print(df['TotalAmount'].mean())
print(df.groupby('Customer')['OrderID'].count(), "\n")

print("--- 4 ---")
print(df[df['TotalAmount'] > 500], "\n")

print("--- 5 ---")
print(df.sort_values(by='OrderDate', ascending=False), "\n")

print("--- 6 ---")
print(df[(df['OrderDate'] >= '2023-06-05') & (df['OrderDate'] <= '2023-06-10')], "\n")

print("--- 7 ---")
print(df.groupby('Category').agg(Items_Count=('Quantity', 'sum'), Total_Sales=('TotalAmount', 'sum')), "\n")

print("--- 8 ---")
print(df.groupby('Customer')['TotalAmount'].sum().nlargest(3), "\n")

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

df.groupby(df['OrderDate'].dt.date)['OrderID'].count().plot(kind='bar', ax=axes[0])
df.groupby('Category')['TotalAmount'].sum().plot(kind='pie', ax=axes[1], autopct='%1.1f%%')

plt.tight_layout()
plt.show()