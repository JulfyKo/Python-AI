import pandas as pd

data = {
    'OrderID': [1001, 1002, 1003],
    'Customer': ['Alice', 'Bob', 'Alice'],
    'Product': ['Laptop', 'Chair', 'Mouse'],
    'Category': ['Electronics', 'Furniture', 'Electronics'],
    'Quantity': [1, 2, 3],
    'Price': [1500, 180, 25],
    'OrderDate': ['2023-06-01', '2023-06-03', '2023-06-05']
}

# 1
df = pd.DataFrame(data)
df['OrderDate'] = pd.to_datetime(df['OrderDate'])

# 2
df['TotalAmount'] = df['Quantity'] * df['Price']

# 3
# a
print(df['TotalAmount'].sum())

# b
print(df['TotalAmount'].mean())

# c
print(df.groupby('Customer')['OrderID'].count())

# 4
print(df[df['TotalAmount'] > 500])

# 5
print(df.sort_values(by='OrderDate', ascending=False))

# 6
print(df[df['OrderDate'].between('2023-06-05', '2023-06-10')])

# 7
# a
print(df.groupby('Category')['Quantity'].sum())

# b
print(df.groupby('Category')['TotalAmount'].sum())

# 8
print(df.groupby('Customer')['TotalAmount'].sum().nlargest(3))