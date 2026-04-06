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

df = pd.DataFrame(data)
df['OrderDate'] = pd.to_datetime(df['OrderDate'])
df['TotalAmount'] = df['Quantity'] * df['Price']

print(df['TotalAmount'].sum())
print(df['TotalAmount'].mean())
print(df.groupby('Customer')['OrderID'].count())

print(df[df['TotalAmount'] > 500])

print(df.sort_values(by='OrderDate', ascending=False))

print(df[df['OrderDate'].between('2023-06-05', '2023-06-10')])

print(df.groupby('Category').agg({'Quantity': 'sum', 'TotalAmount': 'sum'}))

print(df.groupby('Customer')['TotalAmount'].sum().nlargest(3))