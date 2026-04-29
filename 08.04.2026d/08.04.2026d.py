import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

np.random.seed(42)
n_samples = 300

df = pd.DataFrame({
    'temperature': np.random.uniform(-10, 35, n_samples),
    'humidity': np.random.uniform(30, 90, n_samples),
    'hour': np.random.randint(0, 24, n_samples),
    'is_weekend': np.random.randint(0, 2, n_samples),
    'season': np.random.choice(['winter', 'summer'], n_samples),
    'district_type': np.random.choice(['industrial', 'residential'], n_samples)
})

df['consumption'] = (
    100 + 
    df['temperature'] * 2.5 + 
    df['humidity'] * 1.2 + 
    df['hour'] * 4.0 - 
    df['is_weekend'] * 30 + 
    (df['season'] == 'winter') * 80 + 
    (df['district_type'] == 'industrial') * 150 + 
    np.random.normal(0, 20, n_samples)
)

df = pd.get_dummies(df, columns=['season', 'district_type'], drop_first=True, dtype=float)

X = df.drop('consumption', axis=1).values
X = np.c_[np.ones(X.shape[0]), X] 
y = df['consumption'].values

split_idx = int(0.8 * n_samples)
X_train, X_test = X[:split_idx], X[split_idx:]
y_train, y_test = y[:split_idx], y[split_idx:]

theta = np.linalg.inv(X_train.T @ X_train) @ X_train.T @ y_train

y_pred = X_test @ theta

mape = np.mean(np.abs((y_test - y_pred) / y_test)) * 100
print(f"Відсоток помилки: {mape:.2f}%\n")

plt.figure(figsize=(8, 5))
plt.scatter(y_test, y_pred, color='blue', alpha=0.6)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], color='red', linewidth=2)
plt.title('Справжня vs Прогнозована ціна')
plt.xlabel('Справжнє електроспоживання')
plt.ylabel('Прогнозоване електроспоживання')
plt.grid(True, linestyle='--', alpha=0.5)
plt.tight_layout()
plt.show()