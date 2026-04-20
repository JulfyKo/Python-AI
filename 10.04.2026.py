import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
from sklearn.metrics import mean_absolute_error, mean_squared_error

X = np.array([20, 30, 40, 50, 60, 70, 80, 90, 100, 110]).reshape(-1, 1)
Y = np.array([9.0, 7.8, 6.9, 6.5, 6.3, 6.4, 6.7, 7.3, 8.0, 9.2])

degree = 2
model = make_pipeline(PolynomialFeatures(degree), LinearRegression())
model.fit(X, Y)

X_test = np.linspace(20, 140, 100).reshape(-1, 1)
y_pred = model.predict(X_test)

y_train_pred = model.predict(X)
mae = mean_absolute_error(Y, y_train_pred)
mse = mean_squared_error(Y, y_train_pred)

plt.figure(figsize=(10, 6))
plt.scatter(X, Y, color='blue', label='Реальні дані (таблиця)')
plt.plot(X_test, y_pred, label=f'Поліном ступеня {degree}', color='red', linestyle='--')
plt.xlabel('Швидкість (км/год)')
plt.ylabel('Витрата (л/100км)')
plt.title('Залежність витрати пального від швидкості')
plt.legend()
plt.grid(True)
plt.show()

specific_speeds = np.array([[35], [95], [140]])
predictions = model.predict(specific_speeds)

print(f"MAE: {mae:.4f}")
print(f"MSE: {mse:.4f}")
print("-" * 30)

for speed, cons in zip(specific_speeds.flatten(), predictions):
    print(f"Швидкість {speed} км/год -> Прогноз: {cons:.2f} л/100км")