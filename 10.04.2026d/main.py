import numpy as np
import matplotlib.pyplot as plt

np.random.seed(42)
speeds = np.sort(np.random.uniform(20, 160, 60))
consumption = 6 + 0.0012 * (speeds - 85)**2 + np.random.normal(0, 0.4, 60)

best_degree = 1
best_mse = float('inf')
best_model = None

for d in range(1, 5):
    coeffs = np.polyfit(speeds, consumption, d)
    model = np.poly1d(coeffs)
    preds = model(speeds)
    
    mse = np.mean((consumption - preds)**2)
    mae = np.mean(np.abs(consumption - preds))
    
    print(f"Ступінь {d}: MSE = {mse:.4f}, MAE = {mae:.4f}")
    
    if mse < best_mse:
        best_mse = mse
        best_degree = d
        best_model = model

print(f"\nОптимальний ступінь: {best_degree}\n")

for v in [35, 95, 140]:
    print(f"{v} км/год -> {best_model(v):.2f} л/100 км")

plt.figure(figsize=(8, 5))
plt.scatter(speeds, consumption, color='gray')

x_range = np.linspace(20, 160, 100)
plt.plot(x_range, best_model(x_range), color='red')

for v in [35, 95, 140]:
    plt.scatter(v, best_model(v), color='blue', s=100, zorder=5)

plt.grid(True, linestyle='--', alpha=0.5)
plt.tight_layout()
plt.show()