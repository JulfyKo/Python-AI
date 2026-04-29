import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression

np.random.seed(42)
n = 300

eng_levels = ['Elementary', 'Pre-Intermediate', 'Intermediate', 'Upper-Intermediate', 'Advanced']
eng_map = {lvl: i+1 for i, lvl in enumerate(eng_levels)}

df = pd.DataFrame({
    'Experience': np.random.randint(0, 6, n),
    'Grade': np.random.uniform(6, 12, n),
    'EnglishLevel': np.random.choice(eng_levels, n),
    'Age': np.random.randint(18, 35, n),
    'EntryTestScore': np.random.randint(400, 1000, n)
})

df['EngNum'] = df['EnglishLevel'].map(eng_map)

z = -12 + 0.6 * df['Experience'] + 0.4 * df['Grade'] + 1.2 * df['EngNum'] - 0.05 * df['Age'] + 0.007 * df['EntryTestScore']
df['Accepted'] = (np.random.rand(n) < (1 / (1 + np.exp(-z)))).astype(int)

X = df[['Experience', 'Grade', 'EngNum', 'Age', 'EntryTestScore']].values
y = df['Accepted'].values

split_idx = int(0.8 * n)
X_train, X_test = X[:split_idx], X[split_idx:]
y_train, y_test = y[:split_idx], y[split_idx:]

model = LogisticRegression(max_iter=2000)
model.fit(X_train, y_train)

plt.figure(figsize=(9, 5))
test_scores = np.linspace(400, 1000, 100)

for lvl_name, lvl_num in eng_map.items():
    X_plot = np.column_stack((
        np.full(100, df['Experience'].mean()),
        np.full(100, df['Grade'].mean()),
        np.full(100, lvl_num),
        np.full(100, df['Age'].mean()),
        test_scores
    ))
    probs = model.predict_proba(X_plot)[:, 1]
    plt.plot(test_scores, probs, label=lvl_name, lw=2)

plt.xlabel('Entry Test Score')
plt.ylabel('Acceptance Probability')
plt.legend()
plt.grid(True, ls='--', alpha=0.5)
plt.tight_layout()
plt.show()