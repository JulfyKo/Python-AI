import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report

df = pd.read_csv('internship_candidates_cefr_final.csv')

level_map = {'Elementary': 1, 'Pre-Intermediate': 2, 'Intermediate': 3, 'Upper-Intermediate': 4, 'Advanced': 5}
df['EnglishLevel'] = df['EnglishLevel'].astype(str).str.strip().map(level_map)

df = df.dropna()

X = df.drop(columns=['Accepted'])
y = df['Accepted']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

print(classification_report(y_test, model.predict(X_test)))

x_min, x_max = df['EntryTestScore'].min(), df['EntryTestScore'].max()
xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100), np.linspace(1, 5, 100))

grid = np.c_[np.full(xx.ravel().shape, df['Experience'].mean()), 
             np.full(xx.ravel().shape, df['Grade'].mean()), 
             yy.ravel(), 
             np.full(xx.ravel().shape, df['Age'].mean()), 
             xx.ravel()]

probs = model.predict_proba(grid)[:, 1].reshape(xx.shape)

plt.figure(figsize=(10, 6))
plt.contourf(xx, yy, probs, levels=25, cmap="RdYlGn", alpha=0.8)
plt.colorbar(label='Probability')
plt.scatter(df['EntryTestScore'], df['EnglishLevel'], c=df['Accepted'], cmap="RdYlGn", edgecolors='black')
plt.xlabel('Entry Test Score')
plt.ylabel('English Level')
plt.show()