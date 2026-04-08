import numpy as np
import matplotlib.pyplot as plt

x = np.linspace(-10, 10, 500)
y = x**2 * np.sin(x)
plt.figure()
plt.plot(x, y)
plt.title('Task 1')
plt.show()

data_norm = np.random.normal(5, 2, 1000)
plt.figure()
plt.hist(data_norm, bins=30)
plt.title('Task 2')
plt.show()

hobbies = ['Coding', 'Gaming', 'Riding', 'Mountains']
shares = [40, 30, 20, 10]
plt.figure()
plt.pie(shares, labels=hobbies)
plt.title('Task 3')
plt.show()

fruits_data = [np.random.normal(150, 15, 100) for _ in range(4)]
plt.figure()
plt.boxplot(fruits_data, labels=['Apple', 'Banana', 'Orange', 'Pear'])
plt.title('Task 4')
plt.show()

x_scat = np.random.uniform(0, 1, 100)
y_scat = np.random.uniform(0, 1, 100)
plt.figure()
plt.scatter(x_scat, y_scat, color='green', alpha=0.6)
plt.xlabel('X axis')
plt.ylabel('Y axis')
plt.title('Task 5')
plt.show()