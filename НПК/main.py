import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Пример: Генерация случайных данных
X = np.random.rand(100, 1) * 10  # 100 случайных значений от 0 до 10
y = 2.5 * X + np.random.randn(100, 1)  # y зависит от X с добавлением шума

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LinearRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f'Mean Squared Error: {mse}')
print(f'R² Score: {r2}')

plt.scatter(X_test, y_test, color='blue', label='Actual Data')
plt.plot(X_test, y_pred, color='red', label='Predicted Data')
plt.xlabel('X')
plt.ylabel('y')
plt.grid()
plt.title('Линейная регрессия')
plt.legend()
plt.show()