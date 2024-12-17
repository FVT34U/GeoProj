from scipy.io import loadmat
import numpy as np
import matplotlib.pyplot as plt

mat_data = loadmat("D:\\FVT34U\\Something\\Darya\\flow\\barrel_000000.mat")

print(mat_data.keys())

# Извлечение данных u и v
u = mat_data['u']  # Пример: матрица u
v = mat_data['v']  # Пример: матрица v

# Создание координатной сетки
x, y = np.meshgrid(np.arange(u.shape[1]), np.arange(u.shape[0]))

# Построение векторного поля
plt.figure(figsize=(8, 6))
plt.quiver(x, y, u, v, scale=1, scale_units='xy', angles='xy')
plt.title("Векторное поле (u, v)")
plt.xlabel("X")
plt.ylabel("Y")
plt.axis('equal')
plt.grid()
plt.show()