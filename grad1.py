import numpy as np
import matplotlib.pyplot as plt

# Определяем функцию и ее градиент
def f(x, y):
    return x**2 + 4*y**2 - 2*x*y - 4*x - 16*y

def grad_f(x, y):
    df_dx = 2*x - 2*y - 4
    df_dy = 8*y - 2*x - 16
    return np.array([df_dx, df_dy])

# Градиентный спуск с постоянным шагом с записью траектории
def gradient_descent_fixed_step(x0, y0, alpha, eps=0.001, max_iter=100):
    x, y = x0, y0
    trajectory = [(x, y)]
    for i in range(max_iter):
        grad = grad_f(x, y)
        new_x = x - alpha * grad[0]
        new_y = y - alpha * grad[1]
        if np.linalg.norm([new_x - x, new_y - y]) < eps:
            break        
        x, y = new_x, new_y
        trajectory.append((x, y))
    return np.array(trajectory)

# Метод наискорейшего спуска с записью траектории
def steepest_descent(x0, y0, eps=0.001, max_iter=100):
    x, y = x0, y0
    trajectory = [(x, y)]
    for i in range(max_iter):
        grad = grad_f(x, y)
        alpha = find_optimal_alpha(x, y, grad)
        new_x = x - alpha * grad[0]
        new_y = y - alpha * grad[1]
        if np.linalg.norm([new_x - x, new_y - y]) < eps:
            break     
        x, y = new_x, new_y
        trajectory.append((x, y))
    return np.array(trajectory)

def find_optimal_alpha(x, y, grad):
    alpha = 1.0
    best_alpha = alpha
    best_value = f(x - alpha * grad[0], y - alpha * grad[1])
    for trial_alpha in np.linspace(0.01, 1, 100):
        trial_value = f(x - trial_alpha * grad[0], y - trial_alpha * grad[1])
        if trial_value < best_value:
            best_value = trial_value
            best_alpha = trial_alpha
    return best_alpha

# Градиентный спуск с дроблением шага с записью траектории
def gradient_descent_step_halving(x0, y0, initial_alpha=1.0, eps=0.001, max_iter=100):
    x, y = x0, y0
    alpha = initial_alpha
    trajectory = [(x, y)]
    for i in range(max_iter):
        grad = grad_f(x, y)
        new_x = x - alpha * grad[0]
        new_y = y - alpha * grad[1]
        if np.linalg.norm([new_x - x, new_y - y]) < eps:
            break   
        if f(new_x, new_y) < f(x, y):
            x, y = new_x, new_y
            trajectory.append((x, y))
        else:
            alpha /= 2
    return np.array(trajectory)

# Функция для построения контурного графика и траектории
def plot_descent(trajectory, title):
    x_vals = np.linspace(-5, 5, 400)
    y_vals = np.linspace(-5, 5, 400)
    X, Y = np.meshgrid(x_vals, y_vals)
    Z = f(X, Y)

    plt.figure(figsize=(8, 6))
    plt.contour(X, Y, Z, levels=50, cmap='viridis')
    plt.plot(trajectory[:, 0], trajectory[:, 1], 'r-o', markersize=3, label='Траектория')
    plt.scatter(trajectory[0, 0], trajectory[0, 1], color='blue', label='Начальная точка')
    plt.scatter(trajectory[-1, 0], trajectory[-1, 1], color='red', label='Конечная точка')
    plt.title(title)
    plt.xlabel('x')
    plt.ylabel('y')
    plt.legend()
    plt.show()

# Начальные параметры
x0, y0 = 0, 0  # Начальная точка
alpha_fixed = 0.1  # Шаг для метода с постоянным шагом
eps = 0.001  # Условие останова

# Запуск методов и построение графиков
trajectory_fixed = gradient_descent_fixed_step(x0, y0, alpha_fixed, eps)
print(f"Градиентный спуск с постоянным шагом: кол-во итераций = {trajectory_fixed.shape[0]}")
plot_descent(trajectory_fixed, "Градиентный спуск с постоянным шагом")

trajectory_steepest = steepest_descent(x0, y0, eps)
print(f"Метод наискорейшего спуска: кол-во итераций = {trajectory_steepest.shape[0]}")
plot_descent(trajectory_steepest, "Метод наискорейшего спуска")

trajectory_halving = gradient_descent_step_halving(x0, y0, eps=eps)
print(f"Градиентный спуск с дроблением шага: кол-во итераций = {trajectory_halving.shape[0]}")
plot_descent(trajectory_halving, "Градиентный спуск с дроблением шага")
