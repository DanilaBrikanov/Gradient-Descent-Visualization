import numpy as np
import matplotlib.pyplot as plt

def f(x, y):
    return (x + 3*y - 4) ** 2 + (x - 3 * y + 2) ** 2

def grad_f(x, y):
    return np.array([2*(x + 3*y - 4) + 2*(x - 3*y + 2),
                     6*(x + 3*y - 4) - 6*(x - 3*y + 2)])

def line_search(f, grad_f, x, d):
    alpha = 0.0
    low, high = 0.0, 1.0
    tol = 0.001
    while high - low > tol:
        alpha = (low + high) / 2
        if f(*(x + alpha * d)) < f(*(x + (alpha + tol) * d)):
            high = alpha
        else:
            low = alpha
    return alpha

# Метод наискорейшего градиентного спуска
def steepest_descent(x0, grad_f, f, tol=0.001, max_iter=1000):
    x = np.array(x0, dtype=float)
    trajectory = [x.copy()]
    iter_count = 0
    while np.linalg.norm(grad_f(*x)) > tol and iter_count < max_iter:
        grad = grad_f(*x)
        alpha = line_search(f, grad_f, x, -grad)
        x = x - alpha * grad
        trajectory.append(x.copy())
        iter_count += 1
        print(f"Iteration {iter_count}: x = {x}, f(x) = {f(*x)}")
    
    return np.array(trajectory)

# Метод тяжёлого шарика
def heavy_ball(x0, grad_f, f, beta=0.2, tol=0.001, max_iter=1000):
    x = np.array(x0, dtype=float)
    trajectory = [x.copy()]
    x_prev = x.copy()
    iter_count = 0
    while np.linalg.norm(grad_f(*x)) > tol and iter_count < max_iter:
        grad = grad_f(*x)
        alpha = line_search(f, grad_f, x, -grad)
        x_new = x - alpha * grad + beta * (x - x_prev)
        x_prev = x.copy()
        x = x_new
        trajectory.append(x.copy())
        iter_count += 1
        print(f"Iteration {iter_count}: x = {x}, f(x) = {f(*x)}")
    
    return np.array(trajectory)

# Метод Нестерова
def nesterov(x0, grad_f, f, beta=0.1, tol=0.001, max_iter=1000):
    x = np.array(x0, dtype=float)
    y = x.copy()
    trajectory = [x.copy()]
    iter_count = 0
    while np.linalg.norm(grad_f(*x)) > tol and iter_count < max_iter:
        grad = grad_f(*y)
        alpha = line_search(f, grad_f, x, -grad)
        x_new = x - alpha * grad
        y = x_new + beta * (x_new - x)
        x = x_new
        trajectory.append(x.copy())
        iter_count += 1
        print(f"Iteration {iter_count}: x = {x}, f(x) = {f(*x)}")
    
    return np.array(trajectory)


# Функция для построения контурного графика с траекторией
def plot_contour(trajectory, f, title):
    x_vals = np.linspace(-5, 5, 400)
    y_vals = np.linspace(-5, 5, 400)
    X, Y = np.meshgrid(x_vals, y_vals)
    Z = f(X, Y)

    plt.figure(figsize=(8, 6))
    plt.contour(X, Y, Z, levels=50, cmap='viridis')
    plt.plot(trajectory[:, 0], trajectory[:, 1], 'r-o', markersize=5, label='Траектория')
    plt.scatter(trajectory[0, 0], trajectory[0, 1], color='blue', label='Начальная точка')
    plt.scatter(trajectory[-1, 0], trajectory[-1, 1], color='red', label='Конечная точка')
    plt.title(title)
    plt.xlabel('x')
    plt.ylabel('y')
    plt.legend()
    plt.show()

# Начальная точка
x0 = [10, 10]

trajectory = steepest_descent(x0, grad_f, f)
plot_contour(trajectory, f, 'Steepest gradient descent method')

trajectory = heavy_ball(x0, grad_f, f)
plot_contour(trajectory, f, 'Heavy ball method')

trajectory = nesterov(x0, grad_f, f)
plot_contour(trajectory, f, 'Nesterov method')




    
