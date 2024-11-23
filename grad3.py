import numpy as np
import matplotlib.pyplot as plt

# Define the function and its gradient
def f(x, y):
    return x**2 + 4 * y**2 - 2 * x * y - 4 * x - 16 * y

def grad_f(x, y):
    df_dx = 2*x - 2*y - 4
    df_dy = 8*y - 2*x - 16
    return np.array([df_dx, df_dy])

# Step search method (linear search) Dichotomy method
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

# Fletcher-Reeves method
def fletcher_reevs(x0, grad_f, f, tol = 0.001, max_iter=1000):
    x = np.array(x0, dtype=float)
    g = grad_f(*x)
    d = -g
    iter_count = 0
    trajectory = [x.copy()] 
    while np.linalg.norm(g) > tol and iter_count < max_iter:
        alpha = line_search(f, grad_f, x, d)
        x = x + alpha * d
        g_new = grad_f(*x)
        b = np.dot(g_new, g_new) / np.dot(g, g)
        d = -g_new + b * d
        g = g_new
        
        iter_count += 1
        trajectory.append(x.copy())
        print(f"Iteration {iter_count}: x = {x}, f(x) = {f(*x)}")
    
    return np.array(trajectory)

# Pallak raiver method
def pollak_rayver(x0, grad_f, f, tol = 0.001, max_iter=1000):
    x = np.array(x0, dtype=float)
    g = grad_f(*x)
    d = -g
    iter_count = 0
    trajectory = [x.copy()]
    while np.linalg.norm(g) > tol and iter_count < max_iter:
        alpha = line_search(f, grad_f, x, d)
        x = x + alpha * d
        g_new = grad_f(*x)
        b = np.dot(g_new, g_new - g) / np.dot(g, g)
        d = -g_new + b * d
        g = g_new
        
        iter_count += 1
        trajectory.append(x.copy())
        print(f"Iteration {iter_count}: x = {x}, f(x) = {f(*x)}")
    
    return np.array(trajectory)

# Function for constructing a contour plot and trajectory
def plot_contour(trajectory, f, label):
    x_vals = np.linspace(-5, 5, 400)
    y_vals = np.linspace(-5, 5, 400)
    X, Y = np.meshgrid(x_vals, y_vals)
    Z = f(X, Y)

    plt.figure(figsize=(8, 6))
    plt.contour(X, Y, Z, levels=50, cmap='viridis')
    plt.plot(trajectory[:, 0], trajectory[:, 1], 'r-o', markersize=5, label='Траектория')
    plt.scatter(trajectory[0, 0], trajectory[0, 1], color='blue', label='Начальная точка')
    plt.scatter(trajectory[-1, 0], trajectory[-1, 1], color='red', label='Конечная точка')
    plt.title(label)
    plt.xlabel('x')
    plt.ylabel('y')
    plt.legend()
    plt.show()

x0 = [0, 0]

trajectory = fletcher_reevs(x0, grad_f, f)

plot_contour(trajectory, f, "Fletcher-Reeves method")

trajectory = pollak_rayver(x0, grad_f, f)

plot_contour(trajectory, f, "Pollak-River method")
