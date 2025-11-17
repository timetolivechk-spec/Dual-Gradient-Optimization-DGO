import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter

# === Dual Gradient Optimization ===
def dual_gradient_optimization(func, x_min, x_max, tol=1e-3, max_iter=200):
    left, right = x_min, x_max
    history = []

    for i in range(max_iter):
        mid = (left + right) / 2
        delta = (right - left) / 4
        x1 = mid - delta
        x2 = mid + delta
        f1, f2 = func(x1), func(x2)
        history.append((i, left, right, mid, func(mid)))

        if f1 < f2:
            right = mid
        else:
            left = mid

        if abs(right - left) < tol:
            break

    best_x = (left + right) / 2
    return best_x, func(best_x), history


# === Функция пользователя ===
def make_function(func_str):
    def f(x):
        try:
            return eval(func_str, {"np": np, "x": x})
        except Exception as e:
            print("Ошибка в функции:", e)
            return np.inf
    return f


# === Скан по сетке ===
def coarse_scan(func, x_min, x_max, n_points=2000, n_best=5):
    xs = np.linspace(x_min, x_max, n_points)
    ys = func(xs)
    best_idx = np.argsort(ys)[:n_best]
    candidates = xs[best_idx]
    return list(zip(candidates, ys[best_idx]))


# === Мультистарты DGO ===
def multi_start_dgo(func, x_min, x_max, n_starts=5, span=3):
    coarse = coarse_scan(func, x_min, x_max, n_points=2000, n_best=n_starts)
    best = (None, float('inf'))
    all_histories = []
    all_minima = []

    for c, _ in coarse:
        a, b = c - span, c + span
        x, fx, hist = dual_gradient_optimization(func, a, b)
        all_histories.append(hist)
        all_minima.append((x, fx))
        if fx < best[1]:
            best = (x, fx)

    # Ищем все одинаковые минимумы (по значению функции)
    unique_minima = []
    tol = 1e-4
    for x, fx in all_minima:
        if not any(abs(fx - fy) < tol for _, fy in unique_minima):
            unique_minima.append((x, fx))

    return best, all_histories, unique_minima


# === Анимация ===
def animate_dgo(func, histories, x_min, x_max, best, all_minima):
    xs = np.linspace(x_min, x_max, 600)
    ys = func(xs)

    fig, ax = plt.subplots()
    ax.plot(xs, ys, label='f(x)')
    ax.legend()

    point_left, = ax.plot([], [], 'ro', label='Левый агент')
    point_right, = ax.plot([], [], 'bo', label='Правый агент')

    # Покажем все равные минимумы с подписями
    for xm, fm in all_minima:
        ax.plot(xm, fm, 'y*', markersize=12)
        ax.text(xm, fm, f'({xm:.2f}, {fm:.2f})', fontsize=9,
                color='gold', ha='left', va='bottom',
                bbox=dict(facecolor='black', alpha=0.5, boxstyle='round,pad=0.3'))

    text = ax.text(0.05, 0.95, '', transform=ax.transAxes, va='top', fontsize=10)
    ax.legend()

    combined_hist = [p for hist in histories for p in hist]

    def update(frame):
        i, left, right, mid, f_mid = combined_hist[frame]
        point_left.set_data([left], [func(left)])
        point_right.set_data([right], [func(right)])
        text.set_text(f'Итерация: {i}\nЛевый x={left:.3f}, f={func(left):.3f}\n'
                      f'Правый x={right:.3f}, f={func(right):.3f}\n'
                      f'Центр x={mid:.3f}, f={f_mid:.3f}')
        return point_left, point_right, text

    ani = FuncAnimation(fig, update, frames=len(combined_hist), interval=500, blit=True)
    ani.save("dgo_result.gif", writer=PillowWriter(fps=10))
    plt.show()


# === Основная программа ===
if __name__ == "__main__":
    print("=== Dual Gradient Optimization (DGO) ===")
    print("Введите математическую функцию от x. Примеры:")
    print("  np.sin(x) + 0.1*x")
    print("  x**4 - 5*x**2 + 4")
    print("  np.cos(x) + np.sin(2*x)")
    print("-----------------------------------------")

    func_str = input("f(x) = ")
    x_min = float(input("Введите минимальное значение x_min: "))
    x_max = float(input("Введите максимальное значение x_max: "))

    func = make_function(func_str)
    best, histories, all_minima = multi_start_dgo(func, x_min, x_max, n_starts=6, span=3)

    print("\n✅ Результат:")
    for xm, fm in all_minima:
        print(f"x = {xm:.6f}, f(x) = {fm:.6f}")
    print("\nНа графике отмечены все равнозначные минимумы (жёлтые звёзды с подписями).")

    animate_dgo(func, histories, x_min, x_max, best, all_minima)

