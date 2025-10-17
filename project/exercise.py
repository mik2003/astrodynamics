import matplotlib.pyplot as plt
import numpy as np

x_0 = -10.0
y_0 = -100.0
x_0_dot = 0.0
y_0_dot = 0.0198
f_x = 0.0
f_y = 0.0
T = 3600 * 1.5403
n = 2 * np.pi / T


def xy(t):
    nt = n * t
    sinnt = np.sin(nt)
    cosnt = np.cos(nt)
    return (
        x_0 * (4 - 3 * cosnt)
        + x_0_dot / 2 * sinnt
        + 2 * y_0_dot / n * (1 - cosnt)
        + f_x / n**2 * (1 - cosnt)
        + 2 * f_y / n**2 * (nt - sinnt),
        y_0
        - y_0_dot / n * (3 * nt - 4 * sinnt)
        - 6 * x_0 * (nt - sinnt)
        - 2 * x_0_dot / n * (1 - cosnt)
        - 2 * f_x / n**2 * (nt - sinnt)
        + 2 * f_y / 2**n * (2 - 3 / 4 * n**2 * np.square(t) - 2 * cosnt),
    )


t_ = np.arange(0, 3 * T, 1)
x, y = xy(t_)

# Only show x(t) vs y(t) trajectory with time progression
fig, ax = plt.subplots(figsize=(8, 6))
sc = ax.scatter(y, x, c=t_, cmap="viridis", s=2)
ax.set_title("Trajectory: x(t) vs y(t) [time progression]")
ax.set_xlabel("y(t)")
ax.invert_xaxis()
ax.set_ylabel("x(t)")
cb = plt.colorbar(sc, ax=ax, orientation="vertical", label="Time (t)")
plt.tight_layout()
plt.show()
