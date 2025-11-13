import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation

# Set up the figure with black background
fig = plt.figure(figsize=(10, 8), facecolor="black")
ax = plt.axes(xlim=(-2, 2), ylim=(-1.5, 1.5))
ax.set_facecolor("black")
ax.set_aspect("equal")
ax.axis("off")

# Colors for the three bodies
colors = ["#FF6B6B", "#4ECDC4", "#FFE66D"]  # Red, teal, yellow

# Initialize the bodies only (no trails)
bodies = []
for i in range(3):
    (body,) = ax.plot([], [], "o", markersize=12, color=colors[i])
    bodies.append(body)


def figure_eight_orbit(t, T=6.32591398):
    """
    Parametric approximation of the figure-eight orbit.
    Based on the description in the paper.
    """
    # Normalize time to period T
    tau = 2 * np.pi * t / T

    # Parametric equations for a figure-eight curve
    x = np.sin(tau)
    y = np.sin(tau) * np.cos(tau)

    return x, y


def update(frame):
    # Total period from the paper
    T = 6.32591398

    # Current time in the animation (looping)
    t_current = (frame / 300) * T

    # Calculate positions for all three bodies with phase shifts
    x1, y1 = figure_eight_orbit(t_current)
    x2, y2 = figure_eight_orbit(t_current + T / 3)
    x3, y3 = figure_eight_orbit(t_current + 2 * T / 3)

    # Update body positions only
    bodies[0].set_data([x1], [y1])
    bodies[1].set_data([x2], [y2])
    bodies[2].set_data([x3], [y3])

    return bodies


# Create the animation
ani = FuncAnimation(fig, update, frames=300, interval=20, blit=True)

# Save as GIF
print("Saving animation...")
ani.save("figure_eight_three_body_clean.gif", writer="pillow", fps=30, dpi=100)
print("Animation saved as 'figure_eight_three_body_clean.gif'")

plt.show()
