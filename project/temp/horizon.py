from typing import Tuple

import matplotlib.pyplot as plt
import numpy as np


def spherical_distance(theta, h, R=6371):
    """
    Compute great-circle distance d for spherical Earth.
    theta in radians, from vertical.
    Returns NaN if outside valid domain.
    """
    t2 = np.tan(theta) ** 2
    denominator = 2 * R * h + h**2

    if denominator <= 0:
        return 0 if h == 0 else np.nan

    max_t2 = R**2 / denominator

    if t2 >= max_t2:
        return np.nan

    numerator = (R + h) * t2 + np.sqrt(R**2 - t2 * (2 * R * h + h**2))
    ratio = numerator / (t2 + 1) / R
    ratio = np.clip(ratio, -1, 1)
    return R * np.arccos(ratio)


def flat_distance(theta, h):
    """Flat Earth approximation"""
    return h * np.tan(theta)


def relative_error(spherical, flat):
    """Calculate relative error: (spherical - flat) / spherical * 100%"""
    with np.errstate(divide="ignore", invalid="ignore"):
        error = (flat - spherical) / spherical * 100
    error[np.isinf(error)] = np.nan
    return error


# Create mesh grid
R_earth = 6371  # km

# Range of parameters
h_max = 1000  # maximum height in km
theta_max_deg = 89.9  # maximum theta in degrees

precision = 150
h_vals = np.linspace(0.001, h_max, precision)
theta_vals_deg = np.linspace(0.1, theta_max_deg, precision)
theta_vals_rad = np.radians(theta_vals_deg)

H, Theta = np.meshgrid(h_vals, theta_vals_rad)

# Compute distances and errors
D_spherical = np.zeros_like(H)
D_flat = np.zeros_like(H)

for i in range(H.shape[0]):
    for j in range(H.shape[1]):
        D_spherical[i, j] = spherical_distance(Theta[i, j], H[i, j], R_earth)
        D_flat[i, j] = flat_distance(Theta[i, j], H[i, j])

# Calculate errors
relative_err = relative_error(D_spherical, D_flat)
relative_err[relative_err > 0] = 0
absolute_err = D_flat - D_spherical
absolute_err[absolute_err > 0] = 0

# Convert theta back to degrees for plotting
Theta_deg = np.degrees(Theta)

# Create 2D contour plots
fig: plt.Figure
ax1: plt.Axes
ax2: plt.Axes
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

# Plot 1: Relative Error
contour1 = ax1.contourf(
    Theta_deg,
    H,
    relative_err,
    levels=precision * 10,
    cmap="coolwarm_r",
    vmin=-50,
    vmax=0,
    extend="min",
)
# ax1.set_yscale("log")
ax1.set_xlabel("θ from Vertical (degrees)")
ax1.set_ylabel("Height h (km)")
ax1.set_title(
    "Flat Earth Relative Error (%)\n(Spherical - Flat)/Spherical × 100%"
)
cbar1 = fig.colorbar(contour1, ax=ax1)
contour1.set_clim(-50, 0)
cbar1.set_label("Relative Error (%)")

# Add contour lines for relative error
CS1 = ax1.contour(
    Theta_deg,
    H,
    relative_err,
    levels=[-25, -20, -15, -10, -5, -2, -1],
    colors="black",
    linewidths=0.5,
    linestyles="solid",
)
ax1.clabel(CS1, inline=True, fontsize=8, fmt="%.0f%%")

# Plot 2: Absolute Error
contour2 = ax2.contourf(
    Theta_deg,
    H,
    absolute_err,
    levels=precision * 10,
    cmap="plasma_r",
    vmin=-500,
    vmax=0,
    extend="min",
)
# ax2.set_yscale("log")
ax2.set_xlabel("θ from Vertical (degrees)")
ax2.set_ylabel("Height h (km)")
ax2.set_title("Flat Earth Absolute Error (km)\nSpherical - Flat")
cbar2 = fig.colorbar(contour2, ax=ax2)
cbar2.set_label("Absolute Error (km)")

# Add contour lines for absolute error
CS2 = ax2.contour(
    Theta_deg,
    H,
    absolute_err,
    levels=[-200, -100, -50, -20, -10, -5, -2, -1],
    colors="white",
    linewidths=0.5,
    linestyles="solid",
    alpha=0.7,
)
ax2.clabel(CS2, inline=True, fontsize=8, fmt="%.0f km")

# Add the domain boundary (where tan²θ = R²/(2Rh+h²))
boundary_theta = []
boundary_h = []
for h in h_vals:
    if h > 0:
        theta_max = np.arctan(R_earth / np.sqrt(2 * R_earth * h + h**2))
        boundary_theta.append(np.degrees(theta_max))
        boundary_h.append(h)

ax1.plot(
    boundary_theta, boundary_h, "k--", linewidth=2, label="Domain Boundary"
)
ax2.plot(
    boundary_theta, boundary_h, "k--", linewidth=2, label="Domain Boundary"
)
ax1.legend()
ax2.legend()

plt.tight_layout()
plt.show()

# Error statistics
valid_mask = ~np.isnan(relative_err) & ~np.isinf(relative_err)
valid_errors = relative_err[valid_mask]
valid_abs_errors = absolute_err[valid_mask]

print("Error Analysis:")
print(f"Maximum relative error: {np.nanmax(valid_errors):.1f}%")
print(f"Minimum relative error: {np.nanmin(valid_errors):.1f}%")
print(f"Mean relative error: {np.nanmean(valid_errors):.1f}%")
print(f"Maximum absolute error: {np.nanmax(valid_abs_errors):.1f} km")
print(f"Mean absolute error: {np.nanmean(valid_abs_errors):.1f} km")

# Show regions of over/under estimation
overestimate_mask = relative_err > 0
underestimate_mask = relative_err < 0
print(f"\nFlat Earth overestimates in {np.sum(overestimate_mask)} points")
print(f"Flat Earth underestimates in {np.sum(underestimate_mask)} points")
print(
    f"Ratio of overestimation: {np.sum(overestimate_mask)/np.sum(valid_mask)*100:.1f}% of valid domain"
)
