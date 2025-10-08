import os
import sys

import numpy as np
import pygame

from project.data import BodyList
from project.formulas import simulate_n_steps
from project.utilities import A, Dir, load_trails_npy

# Simulation parameters
dt = 3600  # simulation time step (seconds)
time = 3600 * 24 * 365.25 * 100  # simulation time (seconds)
steps = int(time / dt)  # total number of simulation steps

fname = "sun_earth"
file_in = Dir.data_dir.joinpath(fname + ".json")
file_traj = Dir.data_dir.joinpath(f"{fname}_{dt}_{steps}.bin")

# Run simulation first and save trajectory with progress tracker
body_list = BodyList.load(file_in)
if not os.path.exists(file_traj):
    print(f"Simulating {time:.2e} seconds...")
    simulate_n_steps(body_list, steps, dt, file_traj, prnt=True)
    print("\nSimulation complete.")

# Visualization
pygame.init()
WIDTH, HEIGHT = 800, 600
screen = pygame.display.set_mode((WIDTH, HEIGHT), pygame.RESIZABLE)
pygame.display.set_caption("Astrodynamics Simulation")
clock = pygame.time.Clock()

BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
COLORS = [(255, 215, 0), (0, 191, 255), (220, 20, 60), (34, 139, 34)]


trail_step_time = 3600 * 24  # [s]
trail_time = 3600 * 24 * 365.25 * 10  # [s]
TRAIL_STEP = int(trail_step_time / dt)  # Step between trail points
TRAIL_LENGTH = int(
    trail_time / trail_step_time
)  # Number of positions to keep in trail

# Playback speed control
speed = 1.0  # frames per tick (float for finer control)

# Rotation variables (in radians)
rotation_z = 0.0  # rotation around Z axis (in-plane)
rotation_x = 0.0  # tilt (out-of-plane)


def scale_pos_array(pos: A, center: A, scale: float) -> A:
    # Apply Z rotation (in-plane)
    cos_z = np.cos(rotation_z)
    sin_z = np.sin(rotation_z)
    x = pos[:, 0, :]
    y = pos[:, 1, :]
    z = pos[:, 2, :] if pos.shape[1] > 2 else 0
    x_rot = x * cos_z - y * sin_z
    y_rot = x * sin_z + y * cos_z
    # Apply X rotation (tilt)
    cos_x = np.cos(rotation_x)
    sin_x = np.sin(rotation_x)
    y_tilt = y_rot * cos_x - z * sin_x
    z_tilt = y_rot * sin_x + z * cos_x
    result = np.empty_like(pos)
    result[:, 0, :] = center[0] + x_rot * scale
    result[:, 1, :] = center[1] - y_tilt * scale
    result[:, 2, :] = 0  # Z (2D projection)
    return result.astype(int)


def scale_pos(pos, center, scale):
    # Apply Z rotation (in-plane)
    x, y, z = pos[0], pos[1], pos[2] if len(pos) > 2 else 0
    cos_z = np.cos(rotation_z)
    sin_z = np.sin(rotation_z)
    x_rot = x * cos_z - y * sin_z
    y_rot = x * sin_z + y * cos_z
    # Apply X rotation (tilt)
    cos_x = np.cos(rotation_x)
    sin_x = np.sin(rotation_x)
    y_tilt = y_rot * cos_x - z * sin_x
    # z_tilt = y_rot * sin_x + z * cos_x  # not used for 2D
    return int(center[0] + x_rot * scale), int(center[1] - y_tilt * scale)


scale = 1e-9  # Adjust for visualization
center = np.array([WIDTH // 2, HEIGHT // 2, 0])

bar_px = 100
bar_height = 4
bar_color = WHITE
margin = 30
bar_x1 = WIDTH - margin - bar_px
bar_x2 = WIDTH - margin
bar_y = HEIGHT - margin


frame = 0
running = True
num_bodies = len(body_list)
focus_body_idx = 0  # Index of the focused body
trail_cache_focus = None  # Last focus used for cache
trail_cache_frame = 0  # Last frame used for cache
cache_needs_update = False

font = pygame.font.SysFont(None, 36)
small_font = pygame.font.SysFont(None, 24)

mm = np.memmap(
    file_traj,
    dtype="float64",
    mode="r",
    shape=(steps, 9, num_bodies),
)[:, 0:3, :]


def update_trail_cache(cache: A) -> A:
    new_cache = np.roll(cache, -1, axis=0)
    rel_trail_pos = (
        mm[frame, :, :]
        - mm[frame, :, focus_body_idx][np.newaxis, :, np.newaxis]
    )
    scaled_pos = scale_pos_array(rel_trail_pos, center, scale)
    new_cache[-1, :, :] = scaled_pos

    return new_cache


def rebuild_trail_cache() -> A:
    new_cache = np.empty((TRAIL_LENGTH, 3, num_bodies))
    initial_point = max(0, frame - TRAIL_LENGTH * TRAIL_STEP + 1)
    rel_trail_pos = (
        mm[initial_point : frame + 1 : TRAIL_STEP, :, :]
        - mm[
            initial_point : frame + 1 : TRAIL_STEP,
            :,
            focus_body_idx,
        ][:, :, np.newaxis]
    )
    scaled_pos = scale_pos_array(rel_trail_pos, center, scale)
    n = scaled_pos.shape[0]
    new_cache[-n:, :, :] = scaled_pos
    new_cache[0:-n, :, :] = np.repeat(
        scaled_pos[-n, :, :][np.newaxis, :, :], TRAIL_LENGTH - n, axis=0
    )
    return new_cache


trail_cache = rebuild_trail_cache()


while running:
    # Continuous key press handling
    keys = pygame.key.get_pressed()
    if keys[pygame.K_RIGHT]:
        # Increase speed, use finer steps for low speeds
        if speed < 1:
            speed = min(speed + 0.1, 100)
        else:
            speed = min(speed + 1, 100)
        cache_needs_update = True
    if keys[pygame.K_LEFT]:
        # Decrease speed, allow fractional speeds down to 0.1
        if speed <= 1:
            speed = max(speed - 0.1, 0.1)
        else:
            speed = max(speed - 1, 0.1)
        cache_needs_update = True

    if keys[pygame.K_UP]:
        scale *= 1.01
        cache_needs_update = True
    if keys[pygame.K_DOWN]:
        scale /= 1.01
        cache_needs_update = True

    # Rotate view left/right (Z axis) and tilt up/down (X axis)
    if keys[pygame.K_d]:
        rotation_z -= 0.02  # radians per frame
        cache_needs_update = True
    if keys[pygame.K_a]:
        rotation_z += 0.02
        cache_needs_update = True
    if keys[pygame.K_w]:
        rotation_x -= 0.02
        cache_needs_update = True
    if keys[pygame.K_s]:
        rotation_x += 0.02
        cache_needs_update = True

    # Auto playback
    frame += speed
    if frame >= steps:
        frame = 0
        cache_needs_update = True
    # Ensure frame is integer for indexing
    frame = int(frame)

    # Event handling for quit, manual reset, and mouse click
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        elif event.type == pygame.VIDEORESIZE:
            WIDTH, HEIGHT = event.w, event.h
            screen = pygame.display.set_mode((WIDTH, HEIGHT), pygame.RESIZABLE)
            center = np.array([WIDTH // 2, HEIGHT // 2, 0])
            bar_x1 = WIDTH - margin - bar_px
            bar_x2 = WIDTH - margin
            bar_y = HEIGHT - margin
            cache_needs_update = True
        elif event.type == pygame.MOUSEWHEEL:
            if event.y > 0:
                scale *= 1.05
                cache_needs_update = True
            elif event.y < 0:
                scale /= 1.05
                cache_needs_update = True
            cache_needs_update = True
        elif event.type == pygame.KEYDOWN:
            if event.key == pygame.K_r:
                frame = 0
                cache_needs_update = True
        elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
            mouse_x, mouse_y = event.pos
            positions = mm[frame]
            for i in range(num_bodies):
                # Calculate screen position relative to current focus
                if focus_body_idx is not None:
                    rel_pos = positions[:, i] - positions[:, focus_body_idx]
                else:
                    rel_pos = positions[:, i]
                body_x, body_y = scale_pos(rel_pos, center, scale)
                if abs(mouse_x - body_x) < 10 and abs(mouse_y - body_y) < 10:
                    if focus_body_idx != i:
                        focus_body_idx = i
                        cache_needs_update = True
                    break

    screen.fill(BLACK)

    # Draw axis system (X: red, Y: green, Z: blue)
    axis_length = 50 / scale  # world units, so it scales with zoom
    axes = (
        np.array(
            [
                [1, 0, 0],  # X
                [0, 1, 0],  # Y
                [0, 0, 1],  # Z
            ]
        )
        * axis_length
    )
    axis_colors = [(255, 0, 0), (0, 255, 0), (0, 128, 255)]
    axis_labels = ["X", "Y", "Z"]
    for i, axis in enumerate(axes):
        axes_center = np.array([WIDTH - 100, HEIGHT - 100, 0])
        end = scale_pos(axis, axes_center, scale)
        start = (int(axes_center[0]), int(axes_center[1]))
        pygame.draw.line(screen, axis_colors[i], start, end, 2)
        # Draw label at the end of each axis
        label_text = small_font.render(axis_labels[i], True, axis_colors[i])
        offset = 10  # pixels away from axis end
        label_pos = (end[0] + offset, end[1] + offset)
        screen.blit(label_text, label_pos)

    # Draw bodies and their trails at current frame
    positions = mm[frame]
    # If a body is focused, subtract its position from all others
    if focus_body_idx is not None:
        focus_pos = positions[:, focus_body_idx]
    else:
        focus_pos = np.zeros(3)
    if cache_needs_update:
        trail_cache = rebuild_trail_cache()

    trail_cache = update_trail_cache(trail_cache)

    for i in range(num_bodies):
        color = COLORS[i % len(COLORS)]
        if trail_cache.shape[0] > 1:
            lines_list = list(map(tuple, trail_cache[:, 0:2, i]))
            pygame.draw.lines(screen, color, False, lines_list, 1)
        # Draw current position
        rel_pos = positions[:, i] - focus_pos
        x, y = scale_pos(rel_pos, center, scale)
        pygame.draw.circle(screen, color, (x, y), 3)
        if body_list[i].name:
            text = small_font.render(body_list[i].name, True, WHITE)
            screen.blit(text, (x + 15, y - 10))
    trail_cache_frame = frame

    # Draw timer and simulation info
    t = frame * dt
    days = t / (24 * 3600)
    hours = (days - int(days)) * 24
    # Show years if days exceed one sidereal year (365.25 days)
    if days >= 365.25:
        years = int(days // 365.25)
        rem_days = days - years * 365.25
        timer_text = font.render(
            f"Time: {years} years, {int(rem_days)} days, {hours:.1f} hours",
            True,
            WHITE,
        )
    else:
        timer_text = font.render(
            f"Time: {int(days)} days, {hours:.1f} hours", True, WHITE
        )
    screen.blit(timer_text, (10, 10))

    help_text = small_font.render(
        "UP/DOWN: Zoom, LEFT/RIGHT: Speed, R: Reset", True, WHITE
    )
    screen.blit(help_text, (10, HEIGHT - 30))

    length = bar_px / scale
    pygame.draw.line(
        screen, bar_color, (bar_x1, bar_y), (bar_x2, bar_y), bar_height
    )
    label_text = small_font.render(f"{length/1e3:.2e} km", True, WHITE)
    label_rect = label_text.get_rect(
        center=((bar_x1 + bar_x2) // 2, bar_y - 15)
    )
    screen.blit(label_text, label_rect)

    # Limit FPS to 60 and get actual FPS
    clock.tick(60)
    actual_fps = (
        clock.get_fps() or 0
    )  # Avoid division by zero, fallback to 60 if not available

    # Calculate and display simulation speed (simulated seconds per real second)
    sim_speed = speed * dt * actual_fps / 3600 / 24  # sim days per real second
    sim_speed_text = small_font.render(
        f"Sim speed: {sim_speed:.2f} days/s | FPS: {actual_fps:.1f} | Progress: {frame}/{steps}",
        True,
        WHITE,
    )
    screen.blit(sim_speed_text, (10, 50))

    pygame.display.flip()

pygame.quit()
sys.exit()
