import sys
from datetime import datetime, timedelta

import numpy as np
import pygame

from project.data import A
from project.simulation import Simulation


class VisC:
    black = (0, 0, 0)
    white = (255, 255, 255)
    colors = [(255, 215, 0), (0, 191, 255), (220, 20, 60), (34, 139, 34)]


class Visualization:
    def __init__(
        self,
        sim: Simulation,
        trail_step_time: float,
        trail_time: float,
        scale: float = 1e-9,
        width: int = 800,
        height: int = 600,
    ) -> None:
        self.sim = sim

        self.trail_step = int(
            trail_step_time / self.sim.dt
        )  # Step between trail points
        self.trail_length = int(
            trail_time / trail_step_time
        )  # Number of positions to keep in trail

        self.speed = 1.0  # Playback speed [steps\self.frame]
        self.rotation_z = 0.0  # Rotation in-plane [rad]
        self.rotation_x = 0.0  # Rotation out-of-plane [rad]
        self._scale = scale
        self.scale = self._scale

        self.width = width
        self.height = height

        self.frame = 0
        self.running = True

        self.trail_cache: A
        self.focus_body_idx: int | None = None  # Index of the focused body
        self.trail_focus_body_idx: int | None = None  # Index of trail focus
        self.trail_cache_focus: int | None = None  # Last focus used for cache
        self.trail_cache_frame = 0  # Last self.frame used for cache
        self.cache_needs_update = False

        self.screen: pygame.Surface
        self.clock: pygame.time.Clock
        self.font: pygame.font.Font
        self.small_font: pygame.font.Font

    def start(self) -> None:
        pygame.init()
        self.screen = pygame.display.set_mode(
            (self.width, self.height), pygame.RESIZABLE
        )
        pygame.display.set_caption("Astrodynamics Simulation")
        self.clock = pygame.time.Clock()

        self.font = pygame.font.SysFont("Courier New", 24)
        self.small_font = pygame.font.SysFont("Courier New", 16)

        while self.running:
            self.handle_input()

            # if self.cache_needs_update:
            #     self.rebuild_trail_cache()
            # self.update_trail_cache()
            self.rebuild_trail_cache()

            self.draw_frame()
            self.draw_info()

        pygame.quit()
        sys.exit()

    def handle_input(self) -> None:
        # Continuous key press handling
        keys = pygame.key.get_pressed()
        if keys[pygame.K_RIGHT]:
            # Increase speed, use finer steps for low speeds
            if self.speed < 1:
                self.speed = min(self.speed + 0.1, 100)
            else:
                self.speed = min(self.speed + 1, 100)
            self.cache_needs_update = True
        if keys[pygame.K_LEFT]:
            # Decrease speed, allow fractional speeds down to 0.1
            if self.speed <= 1:
                self.speed = max(self.speed - 0.1, 0.1)
            else:
                self.speed = max(self.speed - 1, 0.1)
            self.cache_needs_update = True

        if keys[pygame.K_UP]:
            self.scale *= 1.01
            self.cache_needs_update = True
        if keys[pygame.K_DOWN]:
            self.scale /= 1.01
            self.cache_needs_update = True

        # Rotate view left/right (Z axis) and tilt up/down (X axis)
        if keys[pygame.K_d]:
            self.rotation_z -= 0.02  # radians per frame
            self.cache_needs_update = True
        if keys[pygame.K_a]:
            self.rotation_z += 0.02
            self.cache_needs_update = True
        if keys[pygame.K_w]:
            self.rotation_x -= 0.02
            self.cache_needs_update = True
        if keys[pygame.K_s]:
            self.rotation_x += 0.02
            self.cache_needs_update = True

        # Auto playback
        self.frame += int(self.speed)
        if self.frame >= self.sim.steps:
            self.frame = 0
            self.cache_needs_update = True
        # Ensure frame is integer for indexing
        self.frame = int(self.frame)

        # Event handling for quit, manual reset, and mouse click
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.running = False
            elif event.type == pygame.VIDEORESIZE:
                self.width, self.height = event.w, event.h
                self.screen = pygame.display.set_mode(
                    (self.width, self.height), pygame.RESIZABLE
                )
            elif event.type == pygame.MOUSEWHEEL:
                if event.y > 0:
                    self.scale *= 1.05
                elif event.y < 0:
                    self.scale /= 1.05
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_r:
                    self.frame = 0
                    self.scale = self._scale
                    self.rotation_x = 0
                    self.rotation_z = 0
                    self.focus_body_idx = None
                    self.speed = 1
            elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                mouse_x, mouse_y = event.pos
                body_clicked = False

                for i in range(self.sim.num_bodies):
                    body_x, body_y = self.trail_cache[-1, :2, i]
                    if (
                        abs(mouse_x - body_x) < 10
                        and abs(mouse_y - body_y) < 10
                    ):
                        body_clicked = True

                        if self.focus_body_idx != i:
                            # First click on a different body
                            self.focus_body_idx = i
                        else:
                            # Second click on the same body
                            self.trail_focus_body_idx = i
                        break

                # Only reset if no body was clicked
                if not body_clicked:
                    if self.trail_focus_body_idx is not None:
                        # First click in empty space
                        self.trail_focus_body_idx = None
                    else:
                        # Second click in empty space
                        self.focus_body_idx = None

                print(self.focus_body_idx, self.trail_focus_body_idx)
            self.cache_needs_update = True

    def draw_frame(self) -> None:
        self.screen.fill(VisC.black)

        # Draw axis system (X: red, Y: green, Z: blue)
        axis_length = 50 / self.scale  # world units, so it scales with zoom
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
            axes_center = np.array([self.width - 100, self.height - 100, 0])
            end = self.scale_pos(axis, axes_center, self.scale)
            start = (int(axes_center[0]), int(axes_center[1]))
            pygame.draw.line(self.screen, axis_colors[i], start, end, 2)
            # Draw label at the end of each axis
            label_text = self.small_font.render(
                axis_labels[i], True, axis_colors[i]
            )
            offset = 10  # pixels away from axis end
            label_pos = (end[0] + offset, end[1] + offset)
            self.screen.blit(label_text, label_pos)

        screen_pos = self.trail_cache[-1, :, :]

        for i in range(self.sim.num_bodies):
            color = VisC.colors[i % len(VisC.colors)]

            # Only draw if the body is on screen
            screen_pos_i = list(screen_pos[:2, i])
            if self.is_on_screen(screen_pos_i):
                if self.trail_cache.shape[0] > 1:
                    lines_list = list(map(tuple, self.trail_cache[:, 0:2, i]))
                    pygame.draw.lines(self.screen, color, False, lines_list, 1)
                # Draw current position
                pygame.draw.circle(self.screen, color, screen_pos_i, 3)
                if self.sim.body_list[i].name:
                    text = self.small_font.render(
                        self.sim.body_list[i].name, True, VisC.white
                    )
                    self.screen.blit(
                        text, (screen_pos_i[0] + 15, screen_pos_i[1] - 10)
                    )

        self.trail_cache_frame = self.frame

    def draw_info(self) -> None:
        # Draw timer and simulation info
        t = self.frame * self.sim.dt
        sim_date = datetime.strptime(
            self.sim.epoch, "%Y-%m-%d %H:%M:%S"
        ) + timedelta(seconds=t)
        date_text = self.font.render(
            f"Date: {sim_date.strftime('%Y-%m-%d %H:%M:%S')}", True, VisC.white
        )
        self.screen.blit(date_text, (10, 10))

        help_text = self.small_font.render(
            "UP/DOWN: Zoom, LEFT/RIGHT: Speed, R: Reset", True, VisC.white
        )
        self.screen.blit(help_text, (10, self.height - 30))

        bar_px = 100
        bar_height = 4
        bar_color = VisC.white
        margin = 30
        bar_x1 = self.width - margin - bar_px
        bar_x2 = self.width - margin
        bar_y = self.height - margin

        length = bar_px / self.scale
        pygame.draw.line(
            self.screen,
            bar_color,
            (bar_x1, bar_y),
            (bar_x2, bar_y),
            bar_height,
        )
        label_text = self.small_font.render(
            f"{length/1e3:.2e} km", True, VisC.white
        )
        label_rect = label_text.get_rect(
            center=((bar_x1 + bar_x2) // 2, bar_y - 15)
        )
        self.screen.blit(label_text, label_rect)

        # Limit FPS to 60 and get actual FPS
        self.clock.tick(60)
        actual_fps = (
            self.clock.get_fps() or 0
        )  # Avoid division by zero, fallback to 60 if not available

        # Calculate and display simulation speed (simulated seconds per real second)
        sim_speed = (
            self.speed * self.sim.dt * actual_fps / 3600 / 24
        )  # sim days per real second
        sim_speed_text = self.small_font.render(
            f"Sim speed: {sim_speed:.2f} days/s | FPS: {actual_fps:.1f} |"
            + f" Progress: {self.frame}/{self.sim.steps}",
            True,
            VisC.white,
        )
        self.screen.blit(sim_speed_text, (10, 50))

        pygame.display.flip()

    def scale_pos_array(self, pos: A) -> A:
        # Apply Z rotation (in-plane)
        cos_z = np.cos(self.rotation_z)
        sin_z = np.sin(self.rotation_z)
        x = pos[:, 0, :]
        y = pos[:, 1, :]
        z = pos[:, 2, :] if pos.shape[1] > 2 else 0
        x_rot = x * cos_z - y * sin_z
        y_rot = x * sin_z + y * cos_z
        # Apply X rotation (tilt)
        cos_x = np.cos(self.rotation_x)
        sin_x = np.sin(self.rotation_x)
        y_tilt = y_rot * cos_x - z * sin_x
        # z_tilt = y_rot * sin_x + z * cos_x
        result = np.empty_like(pos)
        result[:, 0, :] = self.width // 2 + x_rot * self.scale
        result[:, 1, :] = self.height // 2 - y_tilt * self.scale
        result[:, 2, :] = 0  # Z (2D projection)
        return result.astype(int)

    def scale_pos(self, pos, center, scale):
        # Apply Z rotation (in-plane)
        x, y, z = pos[0], pos[1], pos[2] if len(pos) > 2 else 0
        cos_z = np.cos(self.rotation_z)
        sin_z = np.sin(self.rotation_z)
        x_rot = x * cos_z - y * sin_z
        y_rot = x * sin_z + y * cos_z
        # Apply X rotation (tilt)
        cos_x = np.cos(self.rotation_x)
        sin_x = np.sin(self.rotation_x)
        y_tilt = y_rot * cos_x - z * sin_x
        # z_tilt = y_rot * sin_x + z * cos_x  # not used for 2D
        return int(center[0] + x_rot * scale), int(center[1] - y_tilt * scale)

    def update_trail_cache(self) -> None:
        new_cache = np.roll(self.trail_cache, -1, axis=0)
        current_pos = self.sim.mm[self.frame, 0:3, :][np.newaxis, :, :]
        if self.trail_focus_body_idx is not None:
            trail_focus_body_pos = self.sim.mm[
                self.frame, 0:3, self.trail_focus_body_idx
            ][np.newaxis, :, np.newaxis]
            current_pos = current_pos - trail_focus_body_pos
        elif self.focus_body_idx is not None:
            focus_body_pos = self.sim.mm[-1, 0:3, self.focus_body_idx][
                np.newaxis, :, np.newaxis
            ]
            current_pos = current_pos - focus_body_pos
        scaled_pos = self.scale_pos_array(current_pos)
        new_cache[-1, :, :] = scaled_pos

        self.trail_cache = new_cache

    def rebuild_trail_cache(self) -> None:
        new_cache = np.empty((self.trail_length, 3, self.sim.num_bodies))
        initial_point = max(
            0, self.frame - self.trail_length * self.trail_step + 1
        )
        current_pos = self.sim.mm[
            initial_point : self.frame + 1 : self.trail_step, 0:3, :
        ]
        if self.trail_focus_body_idx is not None:
            trail_focus_body_pos = self.sim.mm[
                initial_point : self.frame + 1 : self.trail_step,
                0:3,
                self.trail_focus_body_idx,
            ][:, :, np.newaxis]
            current_pos = current_pos - trail_focus_body_pos
        if self.focus_body_idx is not None:
            focus_body_pos = self.sim.mm[
                self.frame,
                0:3,
                self.focus_body_idx,
            ][np.newaxis, :, np.newaxis]
            current_pos = current_pos - focus_body_pos
        scaled_pos = self.scale_pos_array(current_pos)
        n = scaled_pos.shape[0]
        new_cache[-n:, :, :] = scaled_pos
        new_cache[0:-n, :, :] = np.repeat(
            scaled_pos[-n, :, :][np.newaxis, :, :],
            self.trail_length - n,
            axis=0,
        )
        self.trail_cache = new_cache

    def is_on_screen(self, pos, margin=100) -> bool:
        return (
            0 - margin <= pos[0] <= self.width + margin
            and 0 - margin <= pos[1] <= self.height + margin
        )
