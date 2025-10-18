import sys
import time
from datetime import datetime, timedelta
from typing import List

import numpy as np
import pygame

from project.data import A
from project.simulation import Simulation


class Slider:
    def __init__(
        self, x, y, width, height, min_val, max_val, initial_val, label
    ):
        self.rect = pygame.Rect(x, y, width, height)
        self.handle_rect = pygame.Rect(x, y - 5, 10, height + 10)
        self.min_val = min_val
        self.max_val = max_val
        self.val = initial_val
        self.dragging = False
        self.label = label
        self.update_handle_position()

    def update_handle_position(self):
        relative_val = (self.val - self.min_val) / (
            self.max_val - self.min_val
        )
        self.handle_rect.centerx = (
            self.rect.left + relative_val * self.rect.width
        )

    def draw(self, screen, font):
        # Draw slider track
        pygame.draw.rect(screen, (100, 100, 100), self.rect)
        pygame.draw.rect(screen, (200, 200, 200), self.rect, 2)

        # Draw handle
        pygame.draw.rect(screen, (255, 255, 255), self.handle_rect)
        pygame.draw.rect(screen, (150, 150, 150), self.handle_rect, 2)

        # Draw label and value
        # Choose unit dynamically: hours (h), days (d), years (a)
        seconds = self.val
        if seconds < 3600 * 24:
            unit = "h"
            value = seconds / 3600
            value_str = f"{value:.0f}"
        elif seconds < 3600 * 24 * 365:
            unit = "d"
            value = seconds / (3600 * 24)
            value_str = f"{value:.1f}" if value < 10 else f"{value:.0f}"
        else:
            unit = "a"
            value = seconds / (3600 * 24 * 365.25)
            value_str = (
                f"{value:.2f}"
                if value < 10
                else f"{value:.1f}" if value < 100 else f"{value:.0f}"
            )

        label_text = font.render(
            f"{self.label} ({unit}): {value_str}", True, (255, 255, 255)
        )
        screen.blit(label_text, (self.rect.x, self.rect.y - 25))

    def handle_event(self, event):
        if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
            if self.rect.collidepoint(event.pos):
                self.dragging = True

        elif (
            self.dragging
            and event.type == pygame.MOUSEBUTTONUP
            and event.button == 1
        ):
            self.dragging = False

        elif event.type == pygame.MOUSEMOTION and self.dragging:
            # Update value based on mouse position
            relative_x = max(
                0, min(1, (event.pos[0] - self.rect.left) / self.rect.width)
            )
            self.val = self.min_val + relative_x * (
                self.max_val - self.min_val
            )
            self.update_handle_position()
            return True  # Value changed

        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_r:
                self.val = 3600.0
                self.update_handle_position()
                return True

        return False


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

        self.trail_step_time = trail_step_time  # Store the time value
        self.trail_time = trail_time  # Store the time value

        self.trail_step = int(
            trail_step_time / self.sim.dt
        )  # Step between trail points
        self.trail_length = int(
            trail_time / trail_step_time
        )  # Number of positions to keep in trail

        self.speed = 1.0  # Playback speed [days\s]
        self.rotation_z = 0.0  # Rotation in-plane [rad]
        self.rotation_x = 0.0  # Rotation out-of-plane [rad]
        self._scale = scale
        self.scale = self._scale

        self._width = width
        self._height = height
        self.width = width
        self.height = height
        self.fullscreen = False
        self.screen_info: pygame.display._VidInfo

        self.frame = 0
        self.frame_t0 = time.time()
        self.last_frame_time = 0.0
        self._fractional_steps = 0.0
        self.running = True

        self.trail_cache: A
        self.focus_body_idx: int | None = None  # Index of the focused body
        self.trail_focus_body_idx: int | None = None  # Index of trail focus
        self.trail_cache_focus: int | None = None  # Last focus used for cache
        self.trail_cache_frame = 0  # Last self.frame used for cache
        self.cache_needs_update = True

        self.screen: pygame.Surface
        self.clock: pygame.time.Clock
        self.font: pygame.font.Font
        self.small_font: pygame.font.Font

        # Add sliders
        self.sliders: List[Slider] = []
        self.ui_visible: int = 2

    def create_sliders(self):
        """Create sliders for trail parameters"""
        slider_width = 200
        slider_height = 10
        start_x = 10
        start_y = self.height - 150

        # Trail step time slider (seconds between trail points)
        trail_step_slider = Slider(
            start_x,
            start_y,
            slider_width,
            slider_height,
            min_val=3600.0,
            max_val=3600.0 * 24 * 10,
            initial_val=self.trail_step_time,
            label="Trail Step",
        )

        # Trail time slider (total trail duration in seconds)
        trail_time_slider = Slider(
            start_x,
            start_y + 50,
            slider_width,
            slider_height,
            min_val=3600.0,
            max_val=3600.0 * 24 * 365.25 * 100,
            initial_val=self.trail_time,
            label="Trail Length",
        )

        self.sliders = [trail_step_slider, trail_time_slider]

    def update_trail_parameters(self):
        """Update trail parameters from slider values"""
        trail_step_changed = False
        trail_time_changed = False

        for slider in self.sliders:
            if slider.label.startswith("Trail Step"):
                if slider.val != self.trail_step_time:
                    if slider.val <= self.trail_time:
                        self.trail_step_time = slider.val
                        self.trail_step = int(
                            self.trail_step_time / self.sim.dt
                        )
                    else:
                        slider.val = self.trail_time
                        slider.update_handle_position()
                    trail_step_changed = True

            elif slider.label.startswith("Trail Length"):
                if slider.val != self.trail_time:
                    if slider.val >= self.trail_step_time:
                        self.trail_time = slider.val
                    else:
                        slider.val = self.trail_step_time
                        slider.update_handle_position()
                    trail_time_changed = True

        # Update trail length if either parameter changed
        if trail_step_changed or trail_time_changed:
            self.trail_length = int(self.trail_time / self.trail_step_time)
            self.cache_needs_update = True

    def start(self) -> None:
        pygame.init()
        self.screen = pygame.display.set_mode(
            (self.width, self.height), pygame.RESIZABLE
        )
        self.screen_info = pygame.display.Info()
        pygame.display.set_caption("Astrodynamics Simulation")
        self.clock = pygame.time.Clock()

        self.font = pygame.font.SysFont("Courier New", 24)
        self.small_font = pygame.font.SysFont("Courier New", 16)
        self.create_sliders()

        while self.running:
            self.handle_input()
            self.advance_frame()

            if self.cache_needs_update:
                self.rebuild_trail_cache()
            self.update_trail_cache()

            self.draw_frame()
            self.draw_info()

        pygame.quit()
        sys.exit()

    def handle_input(self) -> None:
        # Continuous key press handling
        keys = pygame.key.get_pressed()
        if keys[pygame.K_RIGHT]:
            self.speed *= 1.01
            self.cache_needs_update = True
        if keys[pygame.K_LEFT]:
            self.speed /= 1.01
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

        # Event handling for quit, manual reset, and mouse click
        slider_changed = False
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.running = False
            elif event.type == pygame.VIDEORESIZE:
                self.width, self.height = event.w, event.h
                self.screen = pygame.display.set_mode(
                    (self.width, self.height), pygame.RESIZABLE
                )
                self.cache_needs_update = True
                self.create_sliders()  # Recreate sliders with new dimensions
            elif event.type == pygame.MOUSEWHEEL:
                if event.y > 0:
                    self.scale *= 1.05
                elif event.y < 0:
                    self.scale /= 1.05
                self.cache_needs_update = True
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_F11:
                    self.fullscreen = not self.fullscreen
                    self.update_fullscreen()
                if event.key == pygame.K_ESCAPE and self.fullscreen:
                    self.fullscreen = False
                    self.update_fullscreen()
                if event.key == pygame.K_r:
                    self.frame = 0
                    self.scale = self._scale
                    self.rotation_x = 0
                    self.rotation_z = 0
                    self.focus_body_idx = None
                    self.trail_focus_body_idx = None
                    self.speed = 1
                if event.key == pygame.K_h:
                    self.ui_visible = (self.ui_visible - 1) % 3
                self.cache_needs_update = True
            elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                mouse_x, mouse_y = event.pos
                # body_clicked = False

                for i in range(self.sim.num_bodies):
                    body_x, body_y = self.trail_cache[-1, :2, i]
                    if (
                        abs(mouse_x - body_x) < 10
                        and abs(mouse_y - body_y) < 10
                    ):
                        # body_clicked = True

                        if self.focus_body_idx != i:
                            # First click on a different body
                            self.focus_body_idx = i
                        else:
                            # Second click on the same body
                            self.trail_focus_body_idx = i
                        break

                # # Only reset if no body was clicked
                # if not body_clicked:
                #     if self.trail_focus_body_idx is not None:
                #         # First click in empty space
                #         self.trail_focus_body_idx = None
                #     else:
                #         # Second click in empty space
                #         self.focus_body_idx = None

                self.cache_needs_update = True

            # Handle events for sliders
            for slider in self.sliders:
                if slider.handle_event(event):
                    slider_changed = True

        # Update trail parameters if sliders changed
        if slider_changed:
            self.update_trail_parameters()
            self.cache_needs_update = True

    def update_fullscreen(self):
        if self.fullscreen:
            if (self.screen.get_flags() & pygame.FULLSCREEN) == 0:
                # Update dimensions to match fullscreen
                self.width = self.screen_info.current_w
                self.height = self.screen_info.current_h
                self.screen = pygame.display.set_mode(
                    (self.width, self.height),
                    pygame.FULLSCREEN,
                )
        else:
            if (self.screen.get_flags() & pygame.FULLSCREEN) != 0:
                # Restore original window size
                self.width = self._width
                self.height = self._height
                self.screen = pygame.display.set_mode(
                    (self.width, self.height), pygame.RESIZABLE
                )

        # Recreate sliders to fit new screen size
        self.create_sliders()

    def advance_frame(self) -> None:
        # Calculate real time elapsed since last frame
        current_time = time.time()
        real_time_elapsed = current_time - self.frame_t0
        self.frame_t0 = current_time

        # Calculate how many simulation steps to advance
        days_to_advance = self.speed * real_time_elapsed
        seconds_to_advance = days_to_advance * 86400  # 86400 seconds in a day

        total_steps = seconds_to_advance / self.sim.dt + self._fractional_steps
        steps_to_advance = int(total_steps)
        self._fractional_steps = (
            total_steps - steps_to_advance
        )  # Store remainder

        if steps_to_advance > 0:
            self.frame += steps_to_advance
            if self.frame >= self.sim.steps:
                self.frame = 0
                self.cache_needs_update = True

            # Ensure frame is integer for indexing
            self.frame = int(self.frame)
            self.cache_needs_update = True

        # Store the actual frame time for FPS calculation
        self.last_frame_time = real_time_elapsed

        # Cap at 60 FPS to prevent excessive CPU usage
        if self.last_frame_time < 1 / 60:
            pygame.time.wait(int(1000 * (1 / 60 - self.last_frame_time)))
            self.last_frame_time = 1 / 60  # Use minimum frame time

    def draw_frame(self) -> None:
        self.screen.fill(VisC.black)

        self.draw_bodies()

        if self.ui_visible > 0:
            self.draw_time()
            self.draw_axes()
            self.draw_scale()

        if self.ui_visible > 1:
            self.draw_controls()
            self.draw_info()

        pygame.display.flip()

    def draw_bodies(self) -> None:
        screen_pos = self.trail_cache[-1, :, :]

        for i in range(self.sim.num_bodies):
            color = VisC.colors[i % len(VisC.colors)]

            # Only draw if the body is on screen
            screen_pos_i = list(screen_pos[:2, i])
            if self.is_on_screen(screen_pos_i):
                if self.trail_cache.shape[0] > 1:
                    lines_list = list(map(tuple, self.trail_cache[:, 0:2, i]))
                    pygame.draw.aalines(
                        self.screen, color, False, lines_list, 1
                    )
                # Draw current position
                pygame.draw.circle(self.screen, color, screen_pos_i, 3)
                if self.sim.body_list[i].name and self.ui_visible > 0:
                    text = self.small_font.render(
                        self.sim.body_list[i].name, True, VisC.white
                    )
                    self.screen.blit(
                        text, (screen_pos_i[0] + 15, screen_pos_i[1] - 10)
                    )

        self.trail_cache_frame = self.frame

    def draw_time(self) -> None:
        t = self.frame * self.sim.dt
        sim_date = datetime.strptime(
            self.sim.epoch, "%Y-%m-%d %H:%M:%S"
        ) + timedelta(seconds=t)
        date_text = self.font.render(
            f"Date: {sim_date.strftime('%Y-%m-%d %H:%M:%S')}", True, VisC.white
        )
        self.screen.blit(date_text, (10, 10))

    def draw_axes(self) -> None:
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

    def draw_scale(self) -> None:
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

    def draw_controls(self) -> None:
        help_text = self.small_font.render(
            "UP/DOWN: Zoom, LEFT/RIGHT: Speed, R: Reset", True, VisC.white
        )
        self.screen.blit(help_text, (10, self.height - 30))

        # Draw sliders
        for slider in self.sliders:
            slider.draw(self.screen, self.small_font)

    def draw_info(self) -> None:
        actual_fps = 1 / self.last_frame_time
        sim_speed_text = self.small_font.render(
            f"Sim speed: {self.speed:.2f} days/s | FPS: {actual_fps:.0f} |"
            + f" Progress: {self.frame}/{self.sim.steps}",
            True,
            VisC.white,
        )
        self.screen.blit(sim_speed_text, (10, 50))

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
        if self.trail_focus_body_idx != self.focus_body_idx:
            self.rebuild_trail_cache()
        else:
            new_cache = np.roll(self.trail_cache, -1, axis=0)
            current_pos = self.sim.mm[self.frame, 0:3, :][np.newaxis, :, :]
            if self.trail_focus_body_idx is not None:
                pos_diff = self.sim.mm[
                    self.frame, 0:3, self.trail_focus_body_idx
                ][np.newaxis, :, np.newaxis]
                current_pos = current_pos - pos_diff
            if self.focus_body_idx is not None:
                pos_diff = current_pos[-1, :, self.focus_body_idx][
                    np.newaxis, :, np.newaxis
                ]
                current_pos = current_pos - pos_diff
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
            pos_diff = self.sim.mm[
                initial_point : self.frame + 1 : self.trail_step,
                0:3,
                self.trail_focus_body_idx,
            ][:, :, np.newaxis]
            current_pos = current_pos - pos_diff
        if self.focus_body_idx is not None:
            pos_diff = current_pos[-1, :, self.focus_body_idx][
                np.newaxis, :, np.newaxis
            ]
            current_pos = current_pos - pos_diff

        scaled_pos = self.scale_pos_array(current_pos)
        n = scaled_pos.shape[0]
        new_cache[-n:, :, :] = scaled_pos
        new_cache[0:-n, :, :] = np.repeat(
            scaled_pos[-n, :, :][np.newaxis, :, :],
            self.trail_length - n,
            axis=0,
        )
        self.trail_cache = new_cache
        self.cache_needs_update = False

    def is_on_screen(self, pos, margin=100) -> bool:
        return (
            0 - margin <= pos[0] <= self.width + margin
            and 0 - margin <= pos[1] <= self.height + margin
        )
