import sys
import time
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any, Tuple

import numpy as np
import pygame

from project.simulation import Simulation
from project.ui.constants import VisC
from project.ui.elements import InfoDisplay
from project.utils import FloatArray, Index, IntArray, T, ValueUnitToStr


class CircularTrailBuffer:
    def __init__(self, array: FloatArray, head: int = 0) -> None:
        self._array = array
        self._head = head
        self._len: int = array.shape[0]

    def __getitem__(self, key: Index | Tuple[Index, ...]) -> FloatArray:
        if not isinstance(key, tuple) or len(key) != 3:
            raise IndexError("Expected (index_or_slice, :, :) indexing")

        first, a1, a2 = key

        if isinstance(first, int):
            return self._array[(self._head + first) % self._len, a1, a2]

        elif isinstance(first, slice):
            start, stop, step = first.indices(self._len)
            if step != 1:
                raise NotImplementedError("Step != 1 not supported")
            length = stop - start
            phys_start = (self._head + start) % self._len
            phys_stop = (phys_start + length) % self._len
            if phys_start < phys_stop:
                return self._array[phys_start:phys_stop, a1, a2]
            else:
                return np.concatenate(
                    (self._array[phys_start:, a1, a2], self._array[:phys_stop, a1, a2]),
                    axis=0,
                )

        raise IndexError("Invalid key")

    def __setitem__(self, key: Index | Tuple[Index, ...], value: FloatArray) -> None:
        if not isinstance(key, tuple) or len(key) != 3:
            raise IndexError("Expected (index_or_slice, :, :) indexing")

        first, a1, a2 = key

        if isinstance(first, int):
            self._array[(self._head + first) % self._len, a1, a2] = value
            return

        elif isinstance(first, slice):
            start, stop, step = first.indices(self._len)
            if step != 1:
                raise NotImplementedError("Step != 1 not supported")
            length = stop - start
            if value.shape[0] != length:
                raise ValueError("Value length mismatch")
            phys_start = (self._head + start) % self._len
            phys_stop = (phys_start + length) % self._len
            if phys_start < phys_stop:
                self._array[phys_start:phys_stop, a1, a2] = value
            else:
                first_len = self._len - phys_start
                self._array[phys_start:, a1, a2] = value[:first_len]
                self._array[:phys_stop, a1, a2] = value[first_len:]

            return

        raise IndexError("Invalid key")

    def __sub__(self, other: Any) -> "CircularTrailBuffer":
        if not isinstance(other, CircularTrailBuffer):
            raise NotImplementedError

        self._array = self._array - other._array

        return self

    def _advance(self, n: int) -> None:
        self._head = (self._head + n) % self._len
        print(self._head)

    def add_points(self, points: FloatArray) -> None:
        n = points.shape[0]
        print(f"n: {n}")
        self[-n:, :, :] = points
        self._advance(n)

    @property
    def indices(self) -> IntArray:
        return (np.arange(self._len) + self._head) % self._len

    @property
    def len(self) -> int:
        return self._len


class Coord:
    def __init__(self, x: float = 0, y: float = 0, z: float = 0) -> None:
        self._x = x
        self._y = y
        self._z = z

    def __mul__(self, value: Any) -> None:
        if not isinstance(value, float):
            raise NotImplementedError

        self._x *= value
        self._y *= value
        self._z *= value

    @property
    def x(self) -> float:
        return self._x

    @property
    def y(self) -> float:
        return self._y

    @property
    def z(self) -> float:
        return self._z

    @property
    def tuple2d(self) -> Tuple[float, float]:
        return self._x, self._y

    @property
    def tuple3d(self) -> Tuple[float, float, float]:
        return self._x, self._y, self._z


@dataclass
class VisualizationState:
    running = True
    fullscreen = False
    paused = False

    width = 800  # [px]
    height = 600  # [px]
    scale = 1e9  # [m/px]
    trail_step_time = 1.0  # [s]
    trail_length_time = 1.0  # [s]
    speed = T.d  # Playback speed [s/s]
    rotation_z = 0.0  # Rotation in-plane [rad]
    rotation_x = 0.0  # Rotation out-of-plane [rad]


class VisualizationCache:
    trail: CircularTrailBuffer
    relative_trail: CircularTrailBuffer
    trail_visible: FloatArray
    trail_frame: int = 0  # frames from last trail point
    rebuild_trail = True
    rebuild_relative_trail = True
    speed: float


class Visualization:
    def __init__(
        self,
        sim: Simulation,
    ) -> None:
        self.state = VisualizationState()
        self.cache = VisualizationCache()

        self.sim = sim

        self.state.trail_step_time = VisC.trail_step_time
        self.state.trail_length_time = VisC.trail_length_time
        self.trail_step = self.calculate_trail_step()
        self.trail_length = self.calculate_trail_length()
        self._trail_points_to_add = 0
        self.trail_length_changed = True

        self.screen_info: pygame.display._VidInfo

        self.frame = 0
        self.frame_t0 = time.time()
        self.last_frame_time = 0.0
        self._fractional_steps = 0.0

        self.focus_body_idx: int | None = None  # Index of the focused body
        self.trail_focus_body_idx: int | None = None  # Index of trail focus

        self.screen: pygame.Surface
        self.clock: pygame.time.Clock
        self.font: pygame.font.Font
        self.small_font: pygame.font.Font

        # Add info
        self.info = InfoDisplay()
        self.ui_visible: bool = True

    def calculate_trail_step(self) -> int:
        return max(
            1, int(self.state.trail_step_time / self.sim.dt)
        )  # Step between trail points, at least 1

    def calculate_trail_length(self) -> int:
        return int(
            self.state.trail_length_time / self.state.trail_step_time
        )  # Number of positions to keep in trail

    def place_info(self) -> None:
        self.info.y = VisC.info_display_y

    def create_value_displays(self) -> None:
        """Create value displays for parameters"""
        self.info.add_value_display(
            min_value=0,
            max_value=self.sim.steps - 1,
            initial_value=0,
            label="Step",
            modifiers=["slider"],
            unit=f"/{self.sim.steps}",
            str_format="{:.0f}",
        )

        self.info.add_value_display(
            min_value=0.0,
            max_value=self.sim.time,
            initial_value=T.d,
            label="Speed",
            modifiers=["/10", "/1.1", "*1.1", "*10"],
            unit="s_per_s",
        )

        self.info.add_value_display(
            min_value=sys.float_info.min,
            max_value=sys.float_info.max,
            initial_value=self.state.scale,
            label="Scale",
            modifiers=["/10", "/1.1", "*1.1", "*10"],
            unit="m_per_px",
            str_format="{:.2e}",
        )

        self.info.add_value_display(
            min_value=self.sim.dt,
            max_value=self.sim.time,
            initial_value=self.state.trail_step_time,
            label="Trail Step",
            modifiers=["/10", "/1.1", "*1.1", "*10"],
            unit="s",
        )
        self.info.add_value_display(
            min_value=self.sim.dt,
            max_value=self.sim.time,
            initial_value=self.state.trail_length_time,
            label="Trail Length",
            modifiers=["/10", "/1.1", "*1.1", "*10"],
            unit="s",
        )

        self.place_info()
        self.info.toggle_show()

    def update_parameters(self) -> None:
        """Update trail parameters from value displays"""
        trail_step_changed = False
        trail_time_changed = False

        for value_display in self.info.value_displays:
            if value_display.label == "Step":
                if value_display.val != self.frame:
                    # Value changed
                    self.frame = int(value_display.val)
                    self.cache.rebuild_relative_trail = True
                    self.cache.rebuild_trail = True

            elif value_display.label == "Speed":
                if self.state.paused:
                    self.cache.speed = value_display.val
                else:
                    self.state.speed = value_display.val

            elif value_display.label == "Scale":
                self.state.scale = value_display.val

            elif value_display.label == "Trail Step":
                if value_display.val != self.state.trail_step_time:
                    # Value changed
                    if value_display.val <= self.state.trail_length_time:
                        # Trail step is smaller than trail length
                        if (
                            self.state.trail_length_time / value_display.val
                            > VisC.max_trail_points
                        ):
                            # Maximum trail points reached
                            value_display.val = (
                                self.state.trail_length_time / VisC.max_trail_points
                            )
                    else:
                        # Trail step is larger than trail length
                        value_display.val = self.state.trail_length_time
                    self.state.trail_step_time = value_display.val
                    trail_step_changed = True

            elif value_display.label.startswith("Trail Length"):
                if value_display.val != self.state.trail_length_time:
                    # Value changed
                    if value_display.val >= self.state.trail_step_time:
                        # Trail length is larger than trail step
                        if (
                            value_display.val / self.state.trail_step_time
                            > VisC.max_trail_points
                        ):
                            # Maximum trail points reached
                            value_display.val = (
                                self.state.trail_step_time * VisC.max_trail_points
                            )
                    else:
                        # Trail length is smaller than trail step
                        value_display.val = self.state.trail_step_time
                    self.state.trail_length_time = value_display.val
                    trail_time_changed = True

        # Update trail length if either parameter changed
        if trail_step_changed or trail_time_changed:
            self.trail_step = self.calculate_trail_step()
            self.trail_length = self.calculate_trail_length()
            self.trail_length_changed = True
            # self.cache.rebuild_relative_trail = True
            # self.cache.rebuild_trail = True

    def reset_parameters(self) -> None:
        """Reset parameters"""
        self.frame = 0
        self.state.scale = VisC.scale
        self.state.rotation_x = 0
        self.state.rotation_z = 0
        self.focus_body_idx = None
        self.trail_focus_body_idx = None
        self.state.speed = T.d

        for value_display in self.info.value_displays:
            if value_display.label == "Step":
                value_display.val = self.frame
                if value_display.slider:
                    value_display.slider.val = self.frame
                    value_display.slider.update_handle_position()

            if value_display.label == "Speed":
                value_display.val = self.state.speed

            if value_display.label == "Trail Step":
                value_display.val = self.state.trail_step_time

            if value_display.label.startswith("Trail Length"):
                value_display.val = self.state.trail_length_time

        self.cache.rebuild_relative_trail = True
        self.cache.rebuild_trail = True

    def start(self) -> None:
        pygame.init()
        self.screen = pygame.display.set_mode(
            (self.state.width, self.state.height), pygame.RESIZABLE
        )
        self.screen_info = pygame.display.Info()
        pygame.display.set_caption("Astrodynamics Simulation")
        self.clock = pygame.time.Clock()

        self.font = pygame.font.SysFont("Courier New", VisC.font_size)
        self.small_font = pygame.font.SysFont("Courier New", VisC.small_font_size)
        self.create_value_displays()

        while self.state.running:
            self.handle_input()
            self.advance_frame()
            self.draw_frame()
            self.draw_info()

        pygame.quit()
        sys.exit()

    def handle_input(self) -> None:
        # Continuous key press handling
        keys = pygame.key.get_pressed()

        if keys[pygame.K_RIGHT]:
            self.state.speed *= 1.01
        if keys[pygame.K_LEFT]:
            self.state.speed /= 1.01

        if keys[pygame.K_UP]:
            self.state.scale /= 1.01
            self.cache.rebuild_trail = True
        if keys[pygame.K_DOWN]:
            self.state.scale *= 1.01
            self.cache.rebuild_trail = True

        # Rotate view left/right (Z axis) and tilt up/down (X axis)
        if keys[pygame.K_d]:
            self.state.rotation_z -= 0.02  # radians per frame
            self.cache.rebuild_trail = True
        if keys[pygame.K_a]:
            self.state.rotation_z += 0.02
            self.cache.rebuild_trail = True
        if keys[pygame.K_w]:
            self.state.rotation_x -= 0.02
            self.cache.rebuild_trail = True
        if keys[pygame.K_s]:
            self.state.rotation_x += 0.02
            self.cache.rebuild_trail = True

        if any(keys):
            self.update_info_display()

        # Event handling for quit, manual reset, and mouse click
        info_changed = self.frame == 0
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.state.running = False
            elif event.type == pygame.VIDEORESIZE:
                self.state.width, self.state.height = event.w, event.h
                self.screen = pygame.display.set_mode(
                    (self.state.width, self.state.height), pygame.RESIZABLE
                )
                self.cache.rebuild_trail = True
                self.place_info()
            elif event.type == pygame.MOUSEWHEEL:
                if event.y > 0:
                    self.state.scale /= 1.05
                elif event.y < 0:
                    self.state.scale *= 1.05
                if event.x < 0:
                    self.state.speed /= 1.05
                elif event.x > 0:
                    self.state.speed *= 1.05
                self.update_info_display()
                self.cache.rebuild_trail = True
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_F11:
                    self.state.fullscreen = not self.state.fullscreen
                    self.update_fullscreen()
                if event.key == pygame.K_ESCAPE and self.state.fullscreen:
                    self.state.fullscreen = False
                    self.update_fullscreen()
                if event.key == pygame.K_r:
                    self.reset_parameters()
                if event.key == pygame.K_h:
                    self.ui_visible = not self.ui_visible
                if event.key == pygame.K_SPACE:
                    self.pause()
            elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                mouse_x, mouse_y = event.pos
                # body_clicked = False

                for i in range(self.sim.num_bodies):
                    body_x, body_y = self.cache.trail[-1, :2, i]
                    if abs(mouse_x - body_x) < 10 and abs(mouse_y - body_y) < 10:
                        # body_clicked = True

                        if self.focus_body_idx != i:
                            # First click on a different body
                            self.focus_body_idx = i
                        else:
                            # Second click on the same body
                            self.trail_focus_body_idx = i
                        break

                self.cache.rebuild_relative_trail = True
                self.cache.rebuild_trail = True

            info_changed = self.info.handle_event(event=event)

        if info_changed:
            self.update_parameters()
            # self.cache.rebuild_relative_trail = True
            # self.cache.rebuild_trail = True

    def update_fullscreen(self) -> None:
        if self.state.fullscreen:
            if (self.screen.get_flags() & pygame.FULLSCREEN) == 0:
                # Update dimensions to match fullscreen
                self.state.width = self.screen_info.current_w
                self.state.height = self.screen_info.current_h
                self.screen = pygame.display.set_mode(
                    (self.state.width, self.state.height),
                    pygame.FULLSCREEN,
                )
        else:
            if (self.screen.get_flags() & pygame.FULLSCREEN) != 0:
                # Restore original window size
                self.state.width = VisC.width
                self.state.height = VisC.height
                self.screen = pygame.display.set_mode(
                    (self.state.width, self.state.height), pygame.RESIZABLE
                )

        self.cache.rebuild_trail = True

    def update_info_display(self) -> None:
        """Update value display parameters from value displays"""
        for value_display in self.info.value_displays:
            if value_display.label == "Speed":
                value_display.val = self.state.speed
            elif value_display.label == "Scale":
                value_display.val = self.state.scale

    def pause(self) -> None:
        if not self.state.paused:
            self.state.paused = True
            self.cache.speed = self.state.speed
            self.state.speed = 0
        else:
            self.state.paused = False
            self.state.speed = self.cache.speed

    def advance_frame(self) -> None:
        # Calculate real time elapsed since last frame
        current_time = time.time()
        real_time_elapsed = current_time - self.frame_t0
        self.frame_t0 = current_time

        # Calculate how many simulation steps to advance
        seconds_to_advance = self.state.speed * real_time_elapsed

        total_steps = seconds_to_advance / self.sim.dt + self._fractional_steps
        steps_to_advance = int(total_steps)
        self._fractional_steps = total_steps - steps_to_advance  # Store remainder

        if steps_to_advance > 0:
            self.frame += steps_to_advance
            self.cache.trail_frame += steps_to_advance
            if self.frame >= self.sim.steps:
                self.frame = 0
                self.cache.rebuild_relative_trail = True
                self.cache.rebuild_trail = True

            # Ensure frame is integer for indexing
            self.frame = int(self.frame)

        # Store the actual frame time for FPS calculation
        self.last_frame_time = real_time_elapsed

        # Cap at 60 FPS to prevent excessive CPU usage
        fps = 60
        if self.last_frame_time < 1 / fps:
            pygame.time.wait(int(1000 * (1 / fps - self.last_frame_time)))
            self.last_frame_time = 1 / fps  # Use minimum frame time

        # Advance step slider
        for value_display in self.info.value_displays:
            if value_display.label == "Step":
                value_display.val = self.frame
                if value_display.slider:
                    value_display.slider.val = self.frame
                    value_display.slider.update_handle_position()

        # Update the trail
        self.update_trail()

    def update_trail(self) -> None:
        if self.trail_length_changed:
            self.build_trail_cache()
        if self.cache.rebuild_relative_trail:
            self.rebuild_relative_trail_cache()
        else:
            self.update_relative_trail_cache()
        if self.cache.rebuild_trail:
            self.rebuild_trail_cache()
        else:
            self.update_trail_cache()

    def draw_frame(self) -> None:
        self.screen.fill(VisC.black)

        self.draw_bodies()

        if self.ui_visible:
            self.draw_time()
            self.draw_axes()
            self.draw_scale()
            self.draw_controls()
            self.draw_info()

        pygame.display.flip()

    def draw_bodies(self) -> None:
        for i in range(self.sim.num_bodies):
            if not self.cache.trail_visible[i]:
                continue

            color = VisC.colors[i % len(VisC.colors)]

            if self.cache.trail.len > 1:
                pts = self.cache.trail[:, :2, i].T
                points = pts.astype(int).T.tolist()
                pygame.draw.aalines(self.screen, color, False, points, 1)
            # Draw current position
            actual_radius = self.sim.body_list[i].radius
            actual_radius = actual_radius if actual_radius is not None else 0.0
            radius = max(
                3,
                int(actual_radius / self.state.scale),
            )
            screen_pos_i = self.cache.trail[-1, :2, i].tolist()
            pygame.draw.circle(self.screen, color, screen_pos_i, radius)
            if self.sim.body_list[i].name and self.ui_visible:
                text = self.small_font.render(
                    self.sim.body_list[i].name, True, VisC.white
                )
                self.screen.blit(
                    text,
                    (screen_pos_i[0] + 12 + radius, screen_pos_i[1] - 10),
                )

    def draw_time(self) -> None:
        t = self.frame * self.sim.dt
        if self.sim.epoch:
            sim_date = datetime.strptime(
                self.sim.epoch, "%Y-%m-%d %H:%M:%S"
            ) + timedelta(seconds=t)
            date_text = self.font.render(
                f"Date: {sim_date.strftime('%Y-%m-%d %H:%M:%S')}",
                True,
                VisC.white,
            )
        else:
            label_str = "Date: "
            secs = T.a
            if t >= secs:
                label_str += f"{(t / secs):.0f}:"
            secs = T.d
            if t >= secs:
                label_str += f"{(t / secs) % 365.25:.0f}:"
            secs = T.h
            if t >= secs:
                label_str += f"{(t / secs) % 24:.0f}:"
            secs = T.m
            if t >= secs:
                label_str += f"{(t / secs) % 60:.0f}:"
            label_str += f"{t % 60:.1f}"
            date_text = self.font.render(
                label_str,
                True,
                VisC.white,
            )
        self.screen.blit(date_text, (10, 10))

    def draw_axes(self) -> None:
        # Draw axis system (X: red, Y: green, Z: blue)
        axis_length = 50 * self.state.scale  # world units, so it scales with zoom
        axes = [Coord(x=axis_length), Coord(y=axis_length), Coord(z=axis_length)]
        axis_colors = [(255, 0, 0), (0, 255, 0), (0, 128, 255)]
        axis_labels = ["X", "Y", "Z"]
        for i, axis in enumerate(axes):
            axes_center = Coord(self.state.width - 100, self.state.height - 100)
            end = self.scale_pos(axis, axes_center, self.state.scale)
            start = axes_center.tuple2d
            pygame.draw.line(self.screen, axis_colors[i], start, end, 2)
            # Draw label at the end of each axis
            label_text = self.small_font.render(axis_labels[i], True, axis_colors[i])
            offset = 10  # pixels away from axis end
            label_pos = (end[0] + offset, end[1] + offset)
            self.screen.blit(label_text, label_pos)

    def draw_scale(self) -> None:
        bar_px = 100
        bar_height = 4
        bar_color = VisC.white
        margin = 30
        bar_x1 = self.state.width - margin - bar_px
        bar_x2 = self.state.width - margin
        bar_y = self.state.height - margin

        length = bar_px * self.state.scale
        pygame.draw.line(
            self.screen,
            bar_color,
            (bar_x1, bar_y),
            (bar_x2, bar_y),
            bar_height,
        )
        label_text = self.small_font.render(ValueUnitToStr.m(length), True, VisC.white)
        label_rect = label_text.get_rect(center=((bar_x1 + bar_x2) // 2, bar_y - 15))
        self.screen.blit(label_text, label_rect)

    def draw_controls(self) -> None:
        help_text = self.small_font.render(
            "UP/DOWN: Zoom, LEFT/RIGHT: Speed, R: Reset", True, VisC.white
        )
        self.screen.blit(help_text, (10, self.state.height - 30))

        self.info.draw(screen=self.screen, font=self.small_font)

    def draw_info(self) -> None:
        actual_fps = 1 / self.last_frame_time
        sim_speed_text = self.small_font.render(
            f"FPS: {actual_fps:.0f}",
            True,
            VisC.white,
        )
        self.screen.blit(sim_speed_text, (10, 50))

    def scale_pos_array(self, pos: FloatArray) -> FloatArray:
        # Apply Z rotation (in-plane)
        cos_z = np.cos(self.state.rotation_z)
        sin_z = np.sin(self.state.rotation_z)
        x = pos[:, 0, :]
        y = pos[:, 1, :]
        z = pos[:, 2, :] if pos.shape[1] > 2 else 0
        x_rot = x * cos_z - y * sin_z
        y_rot = x * sin_z + y * cos_z
        # Apply X rotation (tilt)
        cos_x = np.cos(self.state.rotation_x)
        sin_x = np.sin(self.state.rotation_x)
        y_tilt = y_rot * cos_x - z * sin_x
        # z_tilt = y_rot * sin_x + z * cos_x
        result = np.empty_like(pos)
        result[:, 0, :] = self.state.width // 2 + x_rot / self.state.scale
        result[:, 1, :] = self.state.height // 2 - y_tilt / self.state.scale
        result[:, 2, :] = 0  # Z (2D projection)
        return result.astype(int)

    def scale_pos(self, pos: Coord, center: Coord, scale: float) -> Tuple[int, int]:
        # Apply Z rotation (in-plane)
        cos_z = np.cos(self.state.rotation_z)
        sin_z = np.sin(self.state.rotation_z)
        x_rot = pos.x * cos_z - pos.y * sin_z
        y_rot = pos.x * sin_z + pos.y * cos_z
        # Apply X rotation (tilt)
        cos_x = np.cos(self.state.rotation_x)
        sin_x = np.sin(self.state.rotation_x)
        y_tilt = y_rot * cos_x - pos.z * sin_x
        # z_tilt = y_rot * sin_x + z * cos_x  # not used for 2D
        return int(center.x + x_rot / scale), int(center.y - y_tilt / scale)

    def build_trail_cache(self) -> None:
        """Allocate trail cache and build"""
        self.rebuild_relative_trail_cache()
        self.rebuild_trail_cache()
        self.trail_length_changed = False

    def update_relative_trail_cache(self) -> None:
        """Update the relative trail cache with new positions."""

        # Calculate how many new trail points we need to add
        trail_points_to_add = self.cache.trail_frame // self.trail_step
        trail_frame_remainder = self.cache.trail_frame % self.trail_step
        self._trail_points_to_add = trail_points_to_add

        if trail_points_to_add >= self.trail_length:
            # Replace entire cache
            self.rebuild_relative_trail_cache()
            return

        if trail_points_to_add > 0:
            # Get the positions to add (from the simulation data)
            start_frame = self.frame - self.cache.trail_frame
            end_frame = start_frame + trail_points_to_add * self.trail_step
            step = self.trail_step

            # Ensure we don't go out of bounds
            start_frame = max(0, start_frame)
            end_frame = min(self.sim.steps, end_frame)

            if start_frame < end_frame:
                positions_to_add = self.sim.mm.r_vis[start_frame:end_frame:step, :]

                # Apply focus offset if needed
                if self.trail_focus_body_idx is not None:
                    focus_positions = self.sim.mm.r_vis[
                        start_frame:end_frame:step, self.trail_focus_body_idx
                    ]
                    positions_to_add = positions_to_add - focus_positions

                # Add new positions
                self.cache.relative_trail.add_points(positions_to_add)

        # Always update the very latest position
        current_pos = self.sim.mm.r_vis[self.frame, :]

        if self.trail_focus_body_idx is not None:
            focus_pos = self.sim.mm.r_vis[self.frame, self.trail_focus_body_idx]
            current_pos = current_pos - focus_pos

        self.cache.relative_trail[-1, :, :] = current_pos
        self.cache.trail_frame = trail_frame_remainder

    def update_trail_cache(self) -> None:
        """Update the display trail cache from the relative trail cache."""

        # Calculate how many points to update
        trail_points_to_update = self._trail_points_to_add

        # Check if we need a complete rebuild
        needs_rebuild = (
            self.trail_focus_body_idx != self.focus_body_idx
            or self.cache.rebuild_trail
            or trail_points_to_update >= self.trail_length
        )

        if needs_rebuild:
            self.rebuild_trail_cache()
            return

        if trail_points_to_update > 0:
            # Get the positions from relative trail cache that need updating
            start_idx = -trail_points_to_update  # -1 for the current position
            positions_to_update = self.cache.relative_trail[start_idx:, :, :]

            # Apply current focus offset if needed
            if self.focus_body_idx is not None:
                focus_offset = positions_to_update[-1, :, self.focus_body_idx][
                    np.newaxis, :, np.newaxis
                ]
                positions_to_update = positions_to_update - focus_offset

            # Scale positions for display
            scaled_positions = self.scale_pos_array(positions_to_update)

            # Add new positions
            self.cache.trail.add_points(scaled_positions)

        # Always update the current position
        current_relative_pos = self.cache.relative_trail[-1, :, :][np.newaxis, :, :]

        if self.focus_body_idx is not None:
            focus_offset = current_relative_pos[-1, :, self.focus_body_idx][
                np.newaxis, :, np.newaxis
            ]
            current_relative_pos = current_relative_pos - focus_offset

        scaled_current_pos = self.scale_pos_array(current_relative_pos)
        self.cache.trail[-1, :, :] = scaled_current_pos[0]  # Remove the extra dimension

    def rebuild_relative_trail_cache(self) -> None:
        new_cache = np.empty((self.trail_length, 3, self.sim.num_bodies))
        initial_point = max(0, self.frame - self.trail_length * self.trail_step + 1)
        current_pos = self.sim.mm.r_vis[
            initial_point : self.frame + 1 : self.trail_step, :
        ]

        if self.trail_focus_body_idx is not None:
            pos_diff = self.sim.mm.r_vis[
                initial_point : self.frame + 1 : self.trail_step,
                self.trail_focus_body_idx,
            ]
            current_pos = current_pos - pos_diff

        n = current_pos.shape[0]
        new_cache[-n:, :, :] = current_pos
        new_cache[0:-n, :, :] = np.repeat(
            current_pos[-n, :, :][np.newaxis, :, :],
            self.trail_length - n,
            axis=0,
        )

        self.cache.relative_trail = CircularTrailBuffer(np.roll(new_cache, -1, axis=0))

        # Always update the very latest position
        current_pos = self.sim.mm.r_vis[self.frame, :]

        if self.trail_focus_body_idx is not None:
            focus_pos = self.sim.mm.r_vis[self.frame, self.trail_focus_body_idx]
            current_pos = current_pos - focus_pos

        self.cache.relative_trail[-1, :, :] = current_pos

        self.cache.rebuild_relative_trail = False
        self.cache.trail_frame = (self.frame - initial_point) % self.trail_step

    def rebuild_trail_cache(self) -> None:
        new_cache = np.empty((self.trail_length, 3, self.sim.num_bodies))
        current_pos = self.cache.relative_trail[:, :, :]
        if self.focus_body_idx is not None:
            pos_diff = current_pos[-1, :, self.focus_body_idx][
                np.newaxis, :, np.newaxis
            ]
            current_pos = current_pos - pos_diff

        scaled_pos = self.scale_pos_array(current_pos)
        new_cache[:, :, :] = scaled_pos
        self.cache.trail = CircularTrailBuffer(new_cache)
        self.cache.rebuild_trail = False

        self.update_trail_visibility()

    def update_trail_visibility(self) -> None:
        trail = self.cache.trail  # (T, 2, N)

        x = trail[:, 0, :]
        y = trail[:, 1, :]

        visible = (
            (x >= 0) & (x <= self.state.width) & (y >= 0) & (y <= self.state.height)
        )

        # Any point of the trail on screen
        self.cache.trail_visible = np.any(visible, axis=0)
