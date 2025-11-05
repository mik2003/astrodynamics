from dataclasses import dataclass

from project.utilities import T


@dataclass(frozen=True)
class VisC:
    # Colors
    black = (0, 0, 0)
    white = (255, 255, 255)
    colors = [(255, 215, 0), (0, 191, 255), (220, 20, 60), (34, 139, 34)]

    # Defaults for VisualizaitionState
    width = 800  # [px]
    height = 600  # [px]
    scale = 1e9  # [m/px]
    trail_step_time = T.d  # [s]
    trail_length_time = T.a  # [s]
    speed = 1.0  # Playback speed [days/s]
    rotation_z = 0.0  # Rotation in-plane [rad]
    rotation_x = 0.0  # Rotation out-of-plane [rad]

    # Sizes on the screen [px]
    font_size = 24
    small_font_size = 16

    info_display_y = 100
    info_display_handle_width = 25

    value_modifier_width = 40
    value_modifier_height = 20
    value_display_width = 225
    value_display_padding = 5
    value_display_height = (
        3 * value_display_padding + small_font_size + value_modifier_height
    )
    value_modifier_y_offset = (
        value_display_height - value_display_padding - value_modifier_height
    )
    slider_width = value_display_width - 6 * value_display_padding
    slider_height = 10
    slider_handle_width = 10
    slider_handle_height = slider_height + 10
    slider_y_offset = (
        value_modifier_y_offset + (value_modifier_height - slider_height) // 2
    )

    # Other constants
    max_trail_points = T.a // T.h  # Avoid memory allocation errors
