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
    scale = 1e-9  # [?]
    trail_step_time = T.d  # [s]
    trail_length_time = T.a  # [s]
    speed = 1.0  # Playback speed [days\s]
    rotation_z = 0.0  # Rotation in-plane [rad]
    rotation_x = 0.0  # Rotation out-of-plane [rad]

    # Sizes on the screen [px]
    info_font_size = 18

    value_display_width = 175
    value_display_height = 75
    value_display_padding = 3
    value_modifier_width = 35
    value_modifier_height = 20
    value_modifier_y_offset = (
        value_display_height - value_display_padding - value_modifier_height
    )
