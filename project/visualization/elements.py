from typing import List

import pygame

from project.utilities import ValueUnitToStr
from project.visualization.constants import VisC


class InfoDisplay:
    def __init__(
        self,
        x: int = 0,
        y: int = 0,
    ) -> None:
        self._x = 0
        self._y = 0

        self.rect = pygame.Rect(0, 0, 0, 0)
        self.handle_rect = pygame.Rect(0, 0, VisC.info_display_handle_width, 0)

        self.n = 0
        self.value_displays: List["ValueDisplay"] = []
        self.x = x
        self.y = y
        self.show = True

    @property
    def x(self) -> int:
        return self._x

    @x.setter
    def x(self, value: int) -> None:
        if not isinstance(value, int):
            raise ValueError
        self._x = value
        self.rect.x = value
        self.handle_rect.x = value + self.rect.width
        self.place_value_displays()

    @property
    def y(self) -> int:
        return self._y

    @y.setter
    def y(self, value: int) -> None:
        if not isinstance(value, int):
            raise ValueError
        self._y = value
        self.rect.y = value
        self.handle_rect.y = value
        self.place_value_displays()

    def place_value_displays(self) -> None:
        for i, value_display in enumerate(self.value_displays):
            value_display.x = self.x
            value_display.y = self.y + i * VisC.value_display_height

    def add_value_display(
        self,
        min_value: float,
        max_value: float,
        initial_value: float,
        label: str,
        modifiers: List[str],
        unit: str | None = None,
        str_format: str = "{:.2f}",
    ) -> None:
        self.value_displays.append(
            ValueDisplay(
                min_value=min_value,
                max_value=max_value,
                initial_value=initial_value,
                label=label,
                modifiers=modifiers,
                unit=unit,
                str_format=str_format,
            )
        )
        self.rect.height = self.rect.height + VisC.value_display_height
        self.rect.width = max(self.rect.width, VisC.value_display_width)
        self.handle_rect.x = self.rect.x + self.rect.width
        self.handle_rect.height = self.rect.height
        self.n += 1
        self.place_value_displays()

    def draw(
        self,
        screen: pygame.Surface,
        font: pygame.font.Font,
    ) -> None:
        pygame.draw.rect(screen, (100, 100, 255), self.handle_rect)
        # Render the text
        text = font.render("Parameters", True, (255, 255, 255))
        rotated_text = pygame.transform.rotate(text, -90)
        rotated_rect = rotated_text.get_rect(center=self.handle_rect.center)
        screen.blit(rotated_text, rotated_rect)

        if self.show:
            # Draw the main display rectangle
            s = pygame.Surface(
                (VisC.value_display_width, self.n * VisC.value_display_height),
                pygame.SRCALPHA,
            )  # per-pixel alpha
            s.fill((100, 100, 255, 50))  # alpha
            screen.blit(s, (self.x, self.y))

            for value_display in self.value_displays:
                value_display.draw(screen=screen, font=font)

    def handle_event(self, event: pygame.event.Event) -> bool:
        changed = False
        for value_display in self.value_displays:
            changed = changed or value_display.handle_event(event=event)
        if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
            if self.handle_rect.collidepoint(event.pos):
                if self.show:
                    self.show = False
                    self.x = self.x - self.rect.width
                else:
                    self.show = True
                    self.x = self.x + self.rect.width
        return changed


class ValueDisplay:
    def __init__(
        self,
        min_value: float,
        max_value: float,
        initial_value: float,
        label: str,
        modifiers: List[str],
        unit: str | None = None,
        x: int = 0,
        y: int = 0,
        str_format: str = "{:.2f}",
    ) -> None:
        self._x = 0
        self._y = 0

        self.rect = pygame.Rect(
            0, 0, VisC.value_display_width, VisC.value_display_height
        )
        self.min_val = min_value
        self.max_val = max_value
        self.val = initial_value
        self.label = label
        self.unit = unit
        self.str_format = str_format

        self.modifiers: List["ValueModifier"] = []
        self.slider: Slider | None = None

        self.init_modifiers(modifiers)

        self.x = x
        self.y = y

    @property
    def x(self) -> int:
        return self._x

    @x.setter
    def x(self, value: int) -> None:
        if not isinstance(value, int):
            raise ValueError
        self._x = value
        self.rect.x = value
        self.place_modifiers()

    @property
    def y(self) -> int:
        return self._y

    @y.setter
    def y(self, value: int) -> None:
        if not isinstance(value, int):
            raise ValueError
        self._y = value
        self.rect.y = value
        self.place_modifiers()

    def init_modifiers(self, modifiers: List[str]) -> None:
        for modifier in modifiers:
            if modifier.startswith("slider"):
                self.slider = Slider(
                    min_value=self.min_val,
                    max_value=self.max_val,
                    initial_value=self.val,
                )
            else:
                self.modifiers.append(
                    ValueModifier(
                        min_value=self.min_val,
                        max_value=self.max_val,
                        modifier=modifier,
                    )
                )

    def place_modifiers(self) -> None:
        for i, modifier in enumerate(self.modifiers):
            x_offset = int(
                VisC.value_display_padding
                + i * (VisC.value_modifier_width + VisC.value_display_padding)
            )
            modifier.x = self.x + x_offset
            modifier.y = self.y + VisC.value_modifier_y_offset

        if self.slider:
            self.slider.x = self.x + 3 * VisC.value_display_padding
            self.slider.y = self.y + VisC.slider_y_offset

    def draw(
        self,
        screen: pygame.Surface,
        font: pygame.font.Font,
    ) -> None:
        # Draw the current value text
        if self.unit and hasattr(ValueUnitToStr, self.unit):
            value = getattr(ValueUnitToStr, self.unit)(
                value=self.val, str_format=self.str_format
            )
        else:
            value = self.str_format.format(self.val) + (
                self.unit if self.unit else ""
            )
        value_text = font.render(
            f"{self.label}: {value}", True, (255, 255, 255)
        )
        screen.blit(
            value_text,
            (
                self.rect.x + VisC.value_display_padding,
                self.rect.y + VisC.value_display_padding,
            ),
        )

        # Draw modifier buttons
        for modifier in self.modifiers:
            modifier.draw(screen, font)

        if self.slider:
            self.slider.draw(screen)

    def handle_event(self, event: pygame.event.Event) -> bool:
        changed = False
        for modifier in self.modifiers:
            changed = changed or modifier.handle_event(event, self)
        if self.slider:
            changed = changed or self.slider.handle_event(event, self)
        return changed


class ValueModifier:
    def __init__(
        self,
        min_value: float,
        max_value: float,
        modifier: str,
        x: int = 0,
        y: int = 0,
    ) -> None:
        self._x = 0
        self._y = 0

        self.rect = pygame.Rect(
            0, 0, VisC.value_modifier_width, VisC.value_modifier_height
        )
        self.min_value = min_value
        self.max_value = max_value
        self.label = modifier
        self.operation = modifier[0]
        self.operator = float(modifier[1:])

        self.x = x
        self.y = y

    @property
    def x(self) -> int:
        return self._x

    @x.setter
    def x(self, value: int) -> None:
        if not isinstance(value, int):
            raise ValueError
        self._x = value
        self.rect.x = value

    @property
    def y(self) -> int:
        return self._y

    @y.setter
    def y(self, value: int) -> None:
        if not isinstance(value, int):
            raise ValueError
        self._y = value
        self.rect.y = value

    def apply(self, operand: float) -> float:
        if self.operation == "+":
            return min(operand + self.operator, self.max_value)
        elif self.operation == "-":
            return max(operand - self.operator, self.min_value)
        elif self.operation in ["*", "x"]:
            if self.operator >= 1:
                return min(operand * self.operator, self.max_value)
            else:
                return max(operand * self.operator, self.min_value)
        elif self.operation in ["/"]:
            if self.operator >= 1:
                return max(operand / self.operator, self.min_value)
            else:
                return min(operand / self.operator, self.max_value)
        else:
            raise ValueError("Operation not defined.")

    def draw(self, screen: pygame.Surface, font: pygame.font.Font) -> None:
        # Button color based on operation type
        if self.operation == "+":
            color = (80, 160, 80)  # Green for increase
        elif self.operation == "-":
            color = (160, 160, 80)  # Yellow for decrease
        elif self.operation in ["*", "x"]:
            color = (80, 80, 160)  # Blue for multiply
        elif self.operation in ["/"]:
            color = (160, 80, 80)  # Red for divide
        else:
            color = (0, 0, 0)

        pygame.draw.rect(screen, color, self.rect)
        pygame.draw.rect(screen, (200, 200, 200), self.rect, 1)

        # Draw modifier label
        mod_text = font.render(self.label, True, (255, 255, 255))
        text_rect = mod_text.get_rect(center=self.rect.center)
        screen.blit(mod_text, text_rect)

    def handle_event(
        self, event: pygame.event.Event, display: ValueDisplay
    ) -> bool:
        if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
            if self.rect.collidepoint(event.pos):
                display.val = self.apply(display.val)
                return True
        return False


class Slider:
    def __init__(
        self,
        min_value: float,
        max_value: float,
        initial_value: float,
        x: int = 0,
        y: int = 0,
    ) -> None:
        self._x = 0
        self._y = 0

        self.rect = pygame.Rect(0, 0, VisC.slider_width, VisC.slider_height)
        self.handle_rect = pygame.Rect(
            0, 0, VisC.slider_handle_width, VisC.slider_handle_height
        )
        self.min_value = min_value
        self.max_value = max_value
        self.val = initial_value
        self.dragging = False

        self.x = x
        self.y = y

    @property
    def x(self) -> int:
        return self._x

    @x.setter
    def x(self, value: int) -> None:
        if not isinstance(value, int):
            raise ValueError
        self._x = value
        self.rect.x = value
        self.update_handle_position()

    @property
    def y(self) -> int:
        return self._y

    @y.setter
    def y(self, value: int) -> None:
        if not isinstance(value, int):
            raise ValueError
        self._y = value
        self.rect.y = value
        self.handle_rect.y = (
            value - (VisC.slider_handle_height - VisC.slider_height) // 2
        )

    def update_handle_position(self):
        relative_val = (self.val - self.min_value) / (
            self.max_value - self.min_value
        )
        self.handle_rect.centerx = (
            self.rect.left + relative_val * self.rect.width
        )

    def draw(self, screen):
        # Draw slider track
        pygame.draw.rect(screen, (100, 100, 100), self.rect)
        pygame.draw.rect(screen, (200, 200, 200), self.rect, 2)

        # Draw handle
        pygame.draw.rect(screen, (255, 255, 255), self.handle_rect)
        pygame.draw.rect(screen, (150, 150, 150), self.handle_rect, 2)

    def handle_event(
        self, event: pygame.event.Event, display: ValueDisplay
    ) -> bool:
        if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
            if self.rect.collidepoint(event.pos):
                self.dragging = True
                return True

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
            self.val = self.min_value + relative_x * (
                self.max_value - self.min_value
            )
            self.update_handle_position()
            display.val = self.val
            return True  # Value changed

        elif event.type == pygame.KEYDOWN:
            if event.key == pygame.K_r:
                self.val = 1
                self.update_handle_position()
                return True

        return False


def main():
    # Initialize pygame
    pygame.init()
    screen_width, screen_height = 800, 600
    screen = pygame.display.set_mode((screen_width, screen_height))
    pygame.display.set_caption("Value Display Test")
    clock = pygame.time.Clock()
    font = pygame.font.Font(None, VisC.small_font_size)

    # Create test ValueDisplays with different modifiers
    info = InfoDisplay(x=100, y=100)
    info.add_value_display(
        min_value=0,
        max_value=100,
        initial_value=50,
        label="Test Value 1",
        modifiers=["+10", "-10", "x2", "/2"],
    )
    info.add_value_display(
        min_value=0,
        max_value=1.0,
        initial_value=0.5,
        label="Probability",
        modifiers=["+0.1", "-0.1", "x1.5", "/1.5"],
    )
    info.add_value_display(
        min_value=1e-25,
        max_value=1e25,
        initial_value=1,
        label="Range Test",
        modifiers=["/10", "/2", "/1.1", "*1.1", "*2", "*10"],
        unit="m",
    )

    running = True
    while running:
        screen.fill((30, 30, 30))  # Dark background

        # Handle events
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

            # Handle modifier button clicks
            if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                info.handle_event(event=event)

        info.draw(screen, font)

        # Draw instructions
        instructions = [
            "Click modifier buttons to change values",
            "Green: Add, Red: Subtract, Blue: Multiply, Yellow: Divide",
            "Use slider for comparison",
        ]
        for i, instruction in enumerate(instructions):
            text = font.render(instruction, True, (200, 200, 200))
            screen.blit(text, (50, 500 + i * 30))

        pygame.display.flip()
        clock.tick(60)

    pygame.quit()


if __name__ == "__main__":
    main()
