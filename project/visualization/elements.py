from typing import List

import pygame

from project.utilities import T
from project.visualization.constants import VisC


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
        if seconds < T.d:
            unit = "h"
            value = seconds / T.h
            value_str = f"{value:.0f}"
        elif seconds < T.a:
            unit = "d"
            value = seconds / T.d
            value_str = f"{value:.1f}" if value < 10 else f"{value:.0f}"
        else:
            unit = "a"
            value = seconds / T.a
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
                self.val = T.h
                self.update_handle_position()
                return True

        return False


class ValueDisplay:
    def __init__(
        self,
        x: int,
        y: int,
        min_value: float,
        max_value: float,
        initial_value: float,
        label: str,
        modifiers: List[str],
    ) -> None:
        self.rect = pygame.Rect(
            x, y, VisC.value_display_width, VisC.value_display_height
        )
        self.x = x
        self.y = y
        self.min_val = min_value
        self.max_val = max_value
        self.val = initial_value
        self.label = label
        self.modifiers = self.init_modifiers(modifiers)

    def init_modifiers(self, modifiers: List[str]) -> List["ValueModifier"]:
        n = len(modifiers)
        spacing = (
            VisC.value_display_width
            - 2 * VisC.value_display_padding
            - n * VisC.value_modifier_width
        ) / (n - 1)
        out = []
        for i in range(n):
            x_offset = int(
                VisC.value_display_padding
                + i * (VisC.value_modifier_width + spacing)
            )
            out.append(
                ValueModifier(
                    self.x + x_offset,
                    self.y + VisC.value_modifier_y_offset,
                    self.min_val,
                    self.max_val,
                    modifiers[i],
                )
            )
        return out

    def draw(self, screen: pygame.Surface, font: pygame.font.Font) -> None:
        # Draw the main display rectangle
        pygame.draw.rect(screen, (70, 70, 100), self.rect)
        pygame.draw.rect(screen, (200, 200, 200), self.rect, 2)

        # Draw the current value text
        value_text = font.render(
            f"{self.label}: {self.val:.2f}", True, (255, 255, 255)
        )
        screen.blit(value_text, (self.rect.x, self.rect.y - 25))

        # Draw modifier buttons
        for modifier in self.modifiers:
            modifier.draw(screen, font)

    def handle_event(self, event: pygame.event.Event) -> None:
        for modifier in self.modifiers:
            modifier.handle_event(event, self)


class ValueModifier:
    def __init__(
        self,
        x: int,
        y: int,
        min_value: float,
        max_value: float,
        modifier: str,
    ) -> None:
        self.rect = pygame.Rect(
            x, y, VisC.value_modifier_width, VisC.value_modifier_height
        )
        self.min_value = min_value
        self.max_value = max_value
        self.label = modifier
        self.operation = modifier[0]
        self.operator = float(modifier[1:])

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
            color = (160, 80, 80)  # Red for decrease
        elif self.operation in ["*", "x"]:
            color = (80, 80, 160)  # Blue for multiply
        elif self.operation in ["/"]:
            color = (160, 160, 80)  # Yellow for divide
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
    ) -> None:
        if self.rect.collidepoint(event.pos):
            old_value = display.val
            display.val = self.apply(display.val)
            print(
                f"{display.label}: {old_value:.2f} -> {display.val:.2f} "
                f"(using {self.label})"
            )


def main():
    # Initialize pygame
    pygame.init()
    screen_width, screen_height = 800, 600
    screen = pygame.display.set_mode((screen_width, screen_height))
    pygame.display.set_caption("Value Display Test")
    clock = pygame.time.Clock()
    font = pygame.font.Font(None, 24)

    # Create test ValueDisplays with different modifiers
    displays = [
        ValueDisplay(
            x=50,
            y=100,
            min_value=0,
            max_value=100,
            initial_value=50,
            label="Test Value 1",
            modifiers=["+10", "-10", "x2", "/2"],
        ),
        ValueDisplay(
            x=50,
            y=200,
            min_value=0,
            max_value=1.0,
            initial_value=0.5,
            label="Probability",
            modifiers=["+0.1", "-0.1", "x1.5", "/1.5"],
        ),
        ValueDisplay(
            x=50,
            y=300,
            min_value=-100,
            max_value=100,
            initial_value=0,
            label="Range Test",
            modifiers=["+25", "-25", "*3", "/3"],
        ),
    ]

    running = True
    while running:
        screen.fill((30, 30, 30))  # Dark background

        # Handle events
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

            # Handle modifier button clicks
            if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                for display in displays:
                    display.handle_event(event)

        for display in displays:
            display.draw(screen, font)

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
