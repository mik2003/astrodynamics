import sys
from collections import deque

import numpy as np
import pygame

from project.data import BodyList
from project.utilities import Dir

if __name__ == "__main__":

    fname = "sun_earth.json"
    file_in = Dir.in_dir.joinpath(fname)
    file_out = Dir.out_dir.joinpath(fname)

    body_list = BodyList.load(file_in)

    # Pygame setup
    pygame.init()
    WIDTH, HEIGHT = 800, 600
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption("Astrodynamics Simulation")
    clock = pygame.time.Clock()

    # Colors
    BLACK = (0, 0, 0)
    WHITE = (255, 255, 255)
    COLORS = [(255, 215, 0), (0, 191, 255), (220, 20, 60), (34, 139, 34)]

    # Trail settings
    TRAIL_LENGTH = 10000  # Number of positions to keep in trail
    TRAIL_STEP = 100  # Step between trail points
    trails = [None] * len(body_list)

    # Simple scaling for visualization
    def scale_pos(pos, center, scale):
        return int(center[0] + pos[0] * scale), int(center[1] - pos[1] * scale)

    # Simulation parameters
    dt_sim = 1000  # simulation time step (seconds)
    sim_speed = 100  # number of simulation steps per frame

    # Font setup for timer
    font = pygame.font.SysFont(None, 36)
    small_font = pygame.font.SysFont(None, 24)

    # Main loop
    running = True
    scale = 1e-9  # Adjust for visualization
    center = (WIDTH // 2, HEIGHT // 2)
    t = 0  # initial time

    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_UP:
                    scale *= 1.1  # Zoom in
                elif event.key == pygame.K_DOWN:
                    scale /= 1.1  # Zoom out
                elif event.key == pygame.K_r:
                    # Reset simulation
                    body_list = BodyList.load(file_in)
                    trails = [
                        deque(maxlen=TRAIL_LENGTH)
                        for _ in range(len(body_list))
                    ]
                    t = 0

        for _ in range(sim_speed):
            body_list.perform_step(dt_sim)
            t += dt_sim

        # Draw
        screen.fill(BLACK)

        # Draw trails
        for i, body in enumerate(body_list):
            trails[i] = body.r[-1 - TRAIL_LENGTH : -1 : TRAIL_STEP]

            if len(trails[i]) > 1:
                # Draw trail with fading effect
                trail_points = []
                for j, pos in enumerate(trails[i]):
                    x, y = scale_pos(pos, center, scale)
                    trail_points.append((x, y))

                # Draw trail as connected lines with fading alpha
                if len(trail_points) > 1:
                    for j in range(len(trail_points) - 1):
                        # Calculate alpha based on position in trail (fade out)
                        alpha = int(255 * (j / len(trail_points)))
                        trail_color = (*COLORS[i % len(COLORS)][:3], alpha)

                        # Create a surface for the trail segment
                        trail_surface = pygame.Surface(
                            (WIDTH, HEIGHT), pygame.SRCALPHA
                        )
                        pygame.draw.line(
                            trail_surface,
                            trail_color,
                            trail_points[j],
                            trail_points[j + 1],
                            1,
                        )
                        screen.blit(trail_surface, (0, 0))

        # Draw bodies
        for i, body in enumerate(body_list):
            pos = body.r[-1].reshape((3, 1))
            x, y = scale_pos(pos, center, scale)
            color = COLORS[i % len(COLORS)]

            pygame.draw.circle(screen, color, (x, y), 1)

            # Draw body name
            if body_list[i].name:
                text = small_font.render(body_list[i].name, True, WHITE)
                screen.blit(text, (x + 15, y - 10))

        # Draw timer and simulation info
        days = t / (24 * 3600)
        hours = (days - int(days)) * 24

        timer_text = font.render(
            f"Time: {int(days)} days, {hours:.1f} hours", True, WHITE
        )
        screen.blit(timer_text, (10, 10))

        scale_text = small_font.render(f"Scale: {scale:.2e}", True, WHITE)
        screen.blit(scale_text, (10, 50))

        speed_text = small_font.render(
            f"Speed: {sim_speed}x real time", True, WHITE
        )
        screen.blit(speed_text, (10, 80))

        help_text = small_font.render("UP/DOWN: Zoom, R: Reset", True, WHITE)
        screen.blit(help_text, (10, HEIGHT - 30))

        pygame.display.flip()
        # clock.tick(60)  # Limit to 60 FPS

    pygame.quit()

    body_list.dump(file_out)

    sys.exit()
