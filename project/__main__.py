import sys

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

    # Simple scaling for visualization
    def scale_pos(pos, center, scale):
        return int(center[0] + pos[0, 0] * scale), int(
            center[1] - pos[1, 0] * scale
        )

    # Simulation parameters
    dt_sim = 10  # simulation time step (seconds)
    sim_speed = 1000  # number of simulation steps per frame

    # Main loop
    running = True
    scale = 1e-9  # Adjust for visualization
    center = (WIDTH // 2, HEIGHT // 2)
    t = 0  # initial time

    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        for _ in range(sim_speed):
            body_list.perform_step(dt_sim)
            t += dt_sim

        # Draw
        screen.fill(BLACK)
        for i in range(len(body_list)):
            pos = body_list.r_0[3 * i : 3 * i + 3]
            x, y = scale_pos(pos, center, scale)
            color = COLORS[i % len(COLORS)]
            pygame.draw.circle(screen, color, (x, y), 1)
            # Optionally, draw name
            if hasattr(body_list, "names"):
                font = pygame.font.SysFont(None, 24)
                text = font.render(body_list.names[i], True, WHITE)
                screen.blit(text, (x + 12, y - 12))

        pygame.display.flip()
        # clock.tick(60)  # Limit to 60 FPS

    pygame.quit()

    body_list.dump(file_out)

    sys.exit()
