import os
import pickle
import pygame as py
import neat

from car import Car
from road import Road
from world import World
from config_variables import *

import train  # reuse draw_win()

WINNER_FILE = "bestModel\winner_genome.pkl"


def load_config(config_path):
    return neat.config.Config(
        neat.DefaultGenome,
        neat.DefaultReproduction,
        neat.DefaultSpeciesSet,
        neat.DefaultStagnation,
        config_path
    )


def _draw_badge_top_left(win):
    font = py.font.SysFont("consolas", 22, bold=True)
    text = "BEST MODEL SO FAR"
    surf = font.render(text, True, (15, 15, 15))

    pad_x, pad_y = 12, 8
    box = py.Surface((surf.get_width() + 2 * pad_x, surf.get_height() + 2 * pad_y), py.SRCALPHA)
    box.fill((255, 255, 255, 190))
    py.draw.rect(box, (40, 40, 40, 80), box.get_rect(), 1, border_radius=10)

    win.blit(box, (12, 12))
    win.blit(surf, (12 + pad_x, 12 + pad_y))


def _draw_controls_from_outputs(win, outputs):
    """
    Visualize the model outputs directly (no color changes).
    Improved spacing to prevent label/value overlap.
    """
    font_title = py.font.SysFont("consolas", 18, bold=True)
    font_label = py.font.SysFont("consolas", 16)
    font_val = py.font.SysFont("consolas", 14)

    x0, y0 = 12, 70
    w, h = 230, 158  # slightly bigger panel to fit padding better

    panel = py.Surface((w, h), py.SRCALPHA)
    panel.fill((255, 255, 255, 170))
    py.draw.rect(panel, (40, 40, 40, 90), panel.get_rect(), 1, border_radius=10)
    win.blit(panel, (x0, y0))

    title = font_title.render("MODEL OUTPUTS", True, (20, 20, 20))
    win.blit(title, (x0 + 14, y0 + 10))

    border_col = (40, 40, 40, 120)
    text_col = (20, 20, 20)

    labels = ["LEFT", "RIGHT", "ACCEL", "BRAKE"]

    outs = list(outputs) if isinstance(outputs, (list, tuple)) else []
    while len(outs) < 4:
        outs.append(0.0)

    def draw_tile(label, value, x, y):
        # Bigger tile + internal padding so text never collides
        bw, bh = 100, 44
        pad_l = 10
        pad_r = 10

        box = py.Surface((bw, bh), py.SRCALPHA)
        box.fill((255, 255, 255, 210))
        py.draw.rect(box, border_col, box.get_rect(), 1, border_radius=8)
        win.blit(box, (x, y))

        # Label (top-left inside tile)
        label_surf = font_label.render(label, True, text_col)
        win.blit(label_surf, (x + pad_l, y + 7))

        # Value (bottom-right inside tile)
        val_surf = font_val.render(f"{float(value):+.2f}", True, text_col)
        win.blit(val_surf, (x + bw - pad_r - val_surf.get_width(), y + bh - 8 - val_surf.get_height()))

    bx = x0 + 14
    by = y0 + 46
    gap_x = 10
    gap_y = 10

    draw_tile(labels[0], outs[0], bx, by)
    draw_tile(labels[1], outs[1], bx + 100 + gap_x, by)
    draw_tile(labels[2], outs[2], bx, by + 44 + gap_y)
    draw_tile(labels[3], outs[3], bx + 100 + gap_x, by + 44 + gap_y)

    if isinstance(outputs, (list, tuple)) and len(outputs) > 4:
        extra = font_val.render(f"({len(outputs)} outputs)", True, (60, 60, 60))
        win.blit(extra, (x0 + 14, y0 + h - 24))


def run_demo():
    py.font.init()

    local_dir = os.path.dirname(__file__)
    config_path = os.path.join(local_dir, "config_file.txt")
    winner_path = os.path.join(local_dir, WINNER_FILE)

    if not os.path.exists(winner_path):
        raise FileNotFoundError(f"Missing {WINNER_FILE}. Run: python train.py first.")

    config = load_config(config_path)
    with open(winner_path, "rb") as f:
        genome = pickle.load(f)

    net = neat.nn.FeedForwardNetwork.create(genome, config)

    world = World(STARTING_POS, WIN_WIDTH, WIN_HEIGHT)
    road = Road(world)
    clock = py.time.Clock()

    car = Car(0, 0, 0)
    cars = [car]

    world.bestCar = car
    world.updateBestCarPos((car.x, car.y))

    t = 0
    run_loop = True
    while run_loop:
        t += 1
        clock.tick(FPS)

        for event in py.event.get():
            if event.type == py.QUIT:
                run_loop = False

        inp = car.getInputs(world, road)
        inp.append(car.vel / MAX_VEL)

        out = net.activate(tuple(inp))
        car.commands = out
        car.move(road, t)

        world.bestCar = car
        bx, by = world.getBestCarPos()
        if car.y < by:
            world.updateBestCarPos((car.x, car.y))
        else:
            world.updateBestCarPos((bx, by))

        road.update(world)

        # Prevent double-update flicker (main.draw_win calls display.update internally)
        _orig_update = py.display.update
        py.display.update = lambda *args, **kwargs: None
        train.draw_win(cars, road, world, gen=0)
        py.display.update = _orig_update

        _draw_badge_top_left(world.win)
        _draw_controls_from_outputs(world.win, out)

        py.display.flip()

    py.quit()


if __name__ == "__main__":
    run_demo()
