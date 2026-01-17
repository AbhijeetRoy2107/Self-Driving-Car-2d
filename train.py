import os
import pickle
import pygame as py
import neat

from car import Car
from road import Road
from world import World
from NNdraw import NN
from config_variables import *


_bg_cache = None
_fonts_ready = False
GEN = 0

#check pygame health
def ensure_pygame_ready():
    global _fonts_ready
    if not _fonts_ready:
        py.font.init()
        _fonts_ready = True

#draw background
def make_bg(w, h):
    surf = py.Surface((w, h))
    top = (245, 245, 245)
    bottom = (210, 210, 210)
    for y in range(h):
        t = y / max(1, h - 1)
        col = (
            int(top[0] + (bottom[0] - top[0]) * t),
            int(top[1] + (bottom[1] - top[1]) * t),
            int(top[2] + (bottom[2] - top[2]) * t),
        )
        py.draw.line(surf, col, (0, y), (w, y))
    return surf


def get_bg():
    global _bg_cache
    if _bg_cache is None:
        _bg_cache = make_bg(WIN_WIDTH, WIN_HEIGHT)
    return _bg_cache

#draw heads-up UI elements
def draw_hud(world, gen, alive):
    ensure_pygame_ready()

    margin = 12
    panel_w, panel_h = 240, 86
    x = world.win_width - panel_w - margin
    y = margin

    panel = py.Surface((panel_w, panel_h), py.SRCALPHA)
    panel.fill((255, 255, 255, 170))
    py.draw.rect(panel, (40, 40, 40, 60), panel.get_rect(), 1, border_radius=10)
    world.win.blit(panel, (x, y))

    lines = [
        ("Generation", str(gen)),
        ("Alive", str(alive)),
        ("Best fitness", f"{world.getScore():.2f}"),
    ]

    font_label = py.font.SysFont("consolas", 18)
    font_val = py.font.SysFont("consolas", 18, bold=True)

    yy = y + 8
    for k, v in lines:
        label = font_label.render(f"{k}:", True, (25, 25, 25))
        val = font_val.render(v, True, (25, 25, 25))
        world.win.blit(label, (x + 12, yy))
        world.win.blit(val, (x + 140, yy))
        yy += 24


def draw_win(cars, road, world, gen):
    world.win.blit(get_bg(), (0, 0))
    road.draw(world)

    for car in cars:
        car.draw(world)
        if HIGHLIGHT_BEST_CAR and world.bestCar is car:
            p = world.getScreenCoords(car.x, car.y)
            py.draw.circle(world.win, (40, 140, 220), p, 34, 3)

    if SHOW_BEST_RAYS and world.bestCar is not None:
        world.bestCar.drawSensors(world, road, width=2)

    if world.bestNN is not None:
        world.bestNN.draw(world)

    draw_hud(world, gen, len(cars))
    py.display.update()


#save best genome continuously during training even on abrupt exit
class BestGenomeSaver(neat.reporting.BaseReporter):
    def __init__(self, path):
        self.path = path
        self.best_fitness = float("-inf")

    def post_evaluate(self, config, population, species_set, best_genome):
        if best_genome is None or best_genome.fitness is None:
            return
        if best_genome.fitness > self.best_fitness:
            self.best_fitness = best_genome.fitness
            with open(self.path, "wb") as f:
                pickle.dump(best_genome, f)

#generating report for each generation during training(saves as report.txt)
class FileGenerationReporter(neat.reporting.BaseReporter):
    def __init__(self, filename):
        self.filename = filename
        self.generation = 0
        with open(self.filename, "w", encoding="utf-8") as f:
            f.write("NEAT Training Report\n")
            f.write("====================\n\n")
            f.write("Gen | BestFitness | MeanFitness | StdFitness | Species\n")

    def start_generation(self, generation):
        self.generation = generation

    def post_evaluate(self, config, population, species_set, best_genome):
        fits = [g.fitness for g in population.values() if g.fitness is not None]
        if fits:
            mean = sum(fits) / len(fits)
            var = sum((x - mean) ** 2 for x in fits) / len(fits)
            std = var ** 0.5
        else:
            mean = float("nan")
            std = float("nan")

        best = best_genome.fitness if best_genome is not None else float("nan")
        try:
            species_count = len(species_set.species)
        except Exception:
            species_count = "?"

        line = (
            f"{self.generation:4d} | "
            f"{best:10.4f} | "
            f"{mean:10.4f} | "
            f"{std:10.4f} | "
            f"{species_count}\n"
        )
        with open(self.filename, "a", encoding="utf-8") as f:
            f.write(line)


#NEAT training and evaluation loop
def eval_genomes(genomes, config):
    global GEN
    GEN += 1
    ensure_pygame_ready()

    nets, ge, cars, nns = [], [], [], []
    t = 0

    world = World(STARTING_POS, WIN_WIDTH, WIN_HEIGHT)
    road = Road(world)
    clock = py.time.Clock()

    for _, g in genomes:
        nets.append(neat.nn.FeedForwardNetwork.create(g, config))
        cars.append(Car(0, 0, 0))
        g.fitness = 0.0
        ge.append(g)
        nns.append(NN(config, g, (90, 210)))

    run_loop = True
    while run_loop:
        t += 1
        clock.tick(FPS)

        for event in py.event.get():
            if event.type == py.QUIT:
                py.quit()
                return

        (xb, yb) = (0, 0)
        i = 0
        while i < len(cars):
            car = cars[i]

            inp = car.getInputs(world, road)
            inp.append(car.vel / MAX_VEL)
            car.commands = nets[i].activate(tuple(inp))

            y_old = car.y
            (x, y) = car.move(road, t)

            forward_progress = -(y - y_old)
            ge[i].fitness += forward_progress / 100.0
            ge[i].fitness += car.vel * SCORE_VEL_MULTIPLIER

            if t > 10 and (
                car.detectCollision(road)
                or y > world.getBestCarPos()[1] + BAD_GENOME_TRESHOLD
                or y > y_old
                or car.vel < 0.1
            ):
                ge[i].fitness -= 1
                cars.pop(i)
                nets.pop(i)
                ge.pop(i)
                nns.pop(i)
                continue
            else:
                if ge[i].fitness > world.getScore():
                    world.updateScore(ge[i].fitness)
                    world.bestNN = nns[i]
                    world.bestInputs = inp
                    world.bestCommands = car.commands
                    world.bestCar = car
                    world.bestGenome = ge[i]
                i += 1

            if y < yb:
                (xb, yb) = (x, y)

        if len(cars) == 0:
            break

        world.updateBestCarPos((xb, yb))
        road.update(world)
        draw_win(cars, road, world, GEN)


def run_training(config_path, winner_path="winner_genome.pkl", generations=10000):
    global GEN
    GEN = 0

    config = neat.config.Config(
        neat.DefaultGenome,
        neat.DefaultReproduction,
        neat.DefaultSpeciesSet,
        neat.DefaultStagnation,
        config_path
    )

    p = neat.Population(config)
    p.add_reporter(neat.StdOutReporter(True))
    p.add_reporter(neat.StatisticsReporter())
    p.add_reporter(FileGenerationReporter(REPORT_FILE))
    p.add_reporter(BestGenomeSaver(winner_path))

    winner = p.run(eval_genomes, generations)

    if winner is not None:
        with open(winner_path, "wb") as f:
            pickle.dump(winner, f)

    return winner
