import pygame as py
import neat
import time
import os
import random
from car import Car
from road import Road
from world import World
from NNdraw import NN
from config_variables import *
py.font.init()



def make_bg(w, h):
    surf = py.Surface((w, h))
    top = (245, 245, 245)
    bottom = (210, 210, 210)
    for y in range(h):
        t = y / max(1, h-1)
        col = (
            int(top[0] + (bottom[0]-top[0])*t),
            int(top[1] + (bottom[1]-top[1])*t),
            int(top[2] + (bottom[2]-top[2])*t),
        )
        py.draw.line(surf, col, (0, y), (w, y))
    return surf

bg = make_bg(WIN_WIDTH, WIN_HEIGHT)


def draw_hud(world, GEN, alive):
    margin = 12

    panel_w, panel_h = 240, 86
    x = world.win_width - panel_w - margin
    y = margin

    panel = py.Surface((panel_w, panel_h), py.SRCALPHA)
    panel.fill((255, 255, 255, 170))
    py.draw.rect(panel, (40, 40, 40, 60), panel.get_rect(), 1, border_radius=10)

    world.win.blit(panel, (x, y))

    lines = [
        ("Generation", str(GEN)),
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


def draw_win(cars, road, world, GEN):
    # Background first
    world.win.blit(bg, (0,0))

    road.draw(world)

    # Draw all cars, with optional best highlight
    for car in cars:
        car.draw(world)
        if HIGHLIGHT_BEST_CAR and world.bestCar is car:
            # subtle ring around best car
            p = world.getScreenCoords(car.x, car.y)
            py.draw.circle(world.win, (40, 140, 220), p, 34, 3)

    # Show rays for best car so far
    if SHOW_BEST_RAYS and world.bestCar is not None:
        world.bestCar.drawSensors(world, road, width=2)

    # NN panel (existing)
    if world.bestNN is not None:
        world.bestNN.draw(world)

    # HUD
    draw_hud(world, GEN, len(cars))

    py.display.update()


def main(genomes = [], config = []):
    global GEN
    GEN += 1

    nets = []
    ge = []
    cars = []
    t = 0

    world = World(STARTING_POS, WIN_WIDTH, WIN_HEIGHT)

    NNs = []

    # Track best of THIS generation (for reporting)
    gen_best_fitness = float("-inf")

    for _,g in genomes:
        net = neat.nn.FeedForwardNetwork.create(g, config)
        nets.append(net)
        cars.append(Car(0, 0, 0))
        g.fitness = 0
        ge.append(g)
        NNs.append(NN(config, g, (90, 210)))

    road = Road(world)
    clock = py.time.Clock()

    run = True
    while run:
        t += 1
        clock.tick(FPS)

        for event in py.event.get():
            if event.type == py.QUIT:
                run = False
                py.quit()
                quit()

        (xb, yb) = (0,0)
        i = 0
        while(i < len(cars)):
            car = cars[i]

            inp = car.getInputs(world, road)
            inp.append(car.vel/MAX_VEL)
            car.commands = nets[i].activate(tuple(inp))

            y_old = car.y
            (x, y) = car.move(road, t)

            # Fitness shaping: reward forward progress (decreasing y), small reward for staying alive
            forward_progress = -(y - y_old)  # positive if moving "up"
            ge[i].fitness += forward_progress / 100.0
            ge[i].fitness += car.vel * SCORE_VEL_MULTIPLIER

            if ge[i].fitness > gen_best_fitness:
                gen_best_fitness = ge[i].fitness

            # kill conditions (your existing logic)
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
                NNs.pop(i)
            else:
                # update best-of-so-far across the run (for rays + NN draw)
                if ge[i].fitness > world.getScore():
                    world.updateScore(ge[i].fitness)
                    world.bestNN = NNs[i]
                    world.bestInputs = inp
                    world.bestCommands = car.commands
                    world.bestCar = car
                    world.bestGenome = ge[i]
                i += 1

            if y < yb:
                (xb, yb) = (x, y)

        if len(cars) == 0:
            run = False
            break

        world.updateBestCarPos((xb, yb))
        road.update(world)
        draw_win(cars, road, world, GEN)

    # Return a dict-like object for the reporter to consume (neat-python ignores return value
    # but our custom reporter reads statistics from the StatisticsReporter).
    # We keep this function signature unchanged for neat-python.
    return


class FileGenerationReporter(neat.reporting.BaseReporter):
    """
    Writes a per-generation summary to REPORT_FILE.
    """
    def __init__(self, filename):
        self.filename = filename
        self.generation = 0

        # Start fresh each run
        with open(self.filename, "w", encoding="utf-8") as f:
            f.write("NEAT Training Report\n")
            f.write("====================\n\n")
            f.write("Gen | BestFitness | MeanFitness | StdFitness | Species\n")

    def start_generation(self, generation):
        # neat-python calls this with the current generation index
        self.generation = generation

    def post_evaluate(self, config, population, species_set, best_genome):
        # population is a dict: {genome_id: genome}
        fits = [g.fitness for g in population.values() if g.fitness is not None]
        if fits:
            mean = sum(fits) / len(fits)
            var = sum((x - mean) ** 2 for x in fits) / len(fits)
            std = var ** 0.5
        else:
            mean = float("nan")
            std = float("nan")

        best = best_genome.fitness if best_genome is not None else float("nan")

        # species_set is a SpeciesSet; species_set.species is usually a dict
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


# NEAT function
def run(config_path):
    config = neat.config.Config(
        neat.DefaultGenome,
        neat.DefaultReproduction,
        neat.DefaultSpeciesSet,
        neat.DefaultStagnation,
        config_path
    )

    p = neat.Population(config)

    # Console reporter
    p.add_reporter(neat.StdOutReporter(True))

    # Stats reporter (still useful for later graphing; safe to keep)
    stats = neat.StatisticsReporter()
    p.add_reporter(stats)

    # File reporter (robust across neat-python versions)
    p.add_reporter(FileGenerationReporter(REPORT_FILE))

    # This MUST continue across generations; neat-python handles it.
    # main() should NOT call quit() except on user window close.
    winner = p.run(main, 10000)
    return winner


if __name__ == "__main__":
    local_dir = os.path.dirname(__file__)
    config_path = os.path.join(local_dir, "config_file.txt")
    run(config_path)
