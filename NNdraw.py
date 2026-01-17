import pygame as py
from config_variables import *
from node import Node, Connection

py.font.init()


def _clamp(x, a, b):
    return a if x < a else b if x > b else x


class NN:
    def __init__(self, config, genome, pos):
        self.nodes = []
        self.genome = genome
        self.pos = (int(pos[0] + NODE_RADIUS), int(pos[1]))

        input_names = ["Sensor T", "Sensor TR", "Sensor R", "Sensor BR",
                       "Sensor B", "Sensor BL", "Sensor L", "Sensor TL", "Speed"]
        output_names = ["Accelerate", "Brake", "Turn Left", "Turn Right"]

        middle_nodes = [n for n in genome.nodes.keys()]
        nodeIdList = []

        # UI/layout styling
        self.panel_pad = 16
        self.title_h = 34
        self.col_gap = 130

        # ----- Inputs -----
        h = (INPUT_NEURONS - 1) * (NODE_RADIUS * 2 + NODE_SPACING)
        for i, inp in enumerate(config.genome_config.input_keys):
            n = Node(
                inp,
                self.pos[0],
                self.pos[1] + int(-h / 2 + i * (NODE_RADIUS * 2 + NODE_SPACING)),
                INPUT,
                [GREEN_PALE, GREEN, DARK_GREEN_PALE, DARK_GREEN],
                input_names[i],
                i
            )
            self.nodes.append(n)
            nodeIdList.append(inp)

        # ----- Outputs -----
        h = (OUTPUT_NEURONS - 1) * (NODE_RADIUS * 2 + NODE_SPACING)
        for i, out in enumerate(config.genome_config.output_keys):
            n = Node(
                out + INPUT_NEURONS,
                self.pos[0] + 2 * (self.col_gap + 2 * NODE_RADIUS),
                self.pos[1] + int(-h / 2 + i * (NODE_RADIUS * 2 + NODE_SPACING)),
                OUTPUT,
                [RED_PALE, RED, DARK_RED_PALE, DARK_RED],
                output_names[i],
                i
            )
            self.nodes.append(n)
            if out in middle_nodes:
                middle_nodes.remove(out)
            nodeIdList.append(out)

        # ----- Hidden/Middle -----
        h = (len(middle_nodes) - 1) * (NODE_RADIUS * 2 + NODE_SPACING) if len(middle_nodes) > 1 else 0
        for i, m in enumerate(middle_nodes):
            n = Node(
                m,
                self.pos[0] + (self.col_gap + 2 * NODE_RADIUS),
                self.pos[1] + int(-h / 2 + i * (NODE_RADIUS * 2 + NODE_SPACING)),
                MIDDLE,
                [BLUE_PALE, DARK_BLUE, BLUE_PALE, DARK_BLUE]
            )
            self.nodes.append(n)
            nodeIdList.append(m)

        # ----- Connections -----
        self.connections = []
        for c in genome.connections.values():
            if c.enabled:
                inp, out = c.key
                wt = float(c.weight)

                # Your Connection signature is (input, output, wt) and fields are input/output/wt
                conn = Connection(
                    self.nodes[nodeIdList.index(inp)],
                    self.nodes[nodeIdList.index(out)],
                    wt
                )
                self.connections.append(conn)

        self.title_font = py.font.SysFont("consolas", 18, bold=True)

    def _panel_rect(self):
        xs = [n.x for n in self.nodes]
        ys = [n.y for n in self.nodes]
        minx, maxx = min(xs), max(xs)
        miny, maxy = min(ys), max(ys)

        left = minx - (NODE_RADIUS + self.panel_pad + 78)
        right = maxx + (NODE_RADIUS + self.panel_pad + 90)
        top = miny - (NODE_RADIUS + self.panel_pad + self.title_h)
        bottom = maxy + (NODE_RADIUS + self.panel_pad + 10)

        return py.Rect(left, top, right - left, bottom - top)

    def _draw_panel(self, world):
        rect = self._panel_rect()

        # Shadow
        shadow = py.Surface((rect.w + 8, rect.h + 8), py.SRCALPHA)
        py.draw.rect(shadow, (0, 0, 0, 60), shadow.get_rect(), border_radius=18)
        world.win.blit(shadow, (rect.x + 4, rect.y + 4))

        # Panel
        panel = py.Surface((rect.w, rect.h), py.SRCALPHA)
        py.draw.rect(panel, (255, 255, 255, 210), panel.get_rect(), border_radius=18)
        py.draw.rect(panel, (30, 30, 30, 70), panel.get_rect(), width=1, border_radius=18)

        # Header strip
        header_h = self.title_h + 6
        header = py.Rect(0, 0, rect.w, header_h)
        py.draw.rect(panel, (245, 245, 245, 235), header, border_radius=18)
        py.draw.line(panel, (0, 0, 0, 35), (0, header_h), (rect.w, header_h), 1)

        title = self.title_font.render("Neural Network (Best Genome)", True, (20, 20, 20))
        panel.blit(title, (16, 10))

        world.win.blit(panel, (rect.x, rect.y))

    def _draw_pretty_connection(self, world, conn):
        # Your Connection stores weight in conn.wt
        w = float(getattr(conn, "wt", 0.0))
        absw = abs(w)

        # Thickness/alpha based on magnitude
        thickness = int(_clamp(1 + absw * 1.2, 1, 5))
        alpha = int(_clamp(60 + absw * 55, 60, 210))

        # Green for positive, red for negative
        if w >= 0:
            color = (40, 170, 90, alpha)
        else:
            color = (210, 70, 70, alpha)

        # Your Connection endpoints are conn.input and conn.output
        n1 = conn.input
        n2 = conn.output

        x1, y1 = n1.x, n1.y
        x2, y2 = n2.x, n2.y

        # Gentle curve
        mx, my = (x1 + x2) / 2, (y1 + y2) / 2
        offset = _clamp((y2 - y1) * 0.08, -35, 35)
        cx, cy = mx, my - offset

        # Draw on alpha layer
        layer = py.Surface((world.win_width, world.win_height), py.SRCALPHA)

        steps = 18
        prev = (x1, y1)
        for i in range(1, steps + 1):
            t = i / steps
            bx = (1 - t) * (1 - t) * x1 + 2 * (1 - t) * t * cx + t * t * x2
            by = (1 - t) * (1 - t) * y1 + 2 * (1 - t) * t * cy + t * t * y2
            py.draw.line(layer, color, prev, (bx, by), thickness)
            prev = (bx, by)

        world.win.blit(layer, (0, 0))

    def draw(self, world):
        self._draw_panel(world)

        # Connections first
        for c in self.connections:
            self._draw_pretty_connection(world, c)

        # Nodes on top
        for node in self.nodes:
            node.draw_node(world)
