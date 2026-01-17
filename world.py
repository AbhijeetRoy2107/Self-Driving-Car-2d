import pygame as py
from config_variables import CAMERA_TARGET_Y, CAMERA_SMOOTHING

class World:
    initialPos = (0,0)
    bestCarPos = (0,0)

    def __init__(self, starting_pos, world_width, world_height):
        self.initialPos = starting_pos
        self.bestCarPos = (0, 0)
        self.win  = py.display.set_mode((world_width, world_height))
        self.win_width = world_width
        self.win_height = world_height
        self.score = 0

        # camera smoothing state
        self.cam_x = 0.0
        self.cam_y = 0.0

        # best-of-so-far
        self.bestGenome = None
        self.bestCar = None
        self.bestNN = None
        self.bestInputs = None
        self.bestCommands = None

    def updateBestCarPos(self, pos):
        # Smooth follow
        tx, ty = pos
        self.cam_x = (1 - CAMERA_SMOOTHING) * self.cam_x + CAMERA_SMOOTHING * tx
        self.cam_y = (1 - CAMERA_SMOOTHING) * self.cam_y + CAMERA_SMOOTHING * ty
        self.bestCarPos = (self.cam_x, self.cam_y)

    def getScreenCoords(self, x, y):
        # Keep car lower on screen so we see more road ahead.
        target_screen_y = self.win_height * CAMERA_TARGET_Y

        return (int(x + self.initialPos[0] - self.bestCarPos[0]),
                int(y + target_screen_y - self.bestCarPos[1]))

    def getBestCarPos(self):
        return self.bestCarPos

    def updateScore(self, new_score):
        self.score = new_score

    def getScore(self):
        return self.score
