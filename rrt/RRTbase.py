import random
import math
import pygame

class RRTMap:
    def __init__(self, start, goal, map_dimension, obstacles_dimensions, obsicles_number):
        self.start = start
        self.goal = goal
        self.map_dimension = map_dimension
        self.mapw, self.maph = map_dimension

        self.MapWindowName = "RRT Path Planning"
        pygame.display.set_caption(self.MapWindowName)
        self.map = pygame.display.set_mode((self.mapw, self.maph))
        self.map.fill((255, 255, 255))

        self.nodeRad = 0
        self.nodeThickness = 0
        self.edgeThickness = 1

        self.obsticles = []
        self.obsticles_dimensions = obstacles_dimensions
        self.obsicles_number = obsicles_number

        #Colors
        self.Blue = (0, 0, 255)
        self.Red = (255, 0, 0)
        self.Green = (0, 255, 0)
        self.Black = (0, 0, 0)
        self.Gray = (80, 80, 80)
        self.White = (255, 255, 255)

    def drawMap(self, obsticles):
        pygame.draw.circle(self.map, self.Green, self.start, self.nodeRad + 5, 0)
        pygame.draw.circle(self.map, self.Red, self.goal, self.nodeRad + 20, 1)
        self.drawObstacles(obsticles)

    def drawPath(self):
        pass

    def drawObstacles(self, obstacles):
        
        obslist = obstacles.copy()
        while len(obslist) > 0:
            obs = obslist.pop()
            pygame.draw.rect(self.map, self.Gray, obs)


class RRTGraph:
    def __init__(self, start, goal, map_dimension, obstacles_dimensions, obsicles_number):
        (x, y) = start
        self.start = start
        self.goal = goal
        self.goalFlag = False

        self.map_dimension = map_dimension
        self.mapw, self.maph = map_dimension
        
        #tree
        self.x = []
        self.y = []
        self.parent = []

        self.x.append(x)
        self.y.append(y)
        self.parent.append(0)

        self.obsticles = []
        self.obsticles_dimensions = obstacles_dimensions
        self.obsicles_number = obsicles_number

        #path
        self.goalState = None
        self.path = []

    def addNode(self, node):
        pass

    def addEdge(self, node1, node2):
        pass

    def makeRandomRect(self):
        x = int(random.uniform(0, self.mapw - self.obsticles_dimensions))
        y = int(random.uniform(0, self.maph - self.obsticles_dimensions))

        return (x, y)

    def makeObstacles(self):
        obs = []
        for i in range(self.obsicles_number):
            rectang = None
            startGoalColl = True
            while startGoalColl:
                coords = self.makeRandomRect()
                rectang = pygame.Rect(coords, (self.obsticles_dimensions, self.obsticles_dimensions))
                if rectang.collidepoint(self.start) or rectang.collidepoint(self.goal):
                    startGoalColl = True
                else:
                    startGoalColl = False
            obs.append(rectang)
        self.obsticles = obs.copy()
        return obs

    def removeNode(self, node):
        pass

    def removeEdge(self, node1, node2):
        pass

    def numberOfNodes(self):
        pass

    def distance(self, node1, node2):
        pass

    def nearest(self, node):
        pass

    def isFree(self):
        pass

    def crossObstacle(self):
        pass

    def connect(self):
        pass

    def step(self):
        pass

    def pathToGoal(self):
        pass

    def getPathCoords(self):
        pass

    def bias(self):
        pass

    def expand(self):
        pass

    def cost(self):
        pass