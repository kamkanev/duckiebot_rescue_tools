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

        self.nodeRad = 5
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

    def drawPath(self, path):
        for node in path:
            pygame.draw.circle(self.map, self.Red, node, self.nodeRad + 3, 0)

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

    def addNode(self, n, x, y):
        self.x.insert(n, x)
        self.y.append(y)
        

    def addEdge(self, parent, child):
        self.parent.insert(child, parent)

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

    def removeNode(self, n):
        self.x.pop(n)
        self.y.pop(n)

    def removeEdge(self, n):
        self.parent.pop(n)

    def numberOfNodes(self):
        return len(self.x)

    def distance(self, node1, node2):
        (x1, y1) = (self.x[node1], self.y[node1])
        (x2, y2) = (self.x[node2], self.y[node2])
        return math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
    
    def sample_env(self):
        x = int(random.uniform(0, self.mapw))
        y = int(random.uniform(0, self.maph))
        return (x, y)

    def nearest(self, node):
        dmin = self.distance(0, node)
        nnear = 0
        for i in range(0, node):
            if self.distance(i, node) < dmin:
                dmin = self.distance(i, node)
                nnear = i
        return nnear
            

    def isFree(self):
        n = self.numberOfNodes() - 1
        (x,y) = (self.x[n], self.y[n])
        obs = self.obsticles.copy()
        while len(obs) > 0:
            rect = obs.pop()
            if rect.collidepoint(x, y):
                self.removeNode(n)
                return False
        return True
        

    def crossObstacle(self, x1, y1, x2, y2):
        obs = self.obsticles.copy()
        while len(obs) > 0:
            rect = obs.pop()
            for i in range(0,101):
                u = i / 100
                x = x1 * u + x2 * (1 - u)
                y = y1 * u + y2 * (1 - u)
                if rect.collidepoint(x, y):
                    return True
        return False
        

    def connect(self, n1, n2):
        (x1, y1) = (self.x[n1], self.y[n1])
        (x2, y2) = (self.x[n2], self.y[n2])
        if self.crossObstacle(x1, y1, x2, y2):
            self.removeNode(n2)
            return False
        else:
            self.addEdge(n1, n2)
            return True

    def step(self, nnear, nrand, dmax = 35):
        d = self.distance(nnear, nrand)
        if d > dmax:
            if d == 0:
                self.removeNode(nrand)
                return
            u = dmax / d
            (xnear, ynear) = (self.x[nnear], self.y[nnear])
            (xrand, yrand) = (self.x[nrand], self.y[nrand])
            (px, py) = (xrand - xnear, yrand - ynear)
            theta = math.atan2(py, px)
            (x, y) = (int(xnear + dmax * math.cos(theta)), int(ynear + dmax * math.sin(theta)))
            self.removeNode(nrand)
            if abs(x-self.goal[0]) < dmax and abs(y-self.goal[1]) < dmax:
                self.addNode(nrand, self.goal[0], self.goal[1])
                self.goalState = nrand
                self.goalFlag = True
            else:
                self.addNode(nrand, x, y)
        

    def pathToGoal(self):
        if self.goalFlag:
            self.path = []
            self.path.append(self.goalState)
            newp = self.parent[self.goalState]
            while (newp != 0):
                self.path.append(newp)
                newp = self.parent[newp]
            self.path.append(0)
        return self.goalFlag

    def getPathCoords(self):
        pathCoords = []
        for node in self.path:
            x,y = (self.x[node], self.y[node])
            pathCoords.append((x,y))
        return pathCoords
    
    def waypoints2path(self):
        oldPath = self.getPathCoords()
        path = []

        for i in range(0, len(oldPath)-1):
            print(i)
            if i >= len(oldPath):
                break
            (x1, y1) = oldPath[i]
            (x2, y2) = oldPath[i+1]
            print("--------------------")
            print((x1, y1), (x2, y2))
            for j in range(0, 5):
                u = j / 5
                x = x1 * u + x2 * (1 - u)
                y = y1 * u + y2 * (1 - u)
                path.append((x, y))
                print((x, y))
        
        return path

    def bias(self, ngoal):
        n = self.numberOfNodes()
        self.addNode(n, self.goal[0], self.goal[1])
        nnear = self.nearest(n)
        self.step(nnear, n)
        self.connect(nnear, n)
        return self.x, self.y, self.parent

    def expand(self):
        n=self.numberOfNodes()
        x,y = self.sample_env()
        self.addNode(n, x, y)
        if self.isFree():
            xnearest = self.nearest(n)
            self.step(xnearest, n)
            self.connect(xnearest, n)
        return self.x, self.y, self.parent

    def cost(self):
        pass