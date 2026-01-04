import pygame
import math

class Spot:
    def __init__(self, x, y, isWall = False):
        self.position = pygame.Vector2(x, y)
        self.isWall = isWall

        self.size = 12

        self.f = 0 # over all cost g + h
        self.g = 0 #cost to this point
        self.h = 0 #heuristics

        self.previous = None
        self.neighbors = []
        #self.costs = []

    def clear(self):
        self.f = 0
        self.g = 0
        self.h = 0
        self.previous = None

    def addNeighbor(self):
        pass

    def show(self, map, color, showG = False):
        
        pygame.draw.circle(map, color, (self.position.x, self.position.y), self.size, 0)


class AStarGraph:
    def __init__(self, spots = []):
        self.spots = spots
    
    def addSpot(self, spot : Spot):
        self.spots.append(spot)
    
    def getSpot(self, x, y):
        for spot in self.spots:
            if spot.position.x == x and spot.position.y == y:
                return spot

    # TODO: add remove spot method if posisble later
    def removeSpot(self, spot):
        pass

    def addEdge(self, spotA: Spot, spotB : Spot, bidir = True):
        spotA.addNeighbor(spotB)
        if bidir:
            spotB.addNeighbor(spotA)
    
    def getEdges(self, spot : Spot):
        return spot.neighbors
    
    # TODO: maybe if needed
    def removeEdge(self):
        pass

    def clearSpots(self):
        for spot in self.spots:
            spot.clear()

    def clear(self):
        self.spots = []
    
    def getNearestSpot(self, x, y):
        nearestSpot = None
        minDist = 100000

        for spot in self.spots:
            dist = math.sqrt((spot.position.x - x) ** 2 + (spot.position.y - y) ** 2)
            if dist < minDist:
                minDist = dist
                nearestSpot = spot
        return nearestSpot
    
    def getNearestSpotIn(self, x, y, maxDist):
        nearestSpot = None
        minDist = 100000

        for spot in self.spots:
            dist = math.sqrt((spot.position.x - x) ** 2 + (spot.position.y - y) ** 2)
            if dist < minDist and dist <= maxDist:
                minDist = dist
                nearestSpot = spot
        return nearestSpot
    
    def getNearestSpotWithin(self, x, y, minDistance, maxDist):
        nearestSpot = None
        minDist = 100000

        for spot in self.spots:
            dist = math.sqrt((spot.position.x - x) ** 2 + (spot.position.y - y) ** 2)
            if dist < minDist and dist <= maxDist and dist >= minDistance:
                minDist = dist
                nearestSpot = spot
        return nearestSpot
    
    # TODO: add drawing all spots and edges arrocdingly
    def draw(self, screen):
        pass

class AStar:
    def __init__ (self, start: Spot, end: Spot):

        #Colors
        self.Blue = (0, 187, 255)
        self.Red = (255, 0, 0)
        self.Green = (0, 255, 0)
        self.Black = (0, 0, 0)
        self.Gray = (80, 80, 80)
        self.White = (255, 255, 255)

        self._setup(start, end)

    def restart(self, start: Spot, end: Spot):
        self._setup(start, end)

    def _setup(self, start: Spot, end: Spot):
        self.start = start
        self.end = end

        self.start.isWall = False
        self.end.isWall = False

        #Flags
        self.isDone = False
        self.noSolution = False

        self.openSet = []
        self.openSet.append(self.start)
        self.closeSet = []

        self.path = []


    def _distance(self, a: Spot, b:Spot):
        side1 = b.position.x - a.position.x
        side2 = b.position.y - a.position.y

        return math.sqrt(side1*side1 + side2*side2)

    def _heuristic(self, a : Spot, b : Spot, shortest = True):
        hip = self._distance(a, b)
        manhantanDis = math.abs(a.position.x - b.position.x) + math.abs(a.position.y - b.position.y)

        return hip if shortest else manhantanDis

    # TODO: update the algorithm accordingly
    def update(self):
        pass


    def draw(self,screen, showG = False):
        for node in self.path:
            node.show(screen, self.Blue, showG)

    def debugDraw(self,screen, showG = False):
        for node in self.openSet:
            node.show(screen, self.Green, showG)
        
        for node in self.closeSet:
            node.show(screen, self.Red, showG)
        
        self.draw(screen, showG)