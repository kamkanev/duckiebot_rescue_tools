import pygame
import math

class Spot:
    def __init__(self, x, y, isWall = False):
        self.x = x
        self.y = y
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
        
        pygame.draw.circle(map, color, (self.x, self.y), self.size, 0)