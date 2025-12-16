import pygame
from RRTbase import RRTGraph
from RRTbase import RRTMap

def main():
    dimensions = (1000, 600)
    start = (50, 50)
    goal = (550, 350)
    obstacles_dimensions = 50
    obstacles_number = 30

    pygame.init()
    map = RRTMap(start, goal, dimensions, obstacles_dimensions, obstacles_number)
    graph = RRTGraph(start, goal, dimensions, obstacles_dimensions, obstacles_number)

    obstacles = graph.makeObstacles()

    map.drawMap(obstacles)

    pygame.display.update()
    pygame.event.clear()
    pygame.event.wait(0)

if __name__ == "__main__":
    main()