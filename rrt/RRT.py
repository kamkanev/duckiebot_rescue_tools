import pygame
from RRTbase import RRTGraph
from RRTbase import RRTMap
import time

def main():
    dimensions = (1000, 600)
    start = (50, 50)
    goal = (550, 350)
    obstacles_dimensions = 50
    obstacles_number = 30

    iteration = 0
    t1 = 0

    pygame.init()
    map = RRTMap(start, goal, dimensions, obstacles_dimensions, obstacles_number)
    graph = RRTGraph(start, goal, dimensions, obstacles_dimensions, obstacles_number)

    obstacles = graph.makeObstacles()
    map.drawMap(obstacles)

    t1 = time.time()
    while(not graph.pathToGoal()):
        elapsed = time.time() - t1
        t1 = time.time()
        if elapsed > 10:
            raise Exception("Timeout: Could not find path in 10 seconds")
        if iteration % 10 == 0:
            x, y, parent = graph.bias(goal)
            pygame.draw.circle(map.map, map.Gray, (x[-1], y[-1]), map.nodeRad + 2, 0)
            pygame.draw.line(map.map, map.Blue, (x[-1], y[-1]), (x[parent[-1]], y[parent[-1]]), map.edgeThickness)
        else:
            x, y, parent = graph.expand()
            pygame.draw.circle(map.map, map.Gray, (x[-1], y[-1]), map.nodeRad + 2, 0)
            pygame.draw.line(map.map, map.Blue, (x[-1], y[-1]), (x[parent[-1]], y[parent[-1]]), map.edgeThickness)

        
        if iteration % 5 == 0:
            pygame.display.update()
        iteration += 1

        # for event in pygame.event.get():
        #     if event.type == pygame.QUIT:
        #             pygame.quit()
        #             return
        #     if event.type == pygame.KEYDOWN:
        #         if event.key == pygame.K_ESCAPE:
        #             pygame.quit()
        #             return
    map.drawPath(graph.getPathCoords())
    pygame.display.update()
    pygame.event.clear()
    pygame.event.wait(0)

if __name__ == "__main__":
    result = False
    while not result:
        try:
            main()
            result = True
        except:
            result = False