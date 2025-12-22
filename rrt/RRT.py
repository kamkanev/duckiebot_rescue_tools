import pygame
from RRTbase import RRTGraph
from RRTbase import RRTMap

def main():
    dimensions = (1000, 600)
    start = (50, 50)
    goal = (550, 350)
    obstacles_dimensions = 50
    obstacles_number = 30

    iteration = 0

    pygame.init()
    map = RRTMap(start, goal, dimensions, obstacles_dimensions, obstacles_number)
    graph = RRTGraph(start, goal, dimensions, obstacles_dimensions, obstacles_number)

    obstacles = graph.makeObstacles()

    map.drawMap(obstacles)

    while(True):
        x,y = graph.sample_env()
        n = graph.numberOfNodes()
        graph.addNode(n, x, y)
        graph.addEdge(n-1, n)
        x1,y1 = graph.x[n], graph.y[n]
        x2,y2 = graph.x[n-1], graph.y[n-1]
        if graph.isFree():
            pygame.draw.circle(map.map, map.Red, (graph.x[n], graph.y[n]), map.nodeRad, map.nodeThickness)

            if not graph.crossObstacle(x1, y1, x2, y2):
                pygame.draw.line(map.map, map.Blue, (x1, y1), (x2, y2), map.edgeThickness)
        
        
        pygame.display.update()

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                    pygame.quit()
                    return
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    pygame.quit()
                    return
    
    # pygame.event.clear()
    # pygame.event.wait(0)

if __name__ == "__main__":
    main()