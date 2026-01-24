import os
import sys

# Ensure project root is on sys.path
repo_root = os.path.normpath(os.path.join(os.path.dirname(__file__), '..'))
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

from utils.entities.drive import Enviroment, Robot
from utils.graph.RRTbase import RRTMap, RRTGraph
import pygame
import math
import time

def useRRT():

    iteration = 0
    t1 = 0
    graph = RRTGraph((robot.x, robot.y), goal, (dims[1], dims[0]), 0, 0)

    t1 = time.time()
    while(not graph.pathToGoal()):
        elapsed = time.time() - t1
        t1 = time.time()
        if elapsed > 10:
            raise Exception("Timeout: Could not find path in 10 seconds")
        if iteration % 10 == 0:
            x, y, parent = graph.bias(goal)
        else:
            x, y, parent = graph.expand()

        
        iteration += 1

    return graph.getPathCoords()
#TODO: use waypoints method

#init
pygame.init()

start = (200, 200)

goal = None

dims = (800, 1200)


env = Enviroment(dims)
robot = Robot(start, 'utils/entities/duck_bot.png', 0.2 * 3779.52) #0.01 meters in pixels - 1 cm width

FPS = 120
dt = 0
lasttime = pygame.time.get_ticks()

clock = pygame.time.Clock()

run = True

while run:
    clock.tick(FPS)
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            run = False
        if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
            goal = pygame.mouse.get_pos()
            path = None
            result = False
            max_retries = 5
            for _ in range(max_retries):
                try:
                    path = useRRT()
                    result = True
                    break
                except:
                    pass
            
            if result:
                robot.path = path
                robot.waypoint = 0
        # robot.move(event)

    robot.dt = (pygame.time.get_ticks() - lasttime) / 1000
    lasttime = pygame.time.get_ticks()
    env.map.fill(env.Black)

    pos = pygame.mouse.get_pos()
    pygame.draw.circle(env.map, env.Yellow, pos, 5)

    

    if goal is not None:
        pygame.draw.circle(env.map, env.Red, goal, 10)

    # Draw a simple representation of the robot
    # pygame.draw.circle(env.map, env.Blue, start, 20)
    if robot.path:
        for waypoint in robot.path:
            pygame.draw.circle(env.map, env.Green, waypoint, 5)
        robot.move_without_event()
    else:
        robot.move()
    robot.draw(env.map)
    env.show_info(robot.vl, robot.vr, robot.theta)
    env.robot_frame((robot.x, robot.y), robot.theta)


    pygame.display.update()

pygame.quit()