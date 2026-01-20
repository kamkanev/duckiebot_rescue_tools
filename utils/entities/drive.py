import pygame
import math

class Enviroment:
    def __init__(self, dimentions):

        #colors
        self.Black = (0, 0, 0)
        self.White = (255, 255, 255)
        self.Green = (0, 255, 0)
        self.Red = (255, 0, 0)
        self.Blue = (0, 0, 255)
        self.Yellow = (255, 255, 0)

        #map dimentions
        self.height = dimentions[0]
        self.width = dimentions[1]

        pygame.display.set_caption("Differential drive robot simulation")
        self.map = pygame.display.set_mode((self.width, self.height))

        self.text = "text"
        self.font = pygame.font.SysFont('Arial', 18)
        self.text = self.font.render(self.text, True, self.White)
        self.textRect = self.text.get_rect()
        self.textRect.center = (dimentions[1] // 2, dimentions[0] - 80)
    
    def show_info(self, vl, vr, theta):
        text = f"Vl:= {vl/37.7952:.2f} m/s | Vr:= {vr/37.7952:.2f} m/s | Theta:= {math.degrees(theta):.2f} deg"
        self.text = self.font.render(text, True, self.White)
        self.map.blit(self.text, self.textRect)

    def robot_frame(self, pos, rot):
        n = 80

        centerx, centery = pos
        x_axis = (centerx + n * math.cos(-rot), centery + n * math.sin(-rot))
        y_axis = (centerx + n * math.cos(-rot+math.pi/2), centery + n * math.sin(-rot+math.pi/2))

        pygame.draw.line(self.map, self.Red, (centerx, centery), x_axis, 3)
        pygame.draw.line(self.map, self.Green, (centerx, centery), y_axis, 3)


class Robot:
    def __init__(self, start_pos, robotimg, width):
        self.m2p = 3779.52  # meters to pixels conversion
        self.w = width
        self.x = start_pos[0]
        self.y = start_pos[1]
        self.theta = 0
        self.speed = 0.01
        self.vl = self.speed * self.m2p
        self.vr = self.speed * self.m2p

        self.MAXSPEED = 0.03 * self.m2p
        self.MINSPEED = -0.03 * self.m2p

        self.img = pygame.image.load(robotimg)
        self.rotated = self.img
        self.rect = self.rotated.get_rect(center=(self.x, self.y))
    
    def move(self, event=None):

        if event is not None:
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_q:
                    self.vl += self.speed * self.m2p
                elif event.key == pygame.K_a:
                    self.vl -= self.speed * self.m2p
                elif event.key == pygame.K_e:
                    self.vr += self.speed * self.m2p
                elif event.key == pygame.K_d:
                    self.vr -= self.speed * self.m2p
                elif event.key == pygame.K_UP:
                    self.vl += self.speed * self.m2p
                    self.vr += self.speed * self.m2p
                elif event.key == pygame.K_DOWN:
                    self.vl -= self.speed * self.m2p
                    self.vr -= self.speed * self.m2p
                elif event.key == pygame.K_LEFT:
                    self.vl -= self.speed * self.m2p
                    self.vr += self.speed * self.m2p
                elif event.key == pygame.K_RIGHT:
                    self.vl += self.speed * self.m2p
                    self.vr -= self.speed * self.m2p
            elif event.type == pygame.KEYUP:
                if event.key in [pygame.K_q, pygame.K_a, pygame.K_e, pygame.K_d]:
                    self.vl = 0
                    self.vr = 0
        
        self.x += ((self.vl + self.vr) / 2) * math.cos(self.theta) * dt
        self.y -= ((self.vl + self.vr) / 2) * math.sin(self.theta) * dt
        self.theta += (self.vr - self.vl) / self.w * dt
        if self.theta > 2 * math.pi or self.theta < -2 * math.pi:
            self.theta = 0
        
        self.vr = min(self.vr, self.MAXSPEED)
        self.vl = min(self.vl, self.MAXSPEED)

        self.vr = max(self.vr, self.MINSPEED)
        self.vl = max(self.vl, self.MINSPEED)


        self.rotated = pygame.transform.rotate(self.img, math.degrees(self.theta))
        self.rect = self.rotated.get_rect(center=(self.x, self.y))

    def draw(self, map):
        map.blit(self.rotated, self.rect)
        

#init
pygame.init()

start = (200, 200)

dims = (800, 1200)

env = Enviroment(dims)
robot = Robot(start, 'utils/entities/duck_bot.png', 0.01 * 3779.52) #0.01 meters in pixels - 1 cm width

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
        robot.move(event)

    dt = (pygame.time.get_ticks() - lasttime) / 1000
    lasttime = pygame.time.get_ticks()
    env.map.fill(env.Black)

    # Draw a simple representation of the robot
    # pygame.draw.circle(env.map, env.Blue, start, 20)
    robot.move()
    robot.draw(env.map)
    env.show_info(robot.vl, robot.vr, robot.theta)
    env.robot_frame((robot.x, robot.y), robot.theta)


    pygame.display.update()

pygame.quit()