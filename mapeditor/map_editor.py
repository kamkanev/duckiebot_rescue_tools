import pygame
import button

pygame.init()

#define window size and fps
FPS = 120

clock = pygame.time.Clock()

SCREEN_WIDTH = 1200
SCREEN_HEIGHT = 800
LOWER_MARGIN = 200
SIDE_MARGIN = 600

# define grid
ROWS = 10
COLS = 15
TILE_SIZE = SCREEN_HEIGHT // ROWS

#define colors
WHITE = (255, 255, 255)
GRAY = (200, 200, 200)
BLACK = (0, 0, 0)
RED = (255, 0, 0)
BLUE = (0, 0, 255)

TYLE_TYPES = 5
current_tile = 0

#create world map
world_map = []
for row in range(ROWS):
    r = [-1] * COLS
    world_map.append(r)


screen = pygame.display.set_mode((SCREEN_WIDTH + SIDE_MARGIN, SCREEN_HEIGHT + LOWER_MARGIN))
pygame.display.set_caption("Map Editor")

#load images
# load and store tile images
img_list = []
for x in range(TYLE_TYPES):
    img = pygame.image.load(f'img/tiles/{x}_tile.png').convert_alpha()
    img = pygame.transform.scale(img, (TILE_SIZE, TILE_SIZE))
    img_list.append(img)

def draw_grid():
    for row in range(ROWS + 1):
        pygame.draw.line(screen, WHITE, (0, row * TILE_SIZE), (SCREEN_WIDTH, row * TILE_SIZE))
    for col in range(COLS + 1):
        pygame.draw.line(screen, WHITE, (col * TILE_SIZE, 0), (col * TILE_SIZE, SCREEN_HEIGHT))

def draw_world():
    for y, row in enumerate(world_map):
        for x, tile in enumerate(row):
            if tile >= 0:
                screen.blit(img_list[tile], (x * TILE_SIZE, y * TILE_SIZE))

#create buttons
button_list = []
button_col = 0
button_row = 0
for i in range(len(img_list)):
    tile_button = button.Button(SCREEN_WIDTH + (150 * button_col) + 120, (120 * button_row) + 60, img_list[i], 1)
    button_list.append(tile_button)
    button_col += 1
    if button_col == 3:
        button_col = 0
        button_row += 1

run = True
while run:

    clock.tick(FPS)

    draw_grid()
    draw_world()

    pygame.draw.rect(screen, GRAY, (SCREEN_WIDTH, 0, SIDE_MARGIN, SCREEN_HEIGHT))
    pygame.draw.rect(screen, GRAY, (0, SCREEN_HEIGHT, SCREEN_WIDTH + SIDE_MARGIN, LOWER_MARGIN))

    #choose a tile
    button_count = 0
    for button_count, b in enumerate(button_list):
        if b.draw(screen):
            current_tile = button_count
    
    #show the selected tile
    pygame.draw.rect(screen, BLUE, button_list[current_tile].rect, 5)
        

    for event in pygame.event.get():
        if event.type == pygame.QUIT or (event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE):
            run = False

    # screen.fill((255, 255, 255))

    pygame.display.update()

pygame.quit()