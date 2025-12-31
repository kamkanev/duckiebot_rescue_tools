import pygame
import button
import os

pygame.init()

#define window size and fps
FPS = 120

clock = pygame.time.Clock()

SCREEN_WIDTH = 1200
SCREEN_HEIGHT = 800
LOWER_MARGIN = 200
SIDE_MARGIN = 600

name = "untitled_map"

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

font = pygame.font.SysFont('Futura', 28)

TYLE_TYPES = 5
current_tile = 0

#create world map
world_map = []
for row in range(ROWS):
    r = [-1] * COLS
    world_map.append(r)


screen = pygame.display.set_mode((SCREEN_WIDTH + SIDE_MARGIN, SCREEN_HEIGHT + LOWER_MARGIN))
pygame.display.set_caption("Map Editor")

#load images (script-relative)
image_path = os.path.join(os.path.dirname(__file__), 'img')
grass_img = pygame.image.load(os.path.join(image_path, 'grass.jpg')).convert_alpha()
grass_img = pygame.transform.scale(grass_img, (SCREEN_WIDTH, SCREEN_HEIGHT))

button_path = os.path.join(os.path.dirname(__file__), 'img', 'buttons')
save_img = pygame.image.load(os.path.join(button_path, 'save.png')).convert_alpha()
save_img = pygame.transform.scale(save_img, (150, 50))
load_img = pygame.image.load(os.path.join(button_path, 'load.png')).convert_alpha()
load_img = pygame.transform.scale(load_img, (150, 50))
save_hover_img = pygame.image.load(os.path.join(button_path, 'save_hover.png')).convert_alpha()
save_hover_img = pygame.transform.scale(save_hover_img, (150, 50))
load_hover_img = pygame.image.load(os.path.join(button_path, 'load_hover.png')).convert_alpha()
load_hover_img = pygame.transform.scale(load_hover_img, (150, 50))

# load and store tile images
img_list = []
tiles_dir = os.path.join(os.path.dirname(__file__), 'img', 'tiles')
for x in range(TYLE_TYPES):
    img_path = os.path.join(tiles_dir, f'{x}_tile.png')
    img = pygame.image.load(img_path).convert_alpha()
    img = pygame.transform.scale(img, (TILE_SIZE, TILE_SIZE))
    img_list.append(img)


#draw text function
def draw_text(text, font, text_col, x, y):
    img = font.render(text, True, text_col)
    screen.blit(img, (x, y))



#draw functions

def draw_bg():
    screen.fill(BLACK)
    screen.blit(grass_img, (0, 0))

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

def draw_current_tile():
    pygame.draw.rect(screen, BLACK, (SCREEN_WIDTH + SIDE_MARGIN // 2 - TILE_SIZE - 5, SCREEN_HEIGHT - TILE_SIZE - 5, TILE_SIZE * 2 + 10, TILE_SIZE * 2 + 10))
    
    if current_tile < 0:
        pygame.draw.rect(screen, GRAY, (SCREEN_WIDTH + SIDE_MARGIN // 2 - TILE_SIZE, SCREEN_HEIGHT - TILE_SIZE, TILE_SIZE * 2, TILE_SIZE * 2))
    else:
        scaled = pygame.transform.scale(img_list[current_tile], (TILE_SIZE * 2, TILE_SIZE * 2))
        screen.blit(scaled, (SCREEN_WIDTH + SIDE_MARGIN // 2 - TILE_SIZE, SCREEN_HEIGHT - TILE_SIZE))

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

#create save and load btns
save_button = button.Button(SCREEN_WIDTH // 2, SCREEN_HEIGHT  + LOWER_MARGIN - 100, save_img, scale=1, hover_image=save_hover_img)
load_button = button.Button(SCREEN_WIDTH // 2 + 200, SCREEN_HEIGHT + LOWER_MARGIN - 100, load_img, scale=1, hover_image=load_hover_img)


#main loop
run = True
while run:

    clock.tick(FPS)

    draw_bg()
    draw_grid()
    draw_world()

    #side and lower margins
    pygame.draw.rect(screen, GRAY, (SCREEN_WIDTH, 0, SIDE_MARGIN, SCREEN_HEIGHT))
    pygame.draw.rect(screen, GRAY, (0, SCREEN_HEIGHT, SCREEN_WIDTH + SIDE_MARGIN, LOWER_MARGIN))

    #draw text
    draw_text('Tile Selection:', font, BLACK, SCREEN_WIDTH + 20, 20)
    

    #save and load map
    save_button.draw(screen)
    load_button.draw(screen)

    #choose a tile
    button_count = 0
    for button_count, b in enumerate(button_list):
        if b.draw(screen):
            current_tile = button_count
    
    #show the selected tile
    pygame.draw.rect(screen, BLUE, button_list[current_tile].rect, 5)
    draw_current_tile()

    #add tile to the map
    pos = pygame.mouse.get_pos()
    x = pos[0] // TILE_SIZE
    y = pos[1] // TILE_SIZE

    draw_text(f'Mouse pos: ({pos[0]},{pos[1]})', font, BLACK, 10, SCREEN_HEIGHT + 10)
    draw_text(f'Mouse grid: ({x},{y})', font, BLACK, 10, SCREEN_HEIGHT + 40)

    if pos[0] < SCREEN_WIDTH and pos[1] < SCREEN_HEIGHT:
        if pygame.mouse.get_pressed()[0]:
            if world_map[y][x] != current_tile:
                world_map[y][x] = current_tile
        if pygame.mouse.get_pressed()[2]:
            world_map[y][x] = -1
        

    for event in pygame.event.get():
        if event.type == pygame.QUIT or (event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE):
            run = False

    # screen.fill((255, 255, 255))

    pygame.display.update()

pygame.quit()