import pygame
import button
import os
import json
import sys
import csv
import shutil

# Ensure project root is on sys.path so `from utils.graph.AStar import Spot` works
repo_root = os.path.normpath(os.path.join(os.path.dirname(__file__), '..'))
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

try:
    from utils.graph.AStar import Spot, AStarGraph, AStar
except Exception:
    # fallback: load by file path if package import still fails
    try:
        import importlib.util
        astar_path = os.path.join(repo_root, 'utils', 'graph', 'AStar.py')
        spec = importlib.util.spec_from_file_location('a_star', astar_path)
        a_star = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(a_star)
        Spot = a_star.Spot
        AStarGraph = a_star.AStarGraph
        AStar = a_star.AStar
    except Exception:
        Spot = None
        AStarGraph = None
        AStar = None

pygame.init()

#define window size and fps
FPS = 120

clock = pygame.time.Clock()

SCREEN_WIDTH = 1200
SCREEN_HEIGHT = 800
LOWER_MARGIN = 200
SIDE_MARGIN = 600

name = ""
placeholder = "untitled_map"

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
GREEN = (0, 255, 0)

font = pygame.font.SysFont('Futura', 28)

TYLE_TYPES = 5
current_tile = 0

#test graph and spots
s1 = Spot(SCREEN_WIDTH + SIDE_MARGIN // 2, SCREEN_HEIGHT - TILE_SIZE // 2)
s2 = Spot(SCREEN_WIDTH + SIDE_MARGIN // 2, SCREEN_HEIGHT + TILE_SIZE // 2)

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


def save_map(file_name):
    #save the map as a csv file in a folder named the same as the map file and in the saves folder
    if len(file_name) <= 0:
        file_name = placeholder
    save_dir = os.path.join(os.path.dirname(__file__), 'saves', file_name)
    try:
        os.makedirs(save_dir, exist_ok=True)
        tmp_path = os.path.join(save_dir, f'{file_name}.csv.tmp')
        final_path = os.path.join(save_dir, f'{file_name}.csv')
        with open(tmp_path, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile, delimiter=',')
            for row in world_map:
                writer.writerow(row)
        os.replace(tmp_path, final_path)
        print(f'Map saved as {file_name}.csv')
        return True
    except Exception as e:
        print(f'Error saving map {file_name}: {e}')
        try:
            if 'tmp_path' in locals() and os.path.exists(tmp_path):
                os.remove(tmp_path)
        except Exception:
            pass
        return False

def load_map(file_name):
    #load the map from a csv file in a folder named the same as the map file and in the saves folder
    global world_map
    if len(file_name) <= 0:
        file_name = placeholder
    load_dir = os.path.join(os.path.dirname(__file__), 'saves', file_name, f'{file_name}.csv')
    try:
        with open(load_dir, 'r') as csvfile:
            reader = csv.reader(csvfile, delimiter=',')
            loaded_map = []
            for row in reader:
                loaded_map.append([int(tile) for tile in row])

        if not loaded_map:
            print(f'Loaded map {file_name}.csv is empty')
            return False

        # normalize to current ROWS x COLS
        new_map = [[-1] * COLS for _ in range(ROWS)]
        for y, row in enumerate(loaded_map):
            if y >= ROWS:
                break
            for x, val in enumerate(row):
                if x >= COLS:
                    break
                try:
                    new_map[y][x] = int(val)
                except Exception:
                    new_map[y][x] = -1

        world_map = new_map
        print(f'Map {file_name}.csv loaded successfully')
        return True
    except FileNotFoundError:
        print(f'No save file found with the name {file_name}.csv')
        return False
    except Exception as e:
        print(f'Error loading map {file_name}: {e}')
        return False

#rotate image
def blitRotateCenter(surf, image, topleft, angle):

    rotated_image = pygame.transform.rotate(image, angle)
    new_rect = rotated_image.get_rect(center = image.get_rect(topleft = topleft).center)

    surf.blit(rotated_image, new_rect)

#rotate point
def rotatePoint(point, pivot, angle):
    vec = point - pivot
    rotate_vec = vec.rotate(angle)

    new_point = pivot + rotate_vec

    return new_point

#draw text function
def draw_text(text, font, text_col, x, y):
    img = font.render(text, True, text_col)
    screen.blit(img, (x, y))


def _draw_overlay_rect(rect, color=(50, 50, 50, 200)):
    s = pygame.Surface((rect[2], rect[3]), pygame.SRCALPHA)
    s.fill((0, 0, 0, 160))
    screen.blit(s, (rect[0], rect[1]))


def pygame_input(prompt, initial=""):
    """Simple blocking Pygame text input dialog. Returns string or None."""
    input_text = initial
    box_w = 600
    box_h = 160
    box_x = (SCREEN_WIDTH + SIDE_MARGIN // 2) - box_w // 2
    box_y = (SCREEN_HEIGHT + LOWER_MARGIN // 2) - box_h // 2
    active = True
    while active:
        clock.tick(FPS)
        for ev in pygame.event.get():
            if ev.type == pygame.QUIT:
                return None
            if ev.type == pygame.KEYDOWN:
                if ev.key == pygame.K_RETURN:
                    return input_text
                if ev.key == pygame.K_ESCAPE:
                    return None
                if ev.key == pygame.K_BACKSPACE:
                    input_text = input_text[:-1]
                else:
                    if ev.unicode:
                        input_text += ev.unicode
            if ev.type == pygame.MOUSEBUTTONDOWN and ev.button == 1:
                mx, my = ev.pos
                # detect simple Save / Cancel buttons
                save_rect = (box_x + box_w - 200, box_y + box_h - 48, 80, 36)
                cancel_rect = (box_x + box_w - 100, box_y + box_h - 48, 80, 36)
                if save_rect[0] <= mx <= save_rect[0] + save_rect[2] and save_rect[1] <= my <= save_rect[1] + save_rect[3]:
                    return input_text
                if cancel_rect[0] <= mx <= cancel_rect[0] + cancel_rect[2] and cancel_rect[1] <= my <= cancel_rect[1] + cancel_rect[3]:
                    return None

        # draw current screen dimmed
        draw_bg()
        draw_grid()
        draw_world()
        pygame.draw.rect(screen, GRAY, (SCREEN_WIDTH, 0, SIDE_MARGIN, SCREEN_HEIGHT))
        pygame.draw.rect(screen, GRAY, (0, SCREEN_HEIGHT, SCREEN_WIDTH + SIDE_MARGIN, LOWER_MARGIN))
        # overlay box
        _draw_overlay_rect((box_x, box_y, box_w, box_h))
        pygame.draw.rect(screen, WHITE, (box_x, box_y, box_w, box_h), 2)
        draw_text(prompt, font, WHITE, box_x + 16, box_y + 16)
        # input text
        txt_surf = font.render(input_text, True, WHITE)
        screen.blit(txt_surf, (box_x + 16, box_y + 56))
        # buttons
        pygame.draw.rect(screen, (30, 120, 30), (box_x + box_w - 200, box_y + box_h - 48, 80, 36))
        draw_text('Save', font, WHITE, box_x + box_w - 188, box_y + box_h - 44)
        pygame.draw.rect(screen, (120, 30, 30), (box_x + box_w - 100, box_y + box_h - 48, 80, 36))
        draw_text('Cancel', font, WHITE, box_x + box_w - 88, box_y + box_h - 44)

        pygame.display.update()


def pygame_choice_dialog(title, choices, deletable=True):
    """Simple blocking Pygame choice dialog. Returns chosen item or None.

    If deletable is True, a red 'X' is shown next to each choice; clicking it asks
    for confirmation and deletes the save directory, removing it from the list.
    """
    box_w = 600
    box_h = min(400, 40 + 32 * len(choices) + 64)
    box_x = (SCREEN_WIDTH + SIDE_MARGIN // 2) - box_w // 2
    box_y = (SCREEN_HEIGHT + LOWER_MARGIN // 2) - box_h // 2
    selected = 0
    while True:
        clock.tick(FPS)
        for ev in pygame.event.get():
            if ev.type == pygame.QUIT:
                return None
            if ev.type == pygame.KEYDOWN:
                if ev.key == pygame.K_RETURN:
                    return choices[selected]
                if ev.key == pygame.K_ESCAPE:
                    return None
                if ev.key == pygame.K_UP:
                    selected = max(0, selected - 1)
                if ev.key == pygame.K_DOWN:
                    selected = min(len(choices) - 1, selected + 1)
            if ev.type == pygame.MOUSEBUTTONDOWN and ev.button == 1:
                mx, my = ev.pos
                # check list items and delete X area
                for i, _ in enumerate(choices):
                    item_y = box_y + 40 + i * 32
                    text_hit = (box_x + 16 <= mx <= box_x + box_w - 56 and item_y <= my <= item_y + 28)
                    x_hit = (box_x + box_w - 40 <= mx <= box_x + box_w - 16 and item_y <= my <= item_y + 28)
                    if x_hit and deletable:
                        # ask confirm delete
                        confirm = pygame_choice_dialog(f"Delete '{choices[i]}'?", ["Yes", "No"], deletable=False)
                        if confirm == "Yes":
                            # delete directory
                            try:
                                shutil.rmtree(os.path.join(os.path.dirname(__file__), 'saves', choices[i]))
                            except Exception:
                                pass
                            # remove from choices and continue
                            choices.pop(i)
                            if selected >= len(choices):
                                selected = max(0, len(choices) - 1)
                            # recompute box height
                            box_h = min(400, 40 + 32 * len(choices) + 64)
                            break
                        else:
                            break
                    if text_hit:
                        return choices[i]
        # draw
        draw_bg()
        draw_grid()
        draw_world()
        pygame.draw.rect(screen, GRAY, (SCREEN_WIDTH, 0, SIDE_MARGIN, SCREEN_HEIGHT))
        pygame.draw.rect(screen, GRAY, (0, SCREEN_HEIGHT, SCREEN_WIDTH + SIDE_MARGIN, LOWER_MARGIN))
        _draw_overlay_rect((box_x, box_y, box_w, box_h))
        pygame.draw.rect(screen, WHITE, (box_x, box_y, box_w, box_h), 2)
        draw_text(title, font, WHITE, box_x + 16, box_y + 12)
        for i, c in enumerate(choices):
            col = BLUE if i == selected else WHITE
            draw_text(c, font, col, box_x + 16, box_y + 40 + i * 32)
            if deletable:
                # draw red X box
                x_box = (box_x + box_w - 40, box_y + 40 + i * 32, 24, 24)
                pygame.draw.rect(screen, (200, 50, 50), x_box)
                draw_text('X', font, WHITE, x_box[0] + 6, x_box[1] - 2)

        pygame.display.update()


def show_message(text, seconds=1.2):
    end = pygame.time.get_ticks() + int(seconds * 1000)
    while pygame.time.get_ticks() < end:
        clock.tick(FPS)
        for ev in pygame.event.get():
            if ev.type == pygame.QUIT:
                return
        # draw dim overlay with message
        draw_bg()
        draw_grid()
        draw_world()
        _draw_overlay_rect((SCREEN_WIDTH // 4, SCREEN_HEIGHT // 3, SCREEN_WIDTH // 2 + SIDE_MARGIN // 2, 80))
        pygame.draw.rect(screen, WHITE, (SCREEN_WIDTH // 4, SCREEN_HEIGHT // 3, SCREEN_WIDTH // 2 + SIDE_MARGIN // 2, 80), 2)
        draw_text(text, font, WHITE, SCREEN_WIDTH // 4 + 16, SCREEN_HEIGHT // 3 + 20)
        pygame.display.update()



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
                #screen.blit(img_list[tile%TYLE_TYPES], (x * TILE_SIZE, y * TILE_SIZE))
                blitRotateCenter(screen, img_list[tile%TYLE_TYPES], (x * TILE_SIZE, y * TILE_SIZE), (tile // TYLE_TYPES) * 90)

def draw_current_tile():
    pygame.draw.rect(screen, BLACK, (SCREEN_WIDTH + SIDE_MARGIN // 2 - TILE_SIZE - 5, SCREEN_HEIGHT - TILE_SIZE - 5, TILE_SIZE * 2 + 10, TILE_SIZE * 2 + 10))
    
    if current_tile < 0:
        pygame.draw.rect(screen, GRAY, (SCREEN_WIDTH + SIDE_MARGIN // 2 - TILE_SIZE, SCREEN_HEIGHT - TILE_SIZE, TILE_SIZE * 2, TILE_SIZE * 2))
    else:
        scaled = pygame.transform.scale(img_list[current_tile%TYLE_TYPES], (TILE_SIZE * 2, TILE_SIZE * 2))
        #screen.blit(scaled, (SCREEN_WIDTH + SIDE_MARGIN // 2 - TILE_SIZE, SCREEN_HEIGHT - TILE_SIZE))
        blitRotateCenter(screen, scaled, (SCREEN_WIDTH + SIDE_MARGIN // 2 - TILE_SIZE, SCREEN_HEIGHT - TILE_SIZE), (current_tile // TYLE_TYPES) * 90)
        pygame.draw.circle(screen, GREEN, (SCREEN_WIDTH + SIDE_MARGIN // 2, SCREEN_HEIGHT), 12, 0)
        pygame.draw.circle(screen, RED, rotatePoint(s1.position, (SCREEN_WIDTH + SIDE_MARGIN // 2, SCREEN_HEIGHT), (current_tile // TYLE_TYPES) * 90), s1.size, 0)
        pygame.draw.circle(screen, BLUE, rotatePoint(s2.position, (SCREEN_WIDTH + SIDE_MARGIN // 2, SCREEN_HEIGHT), (current_tile // TYLE_TYPES) * 90), s2.size, 0)

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
    draw_text(f'File Name: {placeholder if len(name) <= 0 else name}', font, BLACK, 10, SCREEN_HEIGHT + 10)
    draw_text('Tile Selection:', font, BLACK, SCREEN_WIDTH + 20, 20)
    draw_text('Current Tile:', font, BLACK, SCREEN_WIDTH + 20, SCREEN_HEIGHT - TILE_SIZE - 20)

    draw_text('Press Q or E to rotate tile!', font, BLACK, SCREEN_WIDTH + SIDE_MARGIN // 4 + 20, SCREEN_HEIGHT + TILE_SIZE + 20)
    draw_text('Left Click: Place Tile | Right Click: Remove Tile', font, BLACK, SCREEN_WIDTH + TILE_SIZE, SCREEN_HEIGHT + TILE_SIZE + 50)
    draw_text('Add the JS graph and nodes in the tiles and made mode for connecting them - TODO', font, BLACK, SCREEN_WIDTH // 3, SCREEN_HEIGHT + LOWER_MARGIN - 160)
    
    

    #save and load map 
    if save_button.draw(screen):
        # If a name is already set, ask only for confirmation to overwrite
        if len(name) > 0:
            pick = pygame_choice_dialog(f"Save changes to '{name}'?", ["Yes", "No"], deletable=False) 
            if pick == "Yes":
                ok = save_map(name)
                if ok:
                    show_message(f"Saved map '{name}'")
                else:
                    show_message(f"Failed to save '{name}'")
        else:
            save_name = placeholder
            result = pygame_input("Enter file name (Enter to save, Esc to cancel):", initial=save_name)
            if result is not None:
                save_name = result.strip() if result.strip() != "" else placeholder
                ok = save_map(save_name)
                if ok:
                    show_message(f"Saved map '{save_name}'")
                else:
                    show_message(f"Failed to save '{save_name}'")
                name = save_name

    if load_button.draw(screen):
        saves_root = os.path.join(os.path.dirname(__file__), 'saves')
        choices = []
        if os.path.isdir(saves_root):
            for entry in sorted(os.listdir(saves_root)):
                path = os.path.join(saves_root, entry)
                if os.path.isdir(path):
                    choices.append(entry)
        if not choices:
            show_message('No saved maps found')
        else:
            picked = pygame_choice_dialog('Choose save to load', choices)
            if picked is not None:
                ok = load_map(picked)
                if ok:
                    show_message(f"Loaded '{picked}'")
                else:
                    show_message(f"Failed to load '{picked}'")
                name = picked


    #choose a tile
    button_count = 0
    for button_count, b in enumerate(button_list):
        if b.draw(screen):
            current_tile = button_count
    
    #show the selected tile
    pygame.draw.rect(screen, BLUE, button_list[current_tile%TYLE_TYPES].rect, 5)
    draw_current_tile()

    #draw test spots
    # s1.show(screen, RED)
    # s2.show(screen, BLUE)

    #add tile to the map
    pos = pygame.mouse.get_pos()
    x = pos[0] // TILE_SIZE
    y = pos[1] // TILE_SIZE

    draw_text(f'Mouse pos: ({pos[0]},{pos[1]})', font, BLACK, 10, SCREEN_HEIGHT + 40)
    draw_text(f'Mouse grid: ({x},{y})', font, BLACK, 10, SCREEN_HEIGHT + 65)

    if pos[0] < SCREEN_WIDTH and pos[1] < SCREEN_HEIGHT:
        if pygame.mouse.get_pressed()[0]:
            if world_map[y][x] != current_tile:
                world_map[y][x] = current_tile
        if pygame.mouse.get_pressed()[2]:
            world_map[y][x] = -1
        

    for event in pygame.event.get():
        if event.type == pygame.QUIT or (event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE):
            run = False
        if event.type == pygame.KEYUP:
            #rotate tile left
            if event.key == pygame.K_q:
                if current_tile // TYLE_TYPES < 3:
                    current_tile += TYLE_TYPES
                else:
                    current_tile = current_tile % TYLE_TYPES
            #rotate tile right
            if event.key == pygame.K_e:
                if current_tile // TYLE_TYPES > 0:
                    current_tile -= TYLE_TYPES
                else:
                    current_tile = current_tile + TYLE_TYPES * 3
                pass

    # screen.fill((255, 255, 255))

    pygame.display.update()

pygame.quit()