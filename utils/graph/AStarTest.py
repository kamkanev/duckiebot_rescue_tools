import pygame
import json
import os
import sys
import math

# Ensure project root is on sys.path
repo_root = os.path.normpath(os.path.join(os.path.dirname(__file__), '../..'))
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

from utils.graph.AStar import Spot, AStarGraph, AStar
from llm.route_instructions import parse_text_to_steps, write_turns_bin

# Initialize pygame
pygame.init()

# Constants
SCREEN_WIDTH = 1400
SCREEN_HEIGHT = 900
FPS = 60
FONT_SIZE = 14
TITLE_FONT_SIZE = 20

# Colors
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
GRAY = (128, 128, 128)
LIGHT_GRAY = (200, 200, 200)
DARK_GRAY = (50, 50, 50)
BLUE = (0, 187, 255)
RED = (255, 0, 0)
GREEN = (0, 255, 0)
PURPLE = (66, 96, 228)
YELLOW = (255, 255, 0)
ORANGE = (255, 165, 0)
CHAT_BG = (240, 240, 240)
CHAT_BORDER = (180, 180, 180)

class GraphVisualizer:
    def __init__(self, initial_graph=None):
        self.screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
        pygame.display.set_caption("A* Pathfinding Visualizer")
        self.clock = pygame.time.Clock()
        self.font = pygame.font.SysFont('Arial', FONT_SIZE)
        self.title_font = pygame.font.SysFont('Arial', TITLE_FONT_SIZE, bold=True)
        
        # State
        self.graph = None
        self.astar = None
        self.start_spot = None
        self.end_spot = None
        self.available_graphs = self._get_available_graphs()
        self.selected_graph_idx = 0
        self.is_running = True
        self.algorithm_running = False
        self.algorithm_done = False
        
        # UI state
        self.list_scroll = 0
        self.mode = "select_graph"  # select_graph, select_start, select_end, running, done
        self.graph_offset_x = 0
        self.graph_offset_y = 0

        self.crossroads = []
        self.turn_array = []
        self.text_path = []
        self.chat_active = False
        self.chat_text = ""
        self.chat_caret = 0
        self.chat_view_start = 0
        self.chat_message = ""
        self.chat_message_timer = 0
        self.caret_blink_ms = 500
        self.caret_visible = True
        self.last_caret_toggle = pygame.time.get_ticks()

        if initial_graph and initial_graph in self.available_graphs:
            self.selected_graph_idx = self.available_graphs.index(initial_graph)
            self.load_graph(initial_graph)

        pygame.key.set_repeat(300, 30)
        
        # turns streaming state
        self.turns_stream_path = os.path.join(repo_root, "turns.bin")
        self._turns_written = 0
        
    def _get_available_graphs(self):
        """Get list of available saved graphs"""
        saves_dir = os.path.join(repo_root, 'mapeditor', 'saves')
        graphs = []
        if os.path.exists(saves_dir):
            for folder in os.listdir(saves_dir):
                legacy_path = os.path.join(saves_dir, folder, f'{folder}.json')
                graph_path = os.path.join(saves_dir, folder, 'graph.json')
                if os.path.isfile(legacy_path) or os.path.isfile(graph_path):
                    graphs.append(folder)
        return sorted(graphs)
    
    def load_graph(self, graph_name):
        """Load a graph from JSON"""
        save_root = os.path.join(repo_root, 'mapeditor', 'saves', graph_name)
        graph_path = os.path.join(save_root, 'graph.json')
        legacy_path = os.path.join(save_root, f'{graph_name}.json')
        save_path = graph_path if os.path.isfile(graph_path) else legacy_path
        try:
            with open(save_path, 'r') as f:
                data = json.load(f)
            
            spots_data = data.get('spots', [])
            neigh_data = data.get('neighbors', [])
            cost_data = data.get('weights', [])
            
            new_spots = []
            for sd in spots_data:
                x = sd.get('x', 0)
                y = sd.get('y', 0)
                size = sd.get('size', 5)
                is_wall = sd.get('isWall', False)
                new_spots.append(Spot(x, y, size, is_wall))
            
            self.graph = AStarGraph(new_spots)
            
            # Restore neighbors and weights
            for i, neigh_list in enumerate(neigh_data):
                if i >= len(self.graph.spots):
                    break
                # clear existing neighbor and cost lists
                self.graph.spots[i].neighbors = []
                self.graph.spots[i].costs = []

                # Get corresponding weights list for this spot (if any)
                weights_for_spot = []
                if isinstance(cost_data, list) and i < len(cost_data):
                    weights_for_spot = cost_data[i] if isinstance(cost_data[i], list) else []

                for k, j in enumerate(neigh_list):
                    if 0 <= j < len(self.graph.spots):
                        # use corresponding weight if present, otherwise default to 1
                        w = 1
                        if k < len(weights_for_spot):
                            try:
                                w = float(weights_for_spot[k])
                            except Exception:
                                w = 1
                        # use Spot.addNeighborWithCost to keep neighbors and costs in sync
                        self.graph.spots[i].addNeighborWithCost(self.graph.spots[j], w)
            
            # Calculate offset to center graph in canvas
            self._calculate_graph_offset()
            
            self.start_spot = None
            self.end_spot = None
            self.astar = None
            self.algorithm_running = False
            self.algorithm_done = False
            self.mode = "select_start"
            return True
        except Exception as e:
            print(f"Error loading graph {graph_name}: {e}")
            return False
    
    def _calculate_graph_offset(self):
        """Calculate offset to center the graph in the canvas"""
        if not self.graph or len(self.graph.spots) == 0:
            self.graph_offset_x = 0
            self.graph_offset_y = 0
            return
        
        # Find bounding box
        min_x = min(spot.position.x for spot in self.graph.spots)
        max_x = max(spot.position.x for spot in self.graph.spots)
        min_y = min(spot.position.y for spot in self.graph.spots)
        max_y = max(spot.position.y for spot in self.graph.spots)
        
        # Canvas area (right of sidebar)
        canvas_width = SCREEN_WIDTH - 300
        canvas_height = SCREEN_HEIGHT
        canvas_center_x = 300 + canvas_width / 2
        canvas_center_y = canvas_height / 2
        
        # Graph center
        graph_center_x = (min_x + max_x) / 2
        graph_center_y = (min_y + max_y) / 2
        
        # Calculate offset
        self.graph_offset_x = canvas_center_x - graph_center_x
        self.graph_offset_y = canvas_center_y - graph_center_y
    
    def get_spot_at_position(self, x, y, radius=15):
        """Find a spot near the given position"""
        if not self.graph:
            return None
        for spot in self.graph.spots:
            # Convert screen coordinates to graph coordinates using offset
            spot_screen_x = spot.position.x + self.graph_offset_x
            spot_screen_y = spot.position.y + self.graph_offset_y
            dist = math.sqrt((spot_screen_x - x) ** 2 + (spot_screen_y - y) ** 2)
            if dist <= radius:
                return spot
        return None
    
    def run_astar_step(self):
        """Run one step of A* algorithm"""
        if self.astar and not self.algorithm_done:
            self.astar.update()
            # compute turns for current path and stream any newly discovered turns
            if not self.astar.noSolution and self.astar.path:
                self.crossroads = self.get_all_crossroads(self.astar.path)
                turns_list = self.get_all_turns(self.crossroads, self.astar.path)
                if len(turns_list) > self._turns_written:
                    new_turns = turns_list[self._turns_written:]
                    try:
                        with open(self.turns_stream_path, 'ab') as fh:
                            for t in new_turns:
                                fh.write(bytes([int(t) & 0xFF]))
                            fh.flush()
                            try:
                                os.fsync(fh.fileno())
                            except Exception:
                                pass
                    except Exception as e:
                        print(f"Error appending turns to stream: {e}")
                    self._turns_written = len(turns_list)

            if self.astar.isDone:
                self.algorithm_done = True
                self.algorithm_running = False
                self.mode = "done"
                if not self.astar.noSolution and self.astar.path:
                    # compute final correct turn list and overwrite the turns file
                    self.turn_array = self.get_all_turns(self.crossroads, self.astar.path)
                    try:
                        with open(self.turns_stream_path, 'wb') as fh:
                            fh.write(bytearray(self.turn_array))
                            fh.flush()
                            try:
                                os.fsync(fh.fileno())
                            except Exception:
                                pass
                    except Exception as e:
                        print(f"Error writing final turns file: {e}")
                    for i, turn in enumerate(self.turn_array):
                        print(f"Turn {i}: {turn}")

    
    def save_turns_to_file(self, filename="turns.bin"):
        """Save the turn array to a binary file"""
        if not self.turn_array:
            print("No turns to save.")
            return
        try:
            with open(filename, 'wb') as f:
                f.write(bytearray(self.turn_array))
            print(f"Turn array saved to {filename}")
        except Exception as e:
            print(f"Error saving turn array: {e}")

    def draw_sidebar(self):
        """Draw left sidebar with graph selection"""
        sidebar_rect = pygame.Rect(0, 0, 300, SCREEN_HEIGHT)
        pygame.draw.rect(self.screen, DARK_GRAY, sidebar_rect)
        pygame.draw.line(self.screen, GRAY, (300, 0), (300, SCREEN_HEIGHT), 2)
        
        y_offset = 10
        
        # Title
        title = self.title_font.render("Saved Graphs", True, WHITE)
        self.screen.blit(title, (10, y_offset))
        y_offset += 30
        
        # Draw graph list
        for i, graph_name in enumerate(self.available_graphs):
            is_selected = i == self.selected_graph_idx
            color = YELLOW if is_selected else WHITE
            
            text = self.font.render(f"• {graph_name}", True, color)
            self.screen.blit(text, (20, y_offset))
            y_offset += 25
        
        # Status area
        status_y = SCREEN_HEIGHT - 220
        pygame.draw.line(self.screen, GRAY, (10, status_y), (290, status_y), 1)
        
        status_y += 10
        status_text = self.title_font.render("Status", True, WHITE)
        self.screen.blit(status_text, (10, status_y))
        status_y += 30
        
        if self.graph:
            # Graph info
            info = f"Spots: {len(self.graph.spots)}"
            text = self.font.render(info, True, BLUE)
            self.screen.blit(text, (15, status_y))
            status_y += 25
            
            # Mode
            mode_text = "Mode: " + self.mode.replace('_', ' ').title()
            text = self.font.render(mode_text, True, GREEN)
            self.screen.blit(text, (15, status_y))
        else:
            text = self.font.render("No graph loaded", True, RED)
            self.screen.blit(text, (15, status_y))

        # Chat input area
        chat_box = self._chat_rect()
        pygame.draw.rect(self.screen, CHAT_BG, chat_box)
        pygame.draw.rect(self.screen, CHAT_BORDER, chat_box, 2 if self.chat_active else 1)
        prompt = "Chat: " + self.chat_text
        if not self.chat_text and not self.chat_active:
            prompt = "Chat: type route (requires 'start')"
        # Render tail of text so it "moves along" as it exceeds the bar width
        rendered, start_idx = self._fit_text_to_width(prompt, chat_box.width - 12)
        chat_text = self.font.render(rendered, True, BLACK)
        self.screen.blit(chat_text, (chat_box.x + 6, chat_box.y + 6))

        if self.chat_active and self.caret_visible:
            caret_x = chat_box.x + 6 + self._text_width(rendered[:self._prompt_caret_index(prompt) - start_idx])
            caret_y = chat_box.y + 6
            caret_h = self.font.get_height()
            pygame.draw.line(self.screen, BLACK, (caret_x, caret_y), (caret_x, caret_y + caret_h), 1)

        if self.chat_message:
            lines = self._wrap_text(self.chat_message, chat_box.width - 12)
            for i, line in enumerate(lines[:2]):
                msg = self.font.render(line, True, YELLOW)
                self.screen.blit(msg, (chat_box.x + 6, chat_box.y - 18 - (len(lines[:2]) - 1 - i) * 16))
    
    def draw_canvas(self):
        """Draw the main canvas with graph visualization"""
        canvas_rect = pygame.Rect(300, 0, SCREEN_WIDTH - 300, SCREEN_HEIGHT)
        pygame.draw.rect(self.screen, WHITE, canvas_rect)
        
        if not self.graph:
            text = self.font.render("Load a graph from the left panel", True, GRAY)
            text_rect = text.get_rect(center=(canvas_rect.centerx, canvas_rect.centery))
            self.screen.blit(text, text_rect)
            return
        
        # Draw graph with offset
        self._draw_graph_with_offset()
        
        # Draw start spot
        if self.start_spot:
            screen_x = self.start_spot.position.x + self.graph_offset_x
            screen_y = self.start_spot.position.y + self.graph_offset_y
            pygame.draw.circle(self.screen, GREEN, 
                             (screen_x, screen_y), 
                             self.start_spot.size + 3, 2)
        
        # Draw end spot
        if self.end_spot:
            screen_x = self.end_spot.position.x + self.graph_offset_x
            screen_y = self.end_spot.position.y + self.graph_offset_y
            pygame.draw.circle(self.screen, RED, 
                             (screen_x, screen_y), 
                             self.end_spot.size + 3, 2)
        
        # Draw A* visualization
        if self.astar:
            self._draw_astar_with_offset()
            if not self.astar.noSolution and self.astar.path:
                if len(self.crossroads) > 0:
                    for p in self.crossroads:
                        if p < len(self.astar.path):
                            screen_x = self.astar.path[p].position.x + self.graph_offset_x
                            screen_y = self.astar.path[p].position.y + self.graph_offset_y
                            pygame.draw.circle(self.screen, ORANGE, (screen_x, screen_y), self.astar.path[p].size + 5, 2)
        elif self.text_path:
            self._draw_text_path_with_offset()
        
        # Draw instructions
        instructions_y = 20
        if self.mode == "select_start":
            instr = "Click on a spot to select START point (green)"
            color = GREEN
        elif self.mode == "select_end":
            instr = "Click on a spot to select END point (red)"
            color = RED
        elif self.mode == "running":
            instr = "Press SPACE to step or R to reset"
            color = YELLOW
        elif self.mode == "done":
            instr = "Pathfinding complete! Press R to reset"
            color = BLUE
        elif self.mode == "chat_route":
            instr = "Chat route active. Press R to reset"
            color = ORANGE
        else:
            instr = ""
            color = WHITE
        
        if instr:
            text = self.font.render(instr, True, color)
            self.screen.blit(text, (320, instructions_y))
        
        # Draw stats
        stats_y = SCREEN_HEIGHT - 200
        if self.astar:
            stats = [
                f"Open set: {len(self.astar.openSet)}",
                f"Closed set: {len(self.astar.closeSet)}",
                f"Path length: {len(self.astar.path) if self.astar.path else 0}",
                f"Solution found: {self.astar.isDone and not self.astar.noSolution}"
                # f"Is Uturn: {self.checkforUturn(self.astar.path) if self.astar.path else 'N/A'}"
            ]
            for i, stat in enumerate(stats):
                text = self.font.render(stat, True, BLACK)
                self.screen.blit(text, (SCREEN_WIDTH - 300, stats_y + i * 20))
    
    def draw_controls(self):
        """Draw control instructions"""
        controls_y = SCREEN_HEIGHT - 50
        controls = [
            "UP/DOWN: Select graph",
            "ENTER: Load selected graph",
            # "SPACE: Step A* / Run",
            "R: Reset",
            "ESC: Quit"
        ]
        
        for i, control in enumerate(controls[:4]):
            text = self.font.render(control, True, DARK_GRAY)
            self.screen.blit(text, (320 + i * 330, controls_y))
    
    def handle_events(self):
        """Handle user input"""
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.is_running = False
            
            elif event.type == pygame.KEYDOWN:
                if self.chat_active:
                    if event.key == pygame.K_RETURN:
                        self._submit_chat()
                    elif event.key == pygame.K_BACKSPACE:
                        if self.chat_caret > 0:
                            self.chat_text = self.chat_text[:self.chat_caret - 1] + self.chat_text[self.chat_caret:]
                            self.chat_caret -= 1
                            self._ensure_caret_visible()
                    elif event.key == pygame.K_LEFT:
                        if self.chat_caret > 0:
                            self.chat_caret -= 1
                            self._ensure_caret_visible()
                    elif event.key == pygame.K_RIGHT:
                        if self.chat_caret < len(self.chat_text):
                            self.chat_caret += 1
                            self._ensure_caret_visible()
                    else:
                        if event.unicode and len(self.chat_text) < 120:
                            self.chat_text = (
                                self.chat_text[:self.chat_caret]
                                + event.unicode
                                + self.chat_text[self.chat_caret:]
                            )
                            self.chat_caret += 1
                            self._ensure_caret_visible()
                    continue

                if event.key == pygame.K_ESCAPE:
                    self.is_running = False
                
                elif event.key == pygame.K_UP:
                    self.selected_graph_idx = max(0, self.selected_graph_idx - 1)
                
                elif event.key == pygame.K_DOWN:
                    self.selected_graph_idx = min(len(self.available_graphs) - 1, 
                                                 self.selected_graph_idx + 1)
                
                elif event.key == pygame.K_RETURN:
                    if self.available_graphs:
                        self.load_graph(self.available_graphs[self.selected_graph_idx])
                
                elif event.key == pygame.K_SPACE:
                    if self.mode == "select_start" or self.mode == "select_end":
                        # Run continuous A* if we have both start and end
                        if self.start_spot and self.end_spot and not self.astar:
                            self.start_astar()
                        elif self.astar and not self.algorithm_done:
                            self.algorithm_running = not self.algorithm_running
                    elif self.mode == "running":
                        if not self.algorithm_running:
                            self.run_astar_step()
                
                elif event.key == pygame.K_r:
                    self.reset_pathfinding()
            
            elif event.type == pygame.MOUSEBUTTONDOWN:
                if event.button == 1:  # Left click
                    x, y = event.pos
                    if self._chat_rect().collidepoint(x, y):
                        self.chat_active = True
                        self._set_caret_from_click(x)
                        self._ensure_caret_visible()
                        return
                    else:
                        self.chat_active = False
                    # Only register clicks on canvas area
                    if x > 300:
                        if self.graph:
                            spot = self.get_spot_at_position(x, y)
                            if spot:
                                if self.mode == "select_start":
                                    self.start_spot = spot
                                    self.mode = "select_end"
                                elif self.mode == "chat_route":
                                    self.chat_message = "Chat route active. Press R to reset."
                                    self.chat_message_timer = pygame.time.get_ticks()
                                elif self.mode == "select_end":
                                    self.end_spot = spot
                                    self.mode = "ready_to_run"
                                    # Automatically start A*
                                    self.start_astar()
    
    def start_astar(self):
        """Start the A* algorithm"""
        if self.start_spot and self.end_spot:
            self.graph.clearSpots()
            self.astar = AStar(self.start_spot, self.end_spot)
            self.algorithm_running = True
            self.algorithm_done = False
            self.mode = "running"
            self.text_path = []
            # prepare/clear turns stream file for this run
            try:
                with open(self.turns_stream_path, 'wb'):
                    pass
            except Exception:
                pass
            self._turns_written = 0

    def _chat_rect(self):
        return pygame.Rect(10, SCREEN_HEIGHT - 80, 280, 30)

    def _text_width(self, text):
        return self.font.size(text)[0]

    def _wrap_text(self, text, max_width):
        words = text.split()
        if not words:
            return [""]
        lines = []
        current = words[0]
        for word in words[1:]:
            candidate = current + " " + word
            if self._text_width(candidate) <= max_width:
                current = candidate
            else:
                lines.append(current)
                current = word
        lines.append(current)
        return lines

    def _prompt_caret_index(self, prompt):
        if prompt.startswith("Chat: "):
            return len("Chat: ") + self.chat_caret
        return len(prompt)

    def _fit_text_to_width(self, text, max_width):
        if self._text_width(text) <= max_width:
            self.chat_view_start = 0
            return text, 0
        start = min(self.chat_view_start, len(text))
        end = len(text)
        while self._text_width(text[start:end]) > max_width and end > start:
            end -= 1
        self.chat_view_start = start
        return text[start:end], start

    def _ensure_caret_visible(self):
        prompt = "Chat: " + self.chat_text
        caret_index = self._prompt_caret_index(prompt)
        max_width = self._chat_rect().width - 12
        if self._text_width(prompt) <= max_width:
            self.chat_view_start = 0
            return
        if caret_index < self.chat_view_start:
            self.chat_view_start = caret_index
        while self._text_width(prompt[self.chat_view_start:caret_index]) > max_width and self.chat_view_start < caret_index:
            self.chat_view_start += 1
        while self.chat_view_start > 0 and self._text_width(prompt[self.chat_view_start - 1:]) <= max_width:
            self.chat_view_start -= 1

    def _set_caret_from_click(self, x):
        chat_box = self._chat_rect()
        prompt = "Chat: " + self.chat_text
        rendered, start_idx = self._fit_text_to_width(prompt, chat_box.width - 12)
        rel_x = max(0, x - (chat_box.x + 6))
        # Find closest index in rendered string based on width
        caret_in_render = 0
        for i in range(len(rendered) + 1):
            if self._text_width(rendered[:i]) >= rel_x:
                caret_in_render = i
                break
        else:
            caret_in_render = len(rendered)
        caret_abs = start_idx + caret_in_render
        if prompt.startswith("Chat: "):
            self.chat_caret = max(0, min(len(self.chat_text), caret_abs - len("Chat: ")))
        else:
            self.chat_caret = len(self.chat_text)

    def _load_turns_from_file(self, filepath):
        try:
            with open(filepath, "rb") as f:
                return list(f.read())
        except Exception as e:
            self.chat_message = f"Error reading turns.bin: {e}"
            self.chat_message_timer = pygame.time.get_ticks()
            return []

    def _angle_between(self, ax, ay, bx, by):
        ang = math.degrees(math.atan2(ay, ax) - math.atan2(by, bx))
        return (ang + 180) % 360 - 180

    def _neighbors_sorted(self, spot):
        return sorted(spot.neighbors, key=lambda s: (s.position.x, s.position.y))

    def _turn_code_like_astar(self, prev_spot, curr_spot, next_spot):
        # Mirror print_directions() logic for turn coding
        fx = next_spot.position.x - curr_spot.position.x
        fy = next_spot.position.y - curr_spot.position.y
        nx = prev_spot.position.x - curr_spot.position.x
        ny = prev_spot.position.y - curr_spot.position.y
        ang = self._angle_between(ny, nx, fy, fx)
        if abs(ang) < 30:
            return 1
        if ang > 0:
            return 3 if self.is_Uturn_node(prev_spot, next_spot) else 2
        return 0 if self.is_Uturn_node(prev_spot, next_spot) else 1

    def is_Uturn_node(self, prev_spot, next_spot):
        # Match is_Uturn distance rule without requiring astar.path indices
        return self.astar._distance(prev_spot, next_spot) < 30 if self.astar else (
            ((next_spot.position.x - prev_spot.position.x) ** 2 +
             (next_spot.position.y - prev_spot.position.y) ** 2) ** 0.5 < 30
        )

    def _pick_by_turn(self, prev_spot, curr_spot, candidates, turn_code):
        if prev_spot is None:
            return candidates[0] if candidates else None

        matching = [n for n in candidates if self._turn_code_like_astar(prev_spot, curr_spot, n) == turn_code]
        if matching:
            return matching[0]

        # Fallback: pick by angle heuristic
        fx = curr_spot.position.x - prev_spot.position.x
        fy = curr_spot.position.y - prev_spot.position.y
        best = None
        best_score = None
        for n in candidates:
            nx = n.position.x - curr_spot.position.x
            ny = n.position.y - curr_spot.position.y
            ang = self._angle_between(ny, nx, fy, fx)
            if turn_code == 1:
                score = abs(ang)
            elif turn_code == 2:
                score = -ang if ang > 0 else float("inf")
            elif turn_code == 0:
                score = ang if ang < 0 else float("inf")
            else:
                score = abs(abs(ang) - 180)
            if best is None or score < best_score:
                best = n
                best_score = score
        return best

    def _build_path_from_turns(self, turns_start_to_end):
        if not self.start_spot or not self.graph:
            return []

        def walk_from(first_neighbor):
            path = [self.start_spot, first_neighbor]
            prev = self.start_spot
            curr = first_neighbor
            turns = list(turns_start_to_end)
            steps = 0
            max_steps = len(self.graph.spots) * 4

            while steps < max_steps:
                steps += 1
                neighbors = [n for n in self._neighbors_sorted(curr) if not n.isWall]
                if len(neighbors) == 0:
                    break

                # Straight segments
                if len(neighbors) == 2 and prev in neighbors:
                    nxt = neighbors[0] if neighbors[1] is prev else neighbors[1]
                    path.append(nxt)
                    prev, curr = curr, nxt
                    continue

                if len(neighbors) > 2 and turns:
                    turn = turns.pop(0)
                    candidates = neighbors if turn == 3 else ([n for n in neighbors if n is not prev] or neighbors)
                    nxt = self._pick_by_turn(prev, curr, candidates, turn)
                else:
                    # No turns left: keep going straight if possible
                    if prev is None:
                        nxt = neighbors[0]
                    else:
                        candidates = [n for n in neighbors if n is not prev]
                        if not candidates:
                            break
                        nxt = self._pick_by_turn(prev, curr, candidates, 1)

                if nxt is None:
                    break
                path.append(nxt)
                prev, curr = curr, nxt

                if not turns and len(curr.neighbors) <= 1:
                    break

            return path, turns

        neighbors = [n for n in self._neighbors_sorted(self.start_spot) if not n.isWall]
        if not neighbors:
            return [self.start_spot]

        # If no turns, pick first neighbor and go straight as far as possible
        path = None
        if not turns_start_to_end:
            path, _ = walk_from(neighbors[0])
        else:
            best = None
            for n in neighbors:
                candidate_path, remaining = walk_from(n)
                consumed = len(turns_start_to_end) - len(remaining)
                score = (consumed, len(candidate_path))
                if best is None or score > best[0]:
                    best = (score, candidate_path)
            path = best[1] if best else [self.start_spot, neighbors[0]]

        # Reverse to match A* path order (end -> start)
        return list(reversed(path))

    def _build_text_path_from_turns(self, filepath):
        turns_reversed = self._load_turns_from_file(filepath)
        turns_start_to_end = list(reversed(turns_reversed))
        return self._build_path_from_turns(turns_start_to_end)

    def _submit_chat(self):
        text = self.chat_text.strip()
        if not text:
            return
        if not self.start_spot:
            self.chat_message = "Select a START point first."
            self.chat_message_timer = pygame.time.get_ticks()
            return

        steps, clarification = parse_text_to_steps(text)
        if clarification:
            self.chat_message = clarification
            self.chat_message_timer = pygame.time.get_ticks()
            return

        try:
            write_turns_bin(steps, os.path.join(repo_root, "turns.bin"))
            self.chat_message = f"Wrote turns.bin: {steps}"
            self.chat_message_timer = pygame.time.get_ticks()
            # Text route is separate from A* run; clear any existing end/astar
            self.astar = None
            self.end_spot = None
            self.algorithm_running = False
            self.algorithm_done = False
            self.mode = "chat_route"
            self.text_path = self._build_text_path_from_turns(os.path.join(repo_root, "turns.bin"))
            self.chat_text = ""
            self.chat_caret = 0
            self.chat_view_start = 0
        except Exception as e:
            self.chat_message = f"Write failed: {e}"
            self.chat_message_timer = pygame.time.get_ticks()
            self.text_path = []
    
    def is_crossroad(self, waypoint, path):
        return len(path[waypoint].neighbors) > 2
    
    def _draw_graph_with_offset(self):
        """Draw the graph with offset applied"""
        if not self.graph:
            return
        for spot in self.graph.spots:
            screen_x = spot.position.x + self.graph_offset_x
            screen_y = spot.position.y + self.graph_offset_y
            color = (100, 100, 100) if spot.isWall else BLUE
            pygame.draw.circle(self.screen, color, (screen_x, screen_y), spot.size)
        
        # Draw edges
        for spot in self.graph.spots:
            for neighbor in spot.neighbors:
                screen_x1 = spot.position.x + self.graph_offset_x
                screen_y1 = spot.position.y + self.graph_offset_y
                screen_x2 = neighbor.position.x + self.graph_offset_x
                screen_y2 = neighbor.position.y + self.graph_offset_y
                pygame.draw.line(self.screen, LIGHT_GRAY, (screen_x1, screen_y1), (screen_x2, screen_y2), 1)
    
    def _draw_astar_with_offset(self):
        """Draw A* visualization with offset applied"""
        if not self.astar:
            return
        for spot in self.astar.openSet:
            screen_x = spot.position.x + self.graph_offset_x
            screen_y = spot.position.y + self.graph_offset_y
            pygame.draw.circle(self.screen, GREEN, (screen_x, screen_y), spot.size + 1, 1)
        
        for spot in self.astar.closeSet:
            screen_x = spot.position.x + self.graph_offset_x
            screen_y = spot.position.y + self.graph_offset_y
            pygame.draw.circle(self.screen, RED, (screen_x, screen_y), spot.size + 1, 1)
        
        if self.astar.path:
            for i in range(len(self.astar.path) - 1):
                screen_x1 = self.astar.path[i].position.x + self.graph_offset_x
                screen_y1 = self.astar.path[i].position.y + self.graph_offset_y
                screen_x2 = self.astar.path[i + 1].position.x + self.graph_offset_x
                screen_y2 = self.astar.path[i + 1].position.y + self.graph_offset_y
                pygame.draw.line(self.screen, YELLOW, (screen_x1, screen_y1), (screen_x2, screen_y2), 2)

    def _draw_text_path_with_offset(self):
        """Draw text-driven path visualization with offset applied"""
        if not self.text_path:
            return
        for i in range(len(self.text_path) - 1):
            screen_x1 = self.text_path[i].position.x + self.graph_offset_x
            screen_y1 = self.text_path[i].position.y + self.graph_offset_y
            screen_x2 = self.text_path[i + 1].position.x + self.graph_offset_x
            screen_y2 = self.text_path[i + 1].position.y + self.graph_offset_y
            pygame.draw.line(self.screen, ORANGE, (screen_x1, screen_y1), (screen_x2, screen_y2), 2)
    
    def get_all_crossroads(self, path):
        crossroads = []
        for i in range(len(path)):
            if self.is_crossroad(i, path):
                crossroads.append(i)
        return crossroads

    def get_all_turns(self, crossroads, path):
        """Return list of turn types for given crossroads and path (no I/O)."""
        turn_array = []
        for wp in crossroads:
            turn_type = self.print_directions(wp, path)
            if turn_type is not None:
                turn_array.append(turn_type)
        return turn_array

    def get_all_turns_streamed(self, crossroads, path, filename=None):
        """Compute turn types for crossroads and stream each turn byte to `filename` as soon as it's computed.
        Returns the full list of computed turns.
        """
        turn_array = []
        if filename is None:
            # just compute without writing
            for wp in crossroads:
                turn_type = self.print_directions(wp, path)
                if turn_type is not None:
                    turn_array.append(turn_type)
            return turn_array

        try:
            # open file for writing (truncate) so we write a fresh stream
            with open(filename, 'wb') as fh:
                for wp in crossroads:
                    turn_type = self.print_directions(wp, path)
                    if turn_type is None:
                        continue
                    # append to in-memory list
                    turn_array.append(turn_type)
                    # write one byte and flush to disk for real-time visibility
                    try:
                        fh.write(bytes([int(turn_type) & 0xFF]))
                        fh.flush()
                        try:
                            os.fsync(fh.fileno())
                        except Exception:
                            pass
                    except Exception:
                        # non-fatal: continue computing remaining turns
                        pass
        except Exception as e:
            # surface error in chat message area if available
            try:
                self.chat_message = f"Error writing turns stream: {e}"
                self.chat_message_timer = pygame.time.get_ticks()
            except Exception:
                print(f"Error writing turns stream: {e}")

        return turn_array

    def print_directions(self, waypoint, path):
        # Guard: need a next waypoint to compare against
        if waypoint < 0 or waypoint >= len(path) - 1:
            return

        cur = path[waypoint]
        nxt = path[waypoint + 1]

        # forward vector from current waypoint to next waypoint
        fx = nxt.position.x - cur.position.x
        fy = nxt.position.y - cur.position.y

        print("Next waypoint at ({}, {})".format(nxt.position.x, nxt.position.y))
        print(f"Forward vector: ({fx}, {fy})")

        n = path[waypoint-1]
        # vector from current waypoint to this neighbor
        nx = n.position.x - cur.position.x
        ny = n.position.y - cur.position.y

        print(f"Distance: {self.astar._distance(nxt, n)}")

        # cross product (z component) tells left/right relative to forward vector
        cross = fx * ny - fy * nx

        # compute signed angle (degrees) between forward and neighbor vector
        ang = math.degrees(math.atan2(ny, nx) - math.atan2(fy, fx))
        ang = (ang + 180) % 360 - 180  # normalize to [-180, 180]

        if abs(ang) < 30:
            rel = "ahead"
            return 1
        elif ang > 0:
            rel = "left"
            if self.is_Uturn(waypoint, path):
                rel = "U-turn"
                return 3
            else:
                return 2
        else:
            rel = "ahead"
            if self.is_Uturn(waypoint, path):
                rel = "right"
                return 0
            else:
                return 1

        # print(f"Neighbor at ({n.position.x}, {n.position.y}) -> {rel} (angle {ang:.1f}°, cross {cross:.1f})")
            
            

    def is_Uturn(self, waypoint, path):

        if not self.astar.noSolution and len(path) > 0:
            if waypoint > 0 and waypoint < len(path) - 1:
                print("DEBUG: Checking U-turn at waypoint", waypoint)
                print("----------------------------------")
                print(f"Distance: {self.astar._distance(path[waypoint - 1], path[waypoint + 1])}")
                if self.astar._distance(path[waypoint - 1], path[waypoint + 1]) < 30:
                    print("U turn true")
                    return True
            
        return False
    
    def checkforUturn(self, path):
        for i in range(1, len(path) - 1):
            if self.is_Uturn(i, path):
                return True
        return False

    def reset_pathfinding(self):
        """Reset pathfinding"""
        if self.graph:
            self.graph.clearSpots()
        self.astar = None
        self.start_spot = None
        self.end_spot = None
        self.algorithm_running = False
        self.algorithm_done = False
        self.text_path = []
        self.mode = "select_start"
    
    def run(self):
        """Main loop"""
        while self.is_running:
            self.handle_events()
            
            # Update A* if running
            if self.algorithm_running:
                self.run_astar_step()

            # Draw
            self.screen.fill(WHITE)
            self.draw_sidebar()
            self.draw_canvas()
            self.draw_controls()

            # Clear transient chat message after 4 seconds
            if self.chat_message_timer:
                if pygame.time.get_ticks() - self.chat_message_timer > 4000:
                    self.chat_message = ""
                    self.chat_message_timer = 0

            if self.chat_active:
                now = pygame.time.get_ticks()
                if now - self.last_caret_toggle > self.caret_blink_ms:
                    self.caret_visible = not self.caret_visible
                    self.last_caret_toggle = now
            
            pygame.display.flip()
            self.clock.tick(FPS)
        
        pygame.quit()


if __name__ == "__main__":
    initial = sys.argv[1] if len(sys.argv) > 1 else None
    visualizer = GraphVisualizer(initial_graph=initial)
    visualizer.run()
