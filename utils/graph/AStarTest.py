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

class GraphVisualizer:
    def __init__(self):
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

        self.crossroads = []
        self.turn_array = []
        
    def _get_available_graphs(self):
        """Get list of available saved graphs"""
        saves_dir = os.path.join(repo_root, 'mapeditor', 'saves')
        graphs = []
        if os.path.exists(saves_dir):
            for folder in os.listdir(saves_dir):
                json_path = os.path.join(saves_dir, folder, f'{folder}.json')
                if os.path.isfile(json_path):
                    graphs.append(folder)
        return sorted(graphs)
    
    def load_graph(self, graph_name):
        """Load a graph from JSON"""
        save_path = os.path.join(repo_root, 'mapeditor', 'saves', graph_name, f'{graph_name}.json')
        try:
            with open(save_path, 'r') as f:
                data = json.load(f)
            
            spots_data = data.get('spots', [])
            neigh_data = data.get('neighbors', [])
            
            new_spots = []
            for sd in spots_data:
                x = sd.get('x', 0)
                y = sd.get('y', 0)
                size = sd.get('size', 5)
                is_wall = sd.get('isWall', False)
                new_spots.append(Spot(x, y, size, is_wall))
            
            self.graph = AStarGraph(new_spots)
            
            # Restore neighbors
            for i, neigh_list in enumerate(neigh_data):
                if i >= len(self.graph.spots):
                    break
                self.graph.spots[i].neighbors = []
                for j in neigh_list:
                    if 0 <= j < len(self.graph.spots):
                        self.graph.spots[i].neighbors.append(self.graph.spots[j])
            
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
    
    def get_spot_at_position(self, x, y, radius=15):
        """Find a spot near the given position"""
        if not self.graph:
            return None
        for spot in self.graph.spots:
            dist = math.sqrt((spot.position.x - x) ** 2 + (spot.position.y - y) ** 2)
            if dist <= radius:
                return spot
        return None
    
    def run_astar_step(self):
        """Run one step of A* algorithm"""
        if self.astar and not self.algorithm_done:
            self.astar.update()
            if self.astar.isDone:
                self.algorithm_done = True
                self.algorithm_running = False
                self.mode = "done"
                if not self.astar.noSolution and self.astar.path:
                    self.crossroads = self.get_all_crossroads(self.astar.path)
                    self.turn_array = self.get_all_turns(self.crossroads, self.astar.path)
                    for i, turn in enumerate(self.turn_array):
                        print(f"Turn {i}: {turn}")
                    #TODO: set the turn array
    
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
        status_y = SCREEN_HEIGHT - 150
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
    
    def draw_canvas(self):
        """Draw the main canvas with graph visualization"""
        canvas_rect = pygame.Rect(300, 0, SCREEN_WIDTH - 300, SCREEN_HEIGHT)
        pygame.draw.rect(self.screen, WHITE, canvas_rect)
        
        if not self.graph:
            text = self.font.render("Load a graph from the left panel", True, GRAY)
            text_rect = text.get_rect(center=(canvas_rect.centerx, canvas_rect.centery))
            self.screen.blit(text, text_rect)
            return
        
        # Draw graph
        self.graph.draw(self.screen)
        
        # Draw start spot
        if self.start_spot:
            pygame.draw.circle(self.screen, GREEN, 
                             (self.start_spot.position.x, self.start_spot.position.y), 
                             self.start_spot.size + 3, 2)
        
        # Draw end spot
        if self.end_spot:
            pygame.draw.circle(self.screen, RED, 
                             (self.end_spot.position.x, self.end_spot.position.y), 
                             self.end_spot.size + 3, 2)
        
        # Draw A* visualization
        if self.astar:
            self.astar.debugDraw(self.screen)
            if not self.astar.noSolution and self.astar.path:
                if len(self.crossroads) > 0:
                    for p in self.crossroads:
                        if p < len(self.astar.path):
                            pygame.draw.circle(self.screen, ORANGE, self.astar.path[p].position, self.astar.path[p].size + 5, 2)
        
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
                f"Solution found: {self.astar.isDone and not self.astar.noSolution}",
                f"Is Uturn: {self.checkforUturn(self.astar.path) if self.astar.path else 'N/A'}"
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
                    # Only register clicks on canvas area
                    if x > 300:
                        if self.graph:
                            spot = self.get_spot_at_position(x, y)
                            if spot:
                                if self.mode == "select_start":
                                    self.start_spot = spot
                                    self.mode = "select_end"
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
    
    def is_crossroad(self, waypoint, path):
        return len(path[waypoint].neighbors) > 2
    
    def get_all_crossroads(self, path):
        crossroads = []
        for i in range(len(path)):
            if self.is_crossroad(i, path):
                crossroads.append(i)
        return crossroads

    def get_all_turns(self, crossroads, path):
        turn_array = []
        for wp in crossroads:
            turn_type = self.print_directions(wp, path)
            turn_array.append(turn_type)
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
                if self.astar._distance(path[waypoint - 1], path[waypoint + 1]) < 50:
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
            
            pygame.display.flip()
            self.clock.tick(FPS)
        
        pygame.quit()


if __name__ == "__main__":
    visualizer = GraphVisualizer()
    visualizer.run()
