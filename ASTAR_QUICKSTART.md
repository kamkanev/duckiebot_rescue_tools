# Quick Start Guide: A* Pathfinding Visualizer

## 1. Start the Application
```bash
python3 -m utils.graph.AStarTest
```

You should see a window with:
- Left panel (dark gray): List of saved graphs
- Right panel (white): Canvas for visualization

## 2. Load a Graph
- Use **UP/DOWN arrow keys** to highlight a graph in the list
- Press **ENTER** to load it
- The graph will appear in the main canvas with all its spots and connections

## 3. Select Start Point
- The status panel will say "Mode: Select Start"
- **Click on any spot** in the graph to select it as the starting point
- The selected spot will be highlighted with a green circle

## 4. Select End Point
- The status panel will say "Mode: Select End"
- **Click on another spot** to select it as the ending point
- The selected spot will be highlighted with a red circle
- **A* algorithm automatically starts**

## 5. Watch the Algorithm
- **Green nodes**: Being evaluated (open set)
- **Red nodes**: Already evaluated (closed set)  
- **Blue path**: The final shortest path found

## 6. Control Options During Pathfinding
- **SPACE**: Step through one iteration (or toggle auto-run on/off)
- **R**: Reset and select new start/end points (graph stays loaded)
- **UP/DOWN + ENTER**: Load a different graph
- **ESC**: Exit the application

## Visual Legend
```
Green Circle    = Start point
Red Circle      = End point
Blue Path       = Calculated shortest route
Green Nodes     = Currently being evaluated
Red Nodes       = Already fully evaluated
Gray Arrows     = Connections between spots (from editor)
```

## Example Workflow

1. Launch the app
2. See "save" in the graph list
3. Press ENTER to load it
4. Click on a spot in the lower-right area (start point, turns green)
5. Click on a spot in the upper-left area (end point, turns red)
6. Watch as the algorithm finds the path (blue line appears)
7. Press R to try again with different points
8. Press ESC to quit

## Tips
- If the canvas is all white, the graph loaded but might be off-screen. Try a different graph.
- The clickable area is a 15-pixel radius around each spot
- If you accidentally select the wrong point, press R to reset and try again
