# A* Pathfinding Visualizer

A pygame-based interactive visualization tool for loading saved graphs from the mapeditor and running A* pathfinding algorithm with visual feedback.

## Features

- **Load Saved Graphs**: Browse and load graphs saved from the mapeditor
- **Interactive Point Selection**: Click to select start and end points on the graph
- **A* Algorithm Visualization**: Watch the algorithm explore the graph in real-time
- **Algorithm Control**: Step through the algorithm or run it automatically
- **Visual Feedback**:
  - Green circles: Start point
  - Red circles: End point
  - Blue path: Computed shortest path
  - Green nodes: Open set (to be evaluated)
  - Red nodes: Closed set (already evaluated)

## Running the Visualizer

From the duckietown root directory:

```bash
python3 -m utils.graph.AStarTest
```

Or directly:

```bash
python3 utils/graph/AStarTest.py
```

## Controls

### Graph Selection
- **UP/DOWN Arrow Keys**: Navigate through available saved graphs
- **ENTER**: Load the selected graph

### Pathfinding
- **LEFT CLICK**: Select points on the canvas
  1. First click: Select start point (will turn green)
  2. Second click: Select end point (will turn red)
  3. A* algorithm automatically starts
  
- **SPACE**: Step through the algorithm one iteration at a time (when running)
- **R**: Reset the current pathfinding session (keeps graph loaded)
- **ESC**: Quit the application

## Workflow

1. **Launch the Application**
   - Start the visualizer
   - See list of available graphs in the left panel

2. **Load a Graph**
   - Use UP/DOWN arrows to select a graph
   - Press ENTER to load it
   - Graph will be displayed in the main canvas area

3. **Select Points**
   - The status will show "Mode: Select Start"
   - Click on any spot in the graph to select the start point (appears green)
   - Status changes to "Select End"
   - Click on another spot to select the end point (appears red)
   - A* algorithm automatically initializes

4. **Run A* Algorithm**
   - Watch the algorithm run automatically
   - Green nodes show spots being evaluated (open set)
   - Red nodes show fully evaluated spots (closed set)
   - Blue path shows the computed route from start to end
   - Algorithm completes when it finds the path or determines no solution exists

5. **Reset and Repeat**
   - Press R to reset and select new start/end points
   - Graph stays loaded
   - Or press ENTER on a different graph to load a new one

## Graph Format

Graphs are saved in JSON format with:
- **spots**: Array of node positions with properties (x, y, size, isWall)
- **neighbors**: Array of neighbor indices for each spot showing connectivity

Example structure:
```json
{
  "spots": [
    {"x": 100.0, "y": 100.0, "size": 5, "isWall": false},
    ...
  ],
  "neighbors": [
    [1, 2],
    [0, 2],
    ...
  ]
}
```

## Graph Location

Graphs are loaded from: `mapeditor/saves/[graph_name]/[graph_name].json`

## Algorithm Details

The A* algorithm uses:
- **Heuristic**: Euclidean distance between nodes
- **Cost**: Uniform cost (1) per edge
- **Goal**: Find shortest path from start to end point

The visualizer shows real-time progress with color coding:
- **Open Set (Green)**: Nodes discovered but not yet fully evaluated
- **Closed Set (Red)**: Nodes that have been fully evaluated
- **Path (Blue)**: Final shortest path found

## Status Panel

The left sidebar shows:
- List of available graphs
- Current graph loaded (number of spots)
- Current mode (selecting start, selecting end, running, done)
- Real-time statistics during pathfinding

## Tips

- Select spots near their centers for accurate selection (click radius is 15 pixels)
- If no path exists, the algorithm will complete with an empty path
- The visualizer updates 60 times per second for smooth animation
- You can step through the algorithm frame-by-frame using SPACE for detailed inspection

## Troubleshooting

**No graphs appear in the list?**
- Ensure you have saved graphs in `mapeditor/saves/`
- Graph files must be in format: `saves/[name]/[name].json`

**Can't click spots?**
- Make sure you're clicking in the white canvas area (right side of the screen)
- Try clicking closer to the center of the spot visualization

**A* not starting?**
- Make sure both start and end points are selected
- The algorithm should start automatically after selecting the end point

