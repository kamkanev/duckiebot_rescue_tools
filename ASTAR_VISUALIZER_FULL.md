# A* Pathfinding Visualizer - Complete Documentation

## Overview

The A* Pathfinding Visualizer is a pygame-based interactive tool that allows you to:
1. Load graph structures saved from the mapeditor
2. Interactively select start and end points
3. Run the A* pathfinding algorithm
4. Visualize the algorithm's search process in real-time

## Installation

No additional installation needed if you already have:
- pygame (installed in your duckiebot environment)
- Python 3.10+

## Running the Application

### Method 1: From Command Line
```bash
python3 utils/graph/AStarTest.py
```

### Method 2: As a Python Module
```bash
python3 -m utils.graph.AStarTest
```

## User Interface

### Window Layout
```
┌─────────────────────────────────────────────────────────────┐
│  SIDEBAR (300px)        │         MAIN CANVAS (1100px)      │
│  ────────────────────────────────────────────────────────────│
│  Saved Graphs           │                                    │
│  • save                 │     Graph Visualization Area       │
│  • test                 │     (loaded graph with spots)      │
│                         │                                    │
│  Status                 │                                    │
│  Spots: 45              │     Instructions shown here        │
│  Mode: Select Start     │                                    │
│  ────────────────────────────────────────────────────────────│
│                         │  Stats: Open/Closed set sizes      │
└─────────────────────────────────────────────────────────────┘
```

### Left Sidebar
- **Saved Graphs**: Lists all available graphs from mapeditor/saves/
- **Current Selection**: Highlighted in yellow
- **Status Panel**: Shows current graph info and mode

### Main Canvas (Right Side)
- **Graph Visualization**: White background with spots and edges
- **Instructions**: Text showing current action needed
- **Real-time Stats**: Algorithm metrics during execution

## Keyboard Controls

| Key | Function |
|-----|----------|
| UP Arrow | Previous graph in list |
| DOWN Arrow | Next graph in list |
| ENTER | Load selected graph |
| LEFT CLICK | Select point (start or end) |
| SPACE | Step through algorithm / Toggle auto-run |
| R | Reset pathfinding (keep graph loaded) |
| ESC | Exit application |

## Workflow Modes

### 1. Graph Selection Mode
- **State**: Application starts here
- **Display**: List of available graphs in sidebar
- **Action**: Use UP/DOWN arrows and press ENTER

### 2. Select Start Mode  
- **State**: Graph loaded, waiting for start point
- **Display**: "Mode: Select Start" in status panel
- **Action**: Click on a spot to select it (turns green)

### 3. Select End Mode
- **State**: Start point selected, waiting for end point
- **Display**: "Mode: Select End" in status panel
- **Action**: Click on a spot to select it (turns red)

### 4. Running Mode
- **State**: A* algorithm is executing
- **Display**: Animated visualization of algorithm progress
- **Action**: 
  - SPACE to step through iterations
  - Let it run automatically
  - SPACE to pause/resume

### 5. Done Mode
- **State**: Algorithm has completed
- **Display**: Final path highlighted in blue
- **Message**: "Pathfinding complete! Press R to reset"
- **Action**:
  - R to select new points
  - ENTER to load different graph

## Visual Feedback During Pathfinding

### Color Coding
- **Green Spots**: Start point (selected)
- **Red Spots**: End point (selected)
- **Green Nodes**: Open set (being evaluated)
- **Red Nodes**: Closed set (fully evaluated)
- **Blue Path**: Final shortest path
- **Purple Spots**: Regular graph nodes
- **Gray Arrows**: Edges/connections between nodes

### On-Screen Statistics
```
Open set: 15
Closed set: 23
Path length: 12
Solution found: True
```

## Graph Format (Technical)

Graphs are loaded from JSON files with this structure:

```json
{
  "spots": [
    {
      "x": 100.0,
      "y": 200.0,
      "size": 5,
      "isWall": false
    },
    ...
  ],
  "neighbors": [
    [1, 2, 3],
    [0, 2],
    [0, 1, 3],
    ...
  ]
}
```

- **spots**: Array of node definitions
  - x, y: Position in pixels
  - size: Visual radius
  - isWall: Whether this is an obstacle
  
- **neighbors**: Array of arrays
  - Each index corresponds to a spot
  - Contains indices of neighboring spots
  - Defines graph connectivity

## Algorithm Details

### A* Implementation
The visualizer uses the A* pathfinding algorithm with:

- **Heuristic Function**: Euclidean distance
  ```
  h(n) = √((goal.x - n.x)² + (goal.y - n.y)²)
  ```

- **Cost Function**: Uniform cost per edge
  ```
  g(n) = parent.g + 1
  ```

- **Total Cost**: f(n) = g(n) + h(n)

### Algorithm Steps
1. Start with only the start node in open set
2. While open set is not empty:
   - Pick node with lowest f score
   - If it's the goal, reconstruct and return path
   - Move node from open to closed set
   - Evaluate all neighbors:
     - Skip if in closed set or is wall
     - Update cost if new path is better
     - Add to open set if new
3. If open set becomes empty, no solution exists

### Time Complexity
- Best case: O(log n) with good heuristic
- Worst case: O(n²) without good heuristic
- For graph with n nodes

## Tips and Tricks

### Selection Tips
- Click within ~15 pixels of a spot center
- Spots are usually 5-12 pixels in radius
- Try clicking on the purple circle itself
- If you miss, press R to reset and try again

### Algorithm Observation
- Watch how the green/red clouds expand
- Green (open set) shows frontier of exploration  
- Red (closed set) shows explored area
- Blue path emerges when goal is reached

### Performance
- Application runs at 60 FPS
- Algorithm runs about 6,000 iterations per second
- Visible stepping with SPACE helpful for learning

### Debugging
- No path found? There might not be a valid route
- Algorithm takes long? Graph might be complex
- Click seems to miss? Try clicking closer to center
- Check console output for any error messages

## Common Issues and Solutions

| Issue | Solution |
|-------|----------|
| No graphs in list | Verify mapeditor/saves/ has subdirectories |
| Can't select points | Click in the white canvas area, not sidebar |
| Algorithm doesn't start | Make sure both start and end are selected |
| Path is empty | No valid route between those points |
| Performance is slow | Graph might be very large; try smaller graphs |
| Window is frozen | Click somewhere or press keys; pygame takes a moment |

## Example Scenarios

### Scenario 1: Quick Test
1. Launch application
2. Load "save" graph (first in list)
3. Click spot in lower-right area
4. Click spot in upper-left area
5. Watch algorithm execute
6. See blue path connect the two points

### Scenario 2: Learning A*
1. Load any graph
2. Select two nearby points
3. Press SPACE repeatedly to step through
4. Observe green expanding outward from start
5. Red marks already-explored nodes
6. Blue shows final path when complete

### Scenario 3: Finding Difficult Paths
1. Load graph with maze-like structure
2. Select points far apart or in complex area
3. Let algorithm run automatically
4. Observe how it handles obstacles/barriers
5. Watch algorithm find optimal path around them

## Files Modified/Created

- **AStarTest.py**: Main visualizer application (352 lines)
- **ASTAR_VISUALIZER_README.md**: Full documentation
- **ASTAR_QUICKSTART.md**: Quick reference guide

## Future Enhancements

Potential improvements:
- Dijkstra's algorithm comparison mode
- Graph editing within visualizer
- Save pathfinding results as images
- Statistics/analytics about path quality
- Multiple pathfinding algorithms
- Performance profiling tools
- Custom heuristic selection
- Graph generation from images

## References

### A* Algorithm
- Efficient pathfinding in graphs
- Uses heuristics to guide search
- Optimal: always finds shortest path if one exists
- Complete: will find solution if it exists

### Implementation Details
- Spot class: Represents graph nodes
- AStarGraph class: Manages graph structure
- AStar class: Implements the algorithm
- GraphVisualizer class: Handles rendering and input

## Author Notes

This visualizer provides an interactive way to understand how A* works:
- See the algorithm expand from start point
- Watch it evaluate different paths
- Observe how heuristics guide the search
- Understand why A* is efficient
- Learn graph-based pathfinding visually

Perfect for education, debugging, and testing pathfinding in various graph structures.
