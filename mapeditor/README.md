Map Editor
==========

A small Pygame-based map editor used in this workspace.

Quick start
-----------

- Install dependencies (requires Python 3 and Pygame):

```bash
pip install pygame
```

- Run the editor from the project root:

```bash
python3 mapeditor/map_editor.py
```

Controls
--------

- Left click on the grid: place the currently selected tile.
- Right click on the grid: remove tile.
- Use the tile buttons on the right to choose a tile.
- Q / E: rotate tile.

Save / Load
-----------

- Save: click the Save button.
  
  - If a map `name` is already set, the editor asks for confirmation before overwriting.
  - If no name is set, an in-game text input appears to enter a file name.
  - Saves are stored under `mapeditor/saves/<name>/<name>.csv`.

- Load: click the Load button.
  
  - An in-game list of saved maps appears.
  - Each save in the list has a red 'X' on the right — click the 'X' to delete that save (you will be asked to confirm).

Notes & safety
---------------

- Deletions remove the entire save folder permanently; consider backups before deleting.
- File I/O uses atomic replace for saves to reduce corruption risk.

If you want a different delete behavior (move to a trash folder instead of permanent delete), tell me and I can add it.

TODO
----

- [ ] Add nodes in each tile to a graph can be formed
- [x] Make the nodes rotate according to the tiles
- [x] Add a graph edit mode to see the nodes/spots and connected them correctly (edit the graph)
- [ ] Save it in a JSON file in the map folder
- [ ] Add autoconnection tool maybe
