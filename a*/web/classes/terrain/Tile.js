class Tile extends Point {
  constructor(x, y, type = 0, size = 30) {
    super(x, y);
    this.type = type;
    this.isOccupie = false;
    this.size = size;

    if(Math.random() < 0.2){
      this.isOccupie = true;
    }
  }
}

Tile.types = TERRAIN_TYPES;
