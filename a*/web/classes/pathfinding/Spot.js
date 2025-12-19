class Spot extends Point{
  constructor(x, y, isWall = false) {

    super(x, y);
    this.f = 0; // over all cost g + h
    this.g = 0; // cost to this spot
    this.h = 0; // heuristics
    this.isWall = isWall;

    this.previous = undefined;
    this.neighbors = [];

    // if(Math.random() < 0.4){
      // this.isWall = true;
    // }

  }

  /**
   * Adds neighbors to the current spot.
   * @param {Array} grid a grid with all elements
   * @param {Boolean} withDiagonals to include or exclude the diaogonal neighbors
   */
  addNeighbors(grid, withDiagonals = false){

    var i = this.x
    var j = this.y

    if(i < grid.length - 1)
      this.neighbors.push(grid[i+1][j])
    if(i > 0)
      this.neighbors.push(grid[i-1][j])
    if(j < grid[0].length - 1)
      this.neighbors.push(grid[i][j+1])
    if(j > 0)
      this.neighbors.push(grid[i][j-1])
    if(withDiagonals){
      if(i > 0 && j > 0)
        this.neighbors.push(grid[i-1][j-1])
      if(i < grid.length - 1 && j > 0)
        this.neighbors.push(grid[i+1][j-1])
      if(i < grid.length - 1 && j < grid[0].length - 1)
        this.neighbors.push(grid[i+1][j+1])
      if(i > 0 && j < grid[0].length - 1)
        this.neighbors.push(grid[i-1][j+1])
    }

  }

  addNeighbor(spot){
    this.neighbors.push(spot);
  }

  show(color, size = 30, showG = false){

    context.fillStyle = color;
    context.fillRect(this.x, this.y, size, size);

    if(!showG){
      //draw the coords
      context.fillStyle = "black";
      context.font = "9px Arial";
      context.fillText(`${this.x},${this.y}`, this.x + size/4, this.y + size/2);
    }else{
      //draw the coords
      context.fillStyle = "black";
      context.font = "10px Arial";
      context.fillText(`${this.g}`, this.x + size/2, this.y + size/2);
    }

  }
}
