class AStarGraph {
  constructor(start, end) {

    this.__init(start, end);
  }

  restart(startPoint, endPoint){

    this.__init(startPoint, endPoint);

  }

  __init(startPoint, endPoint){

    this.start = startPoint;
    this.end = endPoint;

    //maybe remove walls from start and end
    this.start.isWall = false;
    this.end.isWall = false;
    //

    this.isDone = false;
    this.noSolution = false;

    this.openSet = [];
    this.openSet.push(this.start);
    this.closeSet = [];

    this.path = [];

  }

  __heuristic(a, b){

    var hip = distance(a, b);
    var manhatanDis = Math.abs(a.x - b.x) + Math.abs(a.y - b.y);

    return hip; //this.withDiagonals == true ? hip : manhatanDis;
  }

  update(){

    if(!this.isDone){

      if(this.openSet.length > 0){

        var winner = 0;

        for (var i = 0; i < this.openSet.length; i++) {
          if(this.openSet[i].f < this.openSet[winner].f){
            winner = i;
          }
        }

        var curr = this.openSet[winner];

        if(curr == this.end){

          console.log(this.path);
          

          console.log("Done");
          this.isDone = true;
          // return;
        }

        removeFromArray(this.openSet, curr);
        this.closeSet.push(curr);

        var neighbors = curr.neighbors;

        for (var i = 0; i < neighbors.length; i++) {
          var neighbor = neighbors[i];

          if(!this.closeSet.includes(neighbor) && !neighbor.isWall){
            var tempG = curr.g + 1;// change to be cost not 1 later

            var newPath = false;

            if(this.openSet.includes(neighbor)){
              if(tempG < neighbor.g){
                neighbor.g = tempG;
                newPath = true;
              }
            }else{
              //debug later
              neighbor.g = tempG;
              newPath = true;
              this.openSet.push(neighbor);
            }

            if(newPath){
              neighbor.h = this.__heuristic(neighbor, this.end);
              neighbor.f = neighbor.g + neighbor.h;
              neighbor.previous = curr;
            }
          }
        }

      }else{
        //no solution
        console.log("No solution");
        this.noSolution = true;
        this.isDone = true;
        // return;
      }

      if(!this.noSolution){
        this.path = [];
        var t = curr;
        this.path.push(t);
        while(t.previous){ //TODO fix problem when unidirectional edges are used
          // console.warn("tracing back");
          // console.log("T:", t);
          // console.log("Previous:", t.previous);
          // console.log(this.path);
          
          this.path.push(t.previous);
          t = t.previous;
        }
      }

    }

  }

  draw(showG = false){

    for (var i = 0; i < this.path.length; i++) {
      this.path[i].show("#00bbff", 30, showG);
    }

  }

  debugDraw(showG = false){

    for (var i = 0; i < this.openSet.length; i++) {
      this.openSet[i].show("#00ff00", 30, showG);
    }

    for (var i = 0; i < this.closeSet.length; i++) {
      this.closeSet[i].show("#ff0000", 30, showG);
    }

    this.draw(showG);

  }

}
