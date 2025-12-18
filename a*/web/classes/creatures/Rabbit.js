class Rabbit {
  constructor(x = 0, y = 0, terrain, size = 30) {
    this.box = new Point(x, y);
    this.terrain = terrain;
    this.screenBox = translatePointToScreen(this.box, terrain.zoom, terrain.deltaPoint);
    this.a = undefined;//new AStar(t, t.map[0][0], t.map[25][18], true);
    this.goal = undefined;
    this.path = undefined;
    this.r = 2;

    this.terrain = terrain;
  }

  setGoal(goal){
    this.goal = goal;
    this.a = new AStar(this.terrain, this.terrain.map[this.box.x][this.box.y], this.terrain.map[goal.x][goal.y], false);
    this.to = this.box;
    this.path = undefined;
    // console.log(this.to);
  }

  update(){

    // console.log(this.to);

    if(this.a != undefined){
      if(this.to != undefined){

        var dx = this.to.x*this.a.zoom - this.screenBox.x;
        var dy = this.to.y*this.a.zoom - this.screenBox.y;
        var dist = Math.sqrt(dx*dx, dy*dy);
        var speed = 2;// TODO: set from constructor later
        var angle = Math.atan2(dy, dx);

        this.screenBox.x += speed * Math.cos(angle);
        this.screenBox.y += speed * Math.sin(angle);

        this.box = translateScreenPointToArray(this.screenBox, this.terrain.zoom);

        // console.log(dist);

        if(Math.round(this.screenBox.x/this.a.zoom) == this.to.x && Math.round(this.screenBox.y/this.a.zoom) == this.to.y){
          this.to = undefined;
        }

      }

      this.a.update();

      if(this.a.isDone && !this.a.noSolution && this.path == undefined){
        this.path = this.a.path.slice(0);
      }

      if(this.to == undefined && this.path != undefined && this.path.length > 0){
        this.to = this.path.pop();
      }

      if(this.to == undefined && this.path != undefined && this.path.length == 0){
        this.a = undefined;
        this.to = undefined;
        this.path = undefined;
        this.box = this.goal;
        this.screenBox = translatePointToScreen(this.box, this.terrain.zoom, this.terrain.deltaPoint);
        this.goal = undefined;
        return;
      }
    }else{


      this.screenBox = translatePointToScreen(this.box, this.terrain.zoom, this.terrain.deltaPoint);
    }

  }

  draw(debug = false){

    if(debug){

      if(this.a != undefined){
        this.a.debugDraw(true);
      }

      for(var x = this.box.x - this.r; x <= this.box.x + this.r; x++){
        for(var y = this.box.y - this.r; y<= this.box.y + this.r; y++){
      
      
          context.globalAlpha = 0.5;
          context.fillStyle = "#fff000";
          var p = translatePointToScreen(new Point(x, y), this.terrain.zoom, 0);
          context.fillRect(p.x, p.y, this.terrain.zoom -1, this.terrain.zoom-1);
      
        }
      }

    }
    context.globalAlpha = 1;
    context.fillStyle = "red";
    context.fillRect(this.screenBox.x, this.screenBox.y, this.terrain.zoom -1, this.terrain.zoom-1);


  }
}
