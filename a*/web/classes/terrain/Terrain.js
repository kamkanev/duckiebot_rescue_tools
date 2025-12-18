class Terrain {
  constructor(isRandom = false, isInverted = false) {

    this.map = [];
    this.zoom = 30;
    this.isInBuildMode = false;
    this.isInverted = isInverted;
    this.isRandom = isRandom;
    this.deltaPoint = new Point();

    if(!isRandom){
      this.__init();
    }else{
      this.__createEmptyMap();
    }

  }

  init(isRandom = false, type = 0){
    if(!isRandom){
      this.__init();
    }else{
      this.__createEmptyMap(type);
    }
  }

  __createEmptyMap(type = 0){

    this.__generateNewEmptyTerrain(type);

  }

  __init(){

    //Generate random noiseScale for the terrain
    this.noiseScale = {
      x: (Math.random() < .90) ? Number((Math.random()/10).toFixed(3)) : Number((Math.random()/5).toFixed(3)),
      y: (Math.random() < .90) ? Number((Math.random()/10).toFixed(3)) : Number((Math.random()/5).toFixed(3)),
      z: Number(((Math.random()*10+1)/2).toFixed(2))
    }

    this.__generateNewTerrain();
  }

  // Renders the map
  __generateNewTerrain(){
    this.map = [];
    for (let x = 0; x < canvas.width; x++) {
      this.map[x] = [];
      for (var y = 0; y < canvas.height; y++) {
        this.map[x][y] = new Tile(x, y, 0, this.zoom);
      }
    }
  }

  __generateNewEmptyTerrain(type = 0){
    this.map = [];
    for (let x = 0; x < canvas.width; x++) {
      this.map[x] = [];
      for (var y = 0; y < canvas.height; y++) {
        this.map[x][y] = new Tile(x, y, 0, this.zoom);
      }
    }
  }

  // Is used to move the map coords when zoomed
  move(x = 0, y = 0){
    if(Math.round(this.map.length/this.zoom) + x < canvas.width){
      if(Math.round(this.map[0].length/this.zoom) + y < canvas.height){
        if(x >= 0 && y >= 0){
          this.deltaPoint = new Point(x, y);
        }
      }
    }
  }

  //Can change the zoom - to see more or less from the map
  changeZoom(z = 30){
    if(z >= 1 && z <= 200){
      this.zoom = z;
    }else if(z < 1){
      this.zoom = 1;
    }else if(z > 200){
      this.zoom = 200;
    }
  }

  //toggles build mode

  setBuildMode(b = true){
    this.isInBuildMode = b;
  }

  changeBuildMode(){
    this.isInBuildMode = !this.isInBuildMode;
  }

  // Draws the map using all properties from above
  draw(){

    var size = this.zoom;

    if(this.isInBuildMode){
      size--;
    }

    for (var x = this.deltaPoint.x; x < Math.round(this.map.length/this.zoom)+this.deltaPoint.x; x++) {

      if(this.map[x] != undefined){

        for (var y = this.deltaPoint.y; y < Math.round(this.map[x].length/this.zoom)+this.deltaPoint.y; y++) {

          if(this.map[x][y] != undefined){

            var c = this.map[x][y].isOccupie;
            // console.log(c);
            if(c){
              context.fillStyle = "#000000";
            }else{
              context.fillStyle = "#ffffff";
            }

            context.fillRect((x - this.deltaPoint.x)*this.zoom, (y - this.deltaPoint.y)*this.zoom, size, size);

          }

        }
      }

    }

  }
}

Terrain.types = TERRAIN_TYPES;
