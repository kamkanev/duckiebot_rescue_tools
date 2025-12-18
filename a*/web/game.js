// Creating variables
var t = new Terrain(false);

var r = new Rabbit(7, 10, t);

var cursor = new Point();

function update() {
    cursor.x = cursor.x+(mouseX-cursor.x)-5;
    cursor.y = cursor.y+(mouseY-cursor.y)-5;

    r.update();

}

function draw() {
    // This is how you draw a rectangle
     t.draw();

    r.draw(true);

    context.fillStyle = "pink";
    context.fillRect(cursor.x, cursor.y, 10, 10);

};

function keyup(key) {
    // Show the pressed keycode in the console
    console.log("Pressed", key);

    if(key == 66){//b
      t.changeBuildMode();
    }
};

function keydown(key) {
    // Show the pressed keycode in the console
    console.log("PressedDown", key);

    if(isKeyPressed[90]){//z
      if(t.zoom <= 40){
        t.changeZoom(t.zoom + 5);
      }
    }

    if(isKeyPressed[88]){//x
      if(t.zoom >= 15){
        t.changeZoom(t.zoom - 5);
      }
    }


    if(isKeyPressed[37]){//left
      t.move(t.deltaPoint.x-1, t.deltaPoint.y);
    }

    if(isKeyPressed[39]){//right
      t.move(t.deltaPoint.x+1, t.deltaPoint.y);
    }

    if(isKeyPressed[38]){//up
      t.move(t.deltaPoint.x, t.deltaPoint.y-1);
    }

    if(isKeyPressed[40]){//down
      t.move(t.deltaPoint.x, t.deltaPoint.y+1);
    }
};

function mouseup() {
    // Show coordinates of mouse on click
    console.log("Mouse clicked at", mouseX, mouseY);

    //clicked = new Point(mouseX, mouseY);

    var trMouse = new Point(Math.floor(mouseX/t.zoom), Math.floor(mouseY/t.zoom))
    r.setGoal(trMouse);
    console.log(r);
};
