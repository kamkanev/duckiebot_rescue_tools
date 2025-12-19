// Creating variables
// var t = new Terrain(false);

// var r = new Rabbit(7, 10, t);
var g = new Graph();
g.generateRandomGraph(12, 800, 600, 0.1);

var aStar = undefined;
var startSpot = undefined;
var endSpot = undefined;

var isFirstRun = true;

var cursor = new Point();

function update() {
    cursor.x = cursor.x+(mouseX-cursor.x)-5;
    cursor.y = cursor.y+(mouseY-cursor.y)-5;

    if (aStar != undefined) {
        aStar.update();
    }

    // r.update();

}

function draw() {
    // This is how you draw a rectangle
    //  t.draw();

    // r.draw(true);

    context.fillStyle = "black";
    context.font = "20px Arial";
    context.fillText(`Click near one point and another to generate shortest path`, 120, 50);

    g.draw();

    if (aStar != undefined) {
        aStar.debugDraw(true);
    }

    context.fillStyle = "pink";
    context.fillRect(cursor.x, cursor.y, 10, 10);

};

function keyup(key) {
    // Show the pressed keycode in the console
    console.log("Pressed", key);

    // if(key == 66){//b
    //   t.changeBuildMode();
    // }
};

function keydown(key) {
    // Show the pressed keycode in the console
    console.log("PressedDown", key);

    // if(isKeyPressed[90]){//z
    //   if(t.zoom <= 40){
    //     t.changeZoom(t.zoom + 5);
    //   }
    // }

    // if(isKeyPressed[88]){//x
    //   if(t.zoom >= 15){
    //     t.changeZoom(t.zoom - 5);
    //   }
    // }


    // if(isKeyPressed[37]){//left
    //   t.move(t.deltaPoint.x-1, t.deltaPoint.y);
    // }

    // if(isKeyPressed[39]){//right
    //   t.move(t.deltaPoint.x+1, t.deltaPoint.y);
    // }

    // if(isKeyPressed[38]){//up
    //   t.move(t.deltaPoint.x, t.deltaPoint.y-1);
    // }

    // if(isKeyPressed[40]){//down
    //   t.move(t.deltaPoint.x, t.deltaPoint.y+1);
    // }
};

function mouseup() {
    // Show coordinates of mouse on click
    console.log("Mouse clicked at", mouseX, mouseY);

    //clicked = new Point(mouseX, mouseY);
    var clickedSpot = g.getNearestSpot(Math.floor(mouseX), Math.floor(mouseY));
    console.log("Nearest spot:", clickedSpot);

    if (isFirstRun) {
        startSpot = clickedSpot;
        isFirstRun = false;
        console.log("Start spot set to:", startSpot);
    } else {
        endSpot = clickedSpot;
        console.log("End spot set to:", endSpot);

        aStar = new AStarGraph(startSpot, endSpot);
        
        
        isFirstRun = true; // Reset for next run
    }


    // var trMouse = new Point(Math.floor(mouseX/t.zoom), Math.floor(mouseY/t.zoom))
    // r.setGoal(trMouse);
    // console.log(r);
};
