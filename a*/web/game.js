// Creating variables
// var t = new Terrain(false);

// var r = new Rabbit(7, 10, t);
var g = new Graph();
g.generateRandomGraph(12, 800, 600, 0.1);

var aStar = undefined;
var startSpot = undefined;
var endSpot = undefined;

var isFirstRun = true;
var buildMode = false;
var editSpot = true;
var removingElements = false;
var bidirectionalEdges = true;


var map = new Image();
map.src = 'map.jpg';

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

    context.drawImage(map, -260, -50, 1700, 650);

    context.strokeStyle = "black";
    context.lineWidth = 1;
    context.strokeRect(0, 0, canvas.width, canvas.height);

    context.lineWidth = 1;
    context.strokeStyle = "black";
    context.strokeRect(canvas.width - 100, canvas.height - 100, 100, 100);

    if(!buildMode){
        context.fillStyle = "green";
        context.beginPath();
        context.strokeStyle = "black";
        context.lineWidth = 2;
        context.moveTo(canvas.width - 80, canvas.height - 50);
        context.lineTo(canvas.width - 50, canvas.height - 80);
        context.lineTo(canvas.width - 20, canvas.height - 20);
        context.stroke();
        context.beginPath();
        context.arc(canvas.width - 20, canvas.height - 20, 10, 0, 2 * Math.PI);
        context.fill();
        context.fillStyle = "red";
        context.beginPath();
        context.arc(canvas.width - 80, canvas.height - 50, 10, 0, 2 * Math.PI);
        context.fill();
        context.beginPath();
        context.arc(canvas.width - 50, canvas.height - 80, 10, 0, 2 * Math.PI);
        context.fill();
        context.closePath();

        context.fillStyle = "black";
        context.font = "20px Arial";
        context.fillText(`Pathfinding`, canvas.width - 100, canvas.height - 110);

        context.fillStyle = "black";
        context.font = "20px Arial";
        context.fillText(`Click near one point and another to generate shortest path`, 120, canvas.height - 30);
        context.fillText(`Press \'B\' to toggle build mode. Press \'C\' to clear all. Press \'R\' to generate a random graph`, 100, canvas.height - 7);

    }else{
        context.fillStyle = "#585858ff";
        context.beginPath();
        
        context.strokeStyle = "black";
        context.lineWidth = 2;
        context.setLineDash([5, 5]);
        context.moveTo(canvas.width - 80, canvas.height - 50);
        context.lineTo(canvas.width - 50, canvas.height - 80);
        context.lineTo(canvas.width - 20, canvas.height - 20);
        context.stroke();
        context.beginPath();
        context.arc(canvas.width - 20, canvas.height - 20, 10, 0, 2 * Math.PI);
        context.fill();
        context.beginPath();
        context.arc(canvas.width - 80, canvas.height - 50, 10, 0, 2 * Math.PI);
        context.fill();
        context.beginPath();
        context.arc(canvas.width - 50, canvas.height - 80, 10, 0, 2 * Math.PI);
        context.fill();
        context.closePath();
        context.setLineDash([]);

        context.fillStyle = "black";
        context.font = "20px Arial";
        context.fillText(`Build mode`, canvas.width - 100, canvas.height - 110);

        context.fillStyle = "black";
        context.font = "17px Arial";
        context.fillText(`Press \'B\' to toggle pathfinding mode. Press 1 to edit spot. Press 2 to add edges. Press 3 to toggle directional edges`, 30, canvas.height - 7);

        if(editSpot){
            context.fillStyle = "black";
            context.font = "22px Arial";
            context.fillText(`N`, canvas.width - 60, canvas.height - 10);

            context.fillStyle = "black";
            context.font = "20px Arial";
            context.fillText(`Click to create spots and click on them to toggle walls`, 120, canvas.height - 30);
        }else{
            context.fillStyle = "black";
            context.font = "22px Arial";
            context.fillText(`E`, canvas.width - 60, canvas.height - 10);

            context.fillStyle = "black";
            context.font = "20px Arial";
            context.fillText(`Click on a node and another to add an edge`, 120, canvas.height - 30);
        }

        if(!bidirectionalEdges){
            context.beginPath();
            context.lineWidth = 2;
            context.strokeStyle = "black";
            canvas_arrow(context, canvas.width - 70, canvas.height - 40, canvas.width - 40, canvas.height - 40);
            context.stroke();
        }

    }

    

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

    if(key == 66){//b
      buildMode = !buildMode;
      isFirstRun = true;
      console.log("Build mode:", buildMode);
    }

    if(key == 49){//1
        editSpot = true;
    }

    if(key == 50){//2
        editSpot = false;
    }

    if(key == 51){//3
        bidirectionalEdges = !bidirectionalEdges;
        console.log("Bidirectional edges:", bidirectionalEdges);
    }

    if(key == 67){//c
        g.clear();
        console.log("Cleared all spots and edges.");
    }

    if(key == 82){//r
        g.clear();
        g.generateRandomGraph(12, 800, 600, 0.1);
        console.log("Generated random graph.");
    }
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

    if(!buildMode){
        //clicked = new Point(mouseX, mouseY);
        var clickedSpot = g.getNearestSpot(Math.floor(mouseX), Math.floor(mouseY), false);
        console.log("Nearest spot:", clickedSpot);

        if (isFirstRun) {
            startSpot = clickedSpot;
            isFirstRun = false;
            console.log("Start spot set to:", startSpot);
        } else {
            endSpot = clickedSpot;
            console.log("End spot set to:", endSpot);

            g.clearSpots(); // Clear previous A* data

            aStar = new AStarGraph(startSpot, endSpot);
            
            
            isFirstRun = true; // Reset for next run
        }
    }else{
        if(editSpot){
            var nearestSpot = g.getNearestSpotWithin(Math.floor(mouseX), Math.floor(mouseY), 15);
            if(nearestSpot == null){
                g.addSpot(new Spot(Math.floor(mouseX), Math.floor(mouseY)));
                console.log("Added new spot at:", Math.floor(mouseX), Math.floor(mouseY));
            }else{
                nearestSpot.isWall = !nearestSpot.isWall;
            }
        }else{
            console.log();
            
            if(isFirstRun){
                var nearestSpot = g.getNearestSpot(Math.floor(mouseX), Math.floor(mouseY));
                console.log(nearestSpot);
                
                if(nearestSpot == null){
                    console.log("No spot nearby to connect from.");
                }else{
                    startSpot = nearestSpot;
                    isFirstRun = false;
                    console.log("Selected first spot for edge:", startSpot);
                }
            }else{
                var nearestSpot = g.getNearestSpot(Math.floor(mouseX), Math.floor(mouseY));
                if(nearestSpot == null){
                    console.log("No spot nearby to connect to.");
                }else{
                    g.addEdge(startSpot, nearestSpot, bidirectionalEdges); //TODO: bidirectional option
                    console.log("Connected", startSpot, "to", nearestSpot);
                    isFirstRun = true;
                }
            }
        }
    }


    // var trMouse = new Point(Math.floor(mouseX/t.zoom), Math.floor(mouseY/t.zoom))
    // r.setGoal(trMouse);
    // console.log(r);
};
