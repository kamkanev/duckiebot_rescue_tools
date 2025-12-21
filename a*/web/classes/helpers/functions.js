function removeFromArray(arr, el) {
  for (var i = arr.length-1; i >= 0; i--) {
    if(arr[i] == el){
      arr.splice(i, 1);
    }
  }
}

function distance(a, b) {
  var side1 = b.x - a.x;
  var side2 = b.y - a.y;

  return Math.sqrt(side1*side1 + side2*side2);
}

function translatePointToScreen(point, size, deltaPoint){

  return new Point( point.x*size - deltaPoint.x*size, point.y*size - deltaPoint.y*size);

}

function translateScreenPointToArray(point, size){

  return new Point( Math.round(point.x/size), Math.round(point.y/size) );

}

function canvas_arrow(context, fromx, fromy, tox, toy) {
  var headlen = 10; // length of head in pixels
  var dx = tox - fromx;
  var dy = toy - fromy;
  var angle = Math.atan2(dy, dx);
  context.moveTo(fromx, fromy);
  context.lineTo(tox, toy);
  context.lineTo(tox - headlen * Math.cos(angle - Math.PI / 6), toy - headlen * Math.sin(angle - Math.PI / 6));
  context.moveTo(tox, toy);
  context.lineTo(tox - headlen * Math.cos(angle + Math.PI / 6), toy - headlen * Math.sin(angle + Math.PI / 6));
}
