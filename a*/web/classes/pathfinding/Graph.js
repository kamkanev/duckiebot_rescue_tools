class Graph {
    constructor(spots = []) {
        this.spots = spots;
    }

    generateRandomGraph(numberodSpots, width, height, wallProbability = 0.2) {
        this.spots = [];
        for(let i = 0; i < numberodSpots; i++) {
            const x = Math.floor(Math.random() * width);
            const y = Math.floor(Math.random() * height);
            const spot = new Spot(x, y);
            if(Math.random() < wallProbability) {
                spot.isWall = true;
            }
            this.spots.push(spot);
        }

        // Connect spots
        for(const spotA of this.spots) {
            for(const spotB of this.spots) {
                if(spotA.x !== spotB.x && spotA.y !== spotB.y) {
                    const dist = Math.sqrt((spotA.x - spotB.x) ** 2 + (spotA.y - spotB.y) ** 2);
                    if(dist < 200) { // arbitrary connection distance
                        if(Math.random() < 0.5) { // random chance to connect
                            this.addEdge(spotA, spotB);
                        }else{
                            this.addEdge(spotA, spotB, false);
                        }
                    }
                }
            }
        }
    }

    addSpot(spot) {
        this.spots.push(spot);
    }

    getSpot(x, y) {
        return this.spots.find(spot => spot.x === x && spot.y === y);
    }

    addEdge(spotA, spotB, bidirectional = true) {
        spotA.addNeighbor(spotB);
        if(bidirectional) {
            spotB.addNeighbor(spotA);
        }
    }

    removeSpot(spot) {
        this.spots = this.spots.filter(s => s.x !== spot.x || s.y !== spot.y);
        for(const s of this.spots) {
            s.neighbors = s.neighbors.filter(n => n.x !== spot.x || n.y !== spot.y);
        }
    }

    removeEdge(spotA, spotB, bidirectional = true) {
        spotA.neighbors = spotA.neighbors.filter(n => n.x !== spotB.x || n.y !== spotB.y);
        if(bidirectional) {
            spotB.neighbors = spotB.neighbors.filter(n => n.x !== spotA.x || n.y !== spotA.y);
        }
    }

    getEdges(spot) {
        return spot.neighbors;
    }

    clearSpots() {
        for(const spot of this.spots) {
            spot.clear();
        }
    }

    getNearestSpot(x, y) {
        let nearestSpot = null;
        let minDist = Infinity;

        for(const spot of this.spots) {
            const dist = Math.sqrt((spot.x - x) ** 2 + (spot.y - y) ** 2);
            if(dist < minDist) {
                minDist = dist;
                nearestSpot = spot;
            }
        }

        return nearestSpot;
    }

    getNearestSpotWithin(x, y, maxDistance) {
        let nearestSpot = null;
        let minDist = Infinity;

        for(const spot of this.spots) {
            const dist = Math.sqrt((spot.x - x) ** 2 + (spot.y - y) ** 2);
            if(dist < minDist && dist <= maxDistance) {
                minDist = dist;
                nearestSpot = spot;
            }
        }

        return nearestSpot;
    }

    draw() {
        for(const spot of this.spots) {
            for(const neighbor of spot.neighbors) {
                context.beginPath();
                // context.moveTo(spot.x, spot.y);
                // context.lineTo(neighbor.x, neighbor.y);
                context.strokeStyle = "#4260e4ff";
                canvas_arrow(context, spot.x, spot.y, neighbor.x, neighbor.y);
                context.stroke();
            }

            if(spot.isWall) {
                spot.show("black");
            } else {
                spot.show("#c01010ff");
            }
            
        }
    }

    clear() {
        this.spots = [];
    }
}