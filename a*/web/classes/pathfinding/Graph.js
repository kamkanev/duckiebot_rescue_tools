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
                        this.addEdge(spotA, spotB);
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

    getEdges(spot) {
        return spot.neighbors;
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

    draw() {
        for(const spot of this.spots) {
            if(spot.isWall) {
                spot.show("black");
            } else {
                spot.show("#c01010ff");
            }
            for(const neighbor of spot.neighbors) {
                context.beginPath();
                context.moveTo(spot.x + 15, spot.y + 15);
                context.lineTo(neighbor.x + 15, neighbor.y + 15);
                context.strokeStyle = "#000000";
                context.stroke();
            }
        }
    }

    clear() {
        this.spots = [];
    }
}