import cv2
import numpy as np
import math

img = cv2.imread("assets/try5.jpg", 1)
original = img.copy()
draw_weights = False 
dotRadius = 6

#cropped image     
# Convert to grayscale
gray = cv2.cvtColor(original, cv2.COLOR_BGR2GRAY)

# Threshold to get the dark area (road/map)
_, mask = cv2.threshold(gray, 100, 255, cv2.THRESH_BINARY_INV)

# Clean up the mask with morphological operations
kernel = np.ones((5, 5), np.uint8)
mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

# Find contours
contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Get the largest contour (main map area)
largest_contour = max(contours, key=cv2.contourArea)

# Get all points from the contour
points = largest_contour.reshape(-1, 2)

# Find min and max x, y values
x_min = int(np.min(points[:, 0]))
x_max = int(np.max(points[:, 0]))
y_min = int(np.min(points[:, 1]))
y_max = int(np.max(points[:, 1]))

# Create bounding rectangle dictionary
bounding_rect = {
    'x_min': x_min,
    'x_max': x_max,
    'y_min': y_min,
    'y_max': y_max,
    'width': x_max - x_min,
    'height': y_max - y_min
}

# Crop the image to the bounding rectangle
cropped = original[y_min:y_max, x_min:x_max]

#red mask
img = cv2.cvtColor(cropped, cv2.COLOR_BGR2HSV)

red_lower = np.array([60, 130, 20]) 
red_upper = np.array([180, 255, 255]) 
red_mask = cv2.inRange(img, red_lower, red_upper) 


resultYellow = cropped.copy()
resultWhite = cropped.copy()


lower_White = np.array([0, 0, 220])
upper_White = np.array([180, 10, 255])
# lower_White = np.array([0, 50, 20])
# upper_White = np.array([10, 255, 255])
white_mask = cv2.inRange(img, lower_White, upper_White) 
resultWhite = cv2.bitwise_and(resultWhite, resultWhite, mask=white_mask) 

lower_yellow = np.array([20, 65, 100])
upper_yellow = np.array([31, 255, 255])
yellow_mask = cv2.inRange(img, lower_yellow, upper_yellow) 
resultYellow = cv2.bitwise_and(resultYellow, resultYellow, mask=yellow_mask) 

mixedMask= cv2.bitwise_or(white_mask, yellow_mask) 
resultBorder = cv2.bitwise_and(img, img, mask=mixedMask) 

result = cropped.copy()

resultRed = cropped.copy()
resultRed = cv2.bitwise_and(resultRed, resultRed, mask=red_mask) 
# Convert to grayscale
gray = cv2.cvtColor(resultRed, cv2.COLOR_BGR2GRAY)

# Create binary mask (anything not black is a block)
_, mask = cv2.threshold(gray, 10, 255, cv2.THRESH_BINARY)

# Find contours (each contour represents a red block)
contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

print(f"Found {len(contours)} red blocks")

# Filter out small dots - only keep actual blocks
MIN_BLOCK_AREA = 30  # Minimum area in pixels to be considered a block
MIN_BLOCK_WIDTH = 5  # Minimum width in pixels
MIN_BLOCK_HEIGHT = 5  # Minimum height in pixels


redBlocks = []
for contour in contours:
    area = cv2.contourArea(contour)
    x, y, w, h = cv2.boundingRect(contour)
    
    # Check if it's a real block (not a small dot)
    if area >= MIN_BLOCK_AREA and w >= MIN_BLOCK_WIDTH and h >= MIN_BLOCK_HEIGHT:
        redBlocks.append(contour)


print(f"Filtered to {len(redBlocks)} actual blocks (removed {len(contours) - len(redBlocks)} small dots)")


# Separate into horizontal and vertical blocks
horizontalBlocks = []
verticalBlocks = []

for contour in redBlocks:
    x, y, w, h = cv2.boundingRect(contour)
    
    if w > h:
        horizontalBlocks.append({'contour': contour, 'x': x, 'y': y, 'w': w, 'h': h})
    else:
        verticalBlocks.append({'contour': contour, 'x': x, 'y': y, 'w': w, 'h': h})

print(f"\nHorizontal blocks: {len(horizontalBlocks)}")
print(f"Vertical blocks: {len(verticalBlocks)}")

#------------------------------------------------------------
grayYel = cv2.cvtColor(resultYellow, cv2.COLOR_BGR2GRAY)
_, maskYel = cv2.threshold(grayYel, 10, 255, cv2.THRESH_BINARY)
contoursYel, _ = cv2.findContours(maskYel, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

MIN_BLOCK_AREAY = 4  # Minimum area in pixels to be considered a block
MIN_BLOCK_WIDTHY = 1 # Minimum width in pixels
MIN_BLOCK_HEIGHTY = 1

yellowBlocks = []
yellowBlockCoord = []
turningPoints = []
turningPointNodes = []

def magnitude(v1, v2):
    v1Mag = math.sqrt((v1[0]**2) + (v1[1]**2))
    v2Mag = math.sqrt((v2[0]**2) + (v2[1]**2))
    return v1Mag * v2Mag

def angle(v1,v2):

    dot = np.dot(v1, v2)
    mag = magnitude(v1, v2)    

    return math.degrees(math.acos(dot / mag))

def is_turning(p1, p2, small):
    # BIG: x was 32, y was 20 for search bounds || this to be used in another function not this
    # SMALL: x is 10, 10
    # displacement of 5 in both x AND y
    if(small == True):
        displacementX = 7
        displacementY = 7
        # cv2.rectangle(result,(p1['x'] - displacementX, p1['y'] - displacementY), (p1['x'] + displacementX, p1['y'] + displacementY),(0,255,0),-1)
        out_X = (p2['x'] < p1['x'] - displacementX) or (p2['x'] > p1['x'] + displacementX)
        out_y = (p2['y'] < p1['y'] - displacementY) or (p2['y'] > p1['y'] + displacementY)
        return (out_X and out_y)
    else: 
        displacementX = 13
        displacementY = 12
        # cv2.rectangle(result,(p1['x'] - displacementX, p1['y'] - displacementY), (p1['x'] + displacementX, p1['y'] + displacementY),(0,255,255),-1)
        out_X = (p2['x'] < p1['x'] - displacementX) or (p2['x'] > p1['x'] + displacementX)
        out_y = (p2['y'] < p1['y'] - displacementY) or (p2['y'] > p1['y'] + displacementY)
        return (out_X and out_y)


def searchBound(coordinates, p):
    #check small first
    SMALL_BOUND = 18
    BIG_BOUND_X = 25
    BIG_BOUND_Y = 25
    small = None
    found = []
    for i, point in enumerate(coordinates):
        if ((p['x'] - SMALL_BOUND) <= point['x'] <= (p['x'] + SMALL_BOUND)) and ((p['y'] - SMALL_BOUND) <= point['y'] <= (p['y'] + SMALL_BOUND)):
            found.append(point)
            small = True
            #cv2.rectangle(result,(p['x'] - SMALL_BOUND, p['y'] - SMALL_BOUND), ((p['x'] + SMALL_BOUND), p['y'] + SMALL_BOUND),(0,0,255),-1)


        elif((p['x'] - BIG_BOUND_X) <= point['x'] <= (p['x'] + BIG_BOUND_X)) and ((p['y'] - BIG_BOUND_Y) <= point['y'] <= (p['y'] + BIG_BOUND_Y)):
            found.append(point)
            small = False  
            #cv2.rectangle(result,(p['x'] - BIG_BOUND_X, p['y'] - BIG_BOUND_Y), ((p['x'] + BIG_BOUND_X), p['y'] + BIG_BOUND_Y),(0,0,255),-1)


    return found, small

# def curve_midpoint(points):
#     # Normalize input format
#     pts = [(p['x'], p['y']) if isinstance(p, dict) else (p[0], p[1]) for p in points]

#     # Compute segment lengths
#     lengths = []
#     for i in range(len(pts) - 1):
#         dx = pts[i+1][0] - pts[i][0]
#         dy = pts[i+1][1] - pts[i][1]
#         lengths.append(math.hypot(dx, dy))

#     total_length = sum(lengths)
#     half_length = total_length / 2

#     # Walk along the curve
#     dist = 0
#     for i, seg_len in enumerate(lengths):
#         if dist + seg_len >= half_length:
#             ratio = (half_length - dist) / seg_len
#             x = pts[i][0] + ratio * (pts[i+1][0] - pts[i][0])
#             y = pts[i][1] + ratio * (pts[i+1][1] - pts[i][1])
#             return [x, y]
#         dist += seg_len

#     return list(pts[-1])

def projected_midpoint(points):
    pts = [(p['x'], p['y']) if isinstance(p, dict) else p for p in points]

    xs, ys = zip(*pts)
    cx, cy = (min(xs) + max(xs)) / 2, (min(ys) + max(ys)) / 2

    best_dist = float('inf')
    best_pt = None

    for i in range(len(pts) - 1):
        x1, y1 = pts[i]
        x2, y2 = pts[i+1]

        dx, dy = x2 - x1, y2 - y1
        if dx == dy == 0:
            continue

        t = ((cx - x1)*dx + (cy - y1)*dy) / (dx*dx + dy*dy)
        t = max(0, min(1, t))

        px = x1 + t * dx
        py = y1 + t * dy

        d = math.hypot(px - cx, py - cy)
        if d < best_dist:
            best_dist = d
            best_pt = (px, py)

    if best_pt is None:
        return None

    return list(best_pt)


foundForTurns = set()

def getPoints(turningPoints, p, start_bound=25, step=10, max_bound=500):
    bound = start_bound
    prev_count = -1
    print(len(turningPoints))
    found = set()
    while bound <= max_bound:
        # cv2.rectangle(result,(p['x'] - bound,p['y'] - bound),(p['x'] + bound,p['y'] + bound),(0,0,255),1)
        for point in turningPoints:
            x = int(point['x'])
            y = int(point['y'])

            if (p['x'] - bound <= x <= p['x'] + bound and
                p['y'] - bound <= y <= p['y'] + bound and 
                (x,y) not in foundForTurns and
                (x,y not in found)):
                found.add((x,y))

        # stop when expansion adds nothing new
        if len(found) == prev_count:
            break

        prev_count = len(found)
        bound += step

    foundForTurns.update(found)
    print(f"Length of found: {len(foundForTurns)}")
    return found


for contour in contoursYel:
    area = cv2.contourArea(contour)
    x, y, w, h = cv2.boundingRect(contour)
    
    # Check if it's a real block (not a small dot)
    if area >= MIN_BLOCK_AREAY and w >= MIN_BLOCK_WIDTHY and h >= MIN_BLOCK_HEIGHTY:
        yellowBlocks.append(contour)
        yellowBlockCoord.append({
            'x': x,
            'y': y
        })
        

for i, point in enumerate(yellowBlockCoord):
    found = []
    small = None
    found, small = searchBound(yellowBlockCoord, point)
    for pointFound in found:
        if (is_turning(point, pointFound, small)):
            if point not in turningPoints:
                    turningPoints.append(point)

#visualize (x,y) or blocks
for block in yellowBlockCoord:
    cv2.circle(result,(block['x'],block['y']),1,(0,0,0),-1)

visited = set()
midTurningPoints = []

for point in turningPoints:
    point_tuple = (point['x'], point['y'])
    if point_tuple in visited:
        continue
    
    gotPoints = getPoints(turningPoints, point, 12,8)

    if(gotPoints):
        midTurningPoints.append(projected_midpoint(gotPoints)) 

    for p in gotPoints:
        visited.add(p)

def drawPoints(drawDown, x, y, img_height, top, left, right, down, checkPrevious):
    # Default bounds
    print("Left, Right, Top, Down")
    print(f"{lastLeft,lastRight,lastTop,lastDown}")
    print(f"{left, right, top, down}")

    x_bound = 14
    y_bound = 14
    flip = False
    global flipped 

    firstNode_x = x - x_bound 
    lastNode_x = x + x_bound 

    if drawDown:
        firstNode_y = y - y_bound 
        lastNode_y = y + y_bound
    else:
        firstNode_y = y + y_bound
        lastNode_y = y - y_bound
    
    if checkPrevious:
        #cv2.circle(result,(x,y),dotRadius + 20,(0,0,255),1)
        if(flipped == True):
            print(f"Checking flip conditions... {flipped}")
            if(top and left) and (lastRight and lastDown):
                flipped = False
            elif (right and down) and (lastTop and lastLeft):
                flipped = False
            elif (top and right) and (lastDown and lastLeft):
                flipped = False
            elif (down and left) and (lastTop and lastRight):
                flipped = False
        elif flipped == False:
            print(f"Checking flip conditions... {flipped}")
            if(top and left) and (lastRight and lastDown):
                flip = True
                flipped = True
            elif (right and down) and (lastTop and lastLeft):
                flip = True
                flipped = True
            elif (top and right) and (lastDown and lastLeft):
                flip = True
                flipped = True
            elif (down and left) and (lastTop and lastRight):
                flip = True 
                flipped = True
    print(f"flip: {flip}")
    type = ('inner', 'outer')
    color = ((0,255,0),(0,255,255))

    if not top and not left or (top and not left and right):
       #first is green
        i = 0
        if(flip):
            i -= 1
        cv2.circle(result, (firstNode_x, firstNode_y), dotRadius, color[i], -1)
        turningPointNodes.append({ # inner
        'point': (int(firstNode_x), int(firstNode_y)),
        'to': (0,0),
        'type': type[i],
        'from': (0,0),
        'road': 0,
        'top': top,
        'left': left,
        'right': right,
        'down': down
        })
        i += 1
        cv2.circle(result, (lastNode_x, lastNode_y), dotRadius, color[i], -1)
        turningPointNodes.append({ #outer
        'point': (int(lastNode_x), int(lastNode_y)),
        'to': (0,0),
        'type': type[i],
        'from': (0,0),
        'road': 0,
        'top': top,
        'left': left,
        'right': right,
        'down': down
        })
    else:
        i = -1
        if (flip):
            i -= 1
        #first is yellow
        cv2.circle(result, (firstNode_x, firstNode_y), dotRadius, color[i], -1)
        turningPointNodes.append({ #outer
        'point': (int(firstNode_x), int(firstNode_y)),
        'to': (0,0),
        'type': type[i],
        'from': (0,0),
        'road': 0,
        'top': top,
        'left': left,
        'right': right,
        'down': down
        })
        i += 1
        cv2.circle(result, (lastNode_x, lastNode_y), dotRadius, color[i], -1)
        turningPointNodes.append({ # inner
        'point': (int(lastNode_x), int(lastNode_y)),
        'to': (0,0),
        'type': type[i],
        'from': (0,0),
        'road': 0,
        'top': top,
        'left': left,
        'right': right,
        'down': down
        })

def checkForTurningPoints(x,y,x_search,y_search,direction):
    hsv = cv2.cvtColor(cropped, cv2.COLOR_BGR2HSV)
    height, width = hsv.shape[:2]  # image dimensions

    if direction == 'left':
        x_start = max(x - 85, 0)
        x_end   = x                  # assuming x <= width
        y_start = max(y - 8, 0)
        y_end   = min(y + 8, height)
    elif direction == 'top':
        x_start = max(x - 8, 0)
        x_end   = min(x + 8, width)
        y_start = max(y - 85, 0)
        y_end   = y
    elif direction == 'right':
        x_start = x
        x_end   = min(x + 85, cropped.shape[1])                 # assuming x <= width
        y_start = max(y - 8, 0)
        y_end   = min(y + 8, height)
    elif direction == 'down':
        x_start = max(x - 8, 0)
        x_end   = min(x + 8, width)
        y_start = y
        y_end   = min(y + 85, cropped.shape[0])
    else:
        return False
    
    #cv2.rectangle(result,(x_start,y_start),(x_end,y_end),(0,0,255),1)

    if(x_start < x_search < x_end) and (y_start < y_search < y_end):
        return True
    

visited = set()
reordered = [] 

def reorder(turning_points, start=0, search=False):
    print(f"Start reordering with start={start}, search={search}")
    
    directions = ['top', 'left', 'right', 'down']

    for i, point in enumerate(turning_points):
        x, y = int(point[0]), int(point[1])
        print(f"Loop {i} with start={start}")
        
        if (x, y) not in visited and (i == start or not search):
            print(f"Adding point {i}: ({x}, {y})")
            reordered.append((x, y))
            visited.add((x, y))
            print(f"Started processing at index {i}")

            # Check each direction
            for direction in directions:
                blocked = checkPlot(x, y, direction, 50)
                if not blocked:
                    for j, other_point in enumerate(turning_points):
                        ox, oy = int(other_point[0]), int(other_point[1])
                        if checkForTurningPoints(x, y, ox, oy, direction) and ((ox,oy) not in visited):
                            print(f"Turning point found at index {j} in direction {direction}")
                            reorder(turning_points, j, True)

            # Continue the main iteration if this was the start
            if start == i:
                reorder(turning_points)
                
    return reordered



def checkPlot(x,y, direction,bound):
    # Define search area based on direction
    hsv = cv2.cvtColor(cropped, cv2.COLOR_BGR2HSV)
    height, width = hsv.shape[:2]  # image dimensions

    if direction == 'left':
        x_start = max(x - bound - 10, 0)
        x_end   = x                  # assuming x <= width
        y_start = max(y - 10, 0)
        y_end   = min(y + 10, height)
    elif direction == 'top':
        x_start = max(x - 10, 0)
        x_end   = min(x + 10, width)
        y_start = max(y - bound - 10, 0)
        y_end   = y
    elif direction == 'right':
        x_start = x
        x_end   = min(x + bound + 10, cropped.shape[1])                 # assuming x <= width
        y_start = max(y - 10, 0)
        y_end   = min(y + 10, height)
    elif direction == 'down':
        x_start = max(x - 10, 0)
        x_end   = min(x + 10, width)
        y_start = y
        y_end   = min(y + bound + 10, cropped.shape[0])
    else:
        return False

    #cv2.rectangle(result, (x_start,y_start),(x_end,y_end),(0,0,255),1)
    # Extract the search region
    region = hsv[y_start:y_end, x_start:x_end]

    maskWhite = cv2.inRange(region, lower_White, upper_White) 

    if region.size == 0:
        return False
    
# If more than 1% of the region is yellow, consider it yellow
    white_ratio = np.sum(maskWhite > 0) / maskWhite.size
    return white_ratio > 0.01

turningPoints = midTurningPoints.copy()

turningPoints = [
    p for p in turningPoints
    if p is not None and len(p) == 2
]

turningPoints = reorder(turningPoints)


#Draw TurningPoints
# for i, point in enumerate(turningPoints):
#     y = int(point[1])
#     x = int(point[0])

#     cv2.circle(result,(x,y),(dotRadius),(0,0,255),-1)

lastTop = False
lastDown = False
lastRight = False
lastLeft = False
flipped = False

for i, point in enumerate(turningPoints):
    height, width = img.shape[:2]
    print(f"Point {i}")
    x = int(point[0])
    y = int(point[1])

    cv2.putText(result,str(i),(x,y),cv2.FONT_HERSHEY_SIMPLEX, 0.26,(0,0,0),1)

    flip = False
    rightCheck = None
    leftCheck = None
    topCheck = None
    downCheck = None

    top = checkPlot(x, y, 'top',50)
    left = checkPlot(x, y, 'left',50)
    right = checkPlot(x,y, 'right',50)
    down = checkPlot(x,y,'down',50)

    drawDown = (top and left) or (not top and not left)
    print(f"Start Here: {left, right,top,down}")

    #first check
    if not left:
        lastLeft = True
        leftCheck = checkPlot(x,y,'left',85)
    elif not right:
        lastRight = True
        rightCheck = checkPlot(x,y,'right',85)

    #2nd check
    if rightCheck or leftCheck:
        if not top:
            lastTop = True
            topCheck = checkPlot(x, y, 'top',85)
        elif not down:
            lastDown = True
            downCheck = checkPlot(x,y,'down',85)
        
    if(rightCheck or leftCheck):
        checkPrevious = True
    else:
        lastTop = False
        lastDown = False
        lastRight = False
        lastLeft = False
        checkPrevious = False

    drawPoints(drawDown, x,y, height, top, left, right, down, checkPrevious)

    lastTop = False
    lastDown = False
    lastRight = False
    lastLeft = False

print(f"Filtered to {len(yellowBlocks)} actual blocks (removed {len(contoursYel) - len(yellowBlocks)} small dots)")
#------------------------------------------------------------
#------------------------------------------------------------

centerBlock = []
horizontalExtensionNodes = []
verticalExtensionNodes = []
intersectionNodes = []
roads = set()

# Convert to HSV for yellow detection
hsv = cv2.cvtColor(cropped, cv2.COLOR_BGR2HSV)

# Define yellow color range
lower_yellow = np.array([20, 100, 100])
upper_yellow = np.array([30, 255, 255])

def is_yellow_in_area(x, y, width, height, direction):
    """Check if yellow exists in a rectangular search area"""
    # Define search area based on direction
    if direction == 'left':
        x_start = max(0, x - 10)
        x_end = x + 5
        y_start = max(0, y - height // 2)
        y_end = min(cropped.shape[0], y + height // 2)
    elif direction == 'right':
        x_start = x
        x_end = min(cropped.shape[1], x + 15)
        y_start = max(0, y - height // 2)
        y_end = min(cropped.shape[0], y + height // 2)
    elif direction == 'top':
        x_start = max(0, x - width // 2)
        x_end = min(cropped.shape[1], x + width // 2)
        y_start = max(0, y - 15)
        y_end = y
    elif direction == 'bottom':
        x_start = max(0, x - width // 2)
        x_end = min(cropped.shape[1], x + width // 2)
        y_start = y
        y_end = min(cropped.shape[0], y + 15)
    else:
        return False
    
    # Extract the search region
    region = hsv[y_start:y_end, x_start:x_end]
    
    if region.size == 0:
        return False
    
    # Create yellow mask for this region
    yellow_mask = cv2.inRange(region, lower_yellow, upper_yellow)
    
    # If more than 10% of the region is yellow, consider it yellow
    yellow_ratio = np.sum(yellow_mask > 0) / yellow_mask.size
    return yellow_ratio > 0.01
#!!!!!! change name of function, reduce it dont need direction
def is_Intersection(x, y, width, height,direction):
    if(direction == 'top'):
        x_start = x
        x_end = max(x + 150, x + height * 3)
        y_start = max(0, y - height * 2)
        y_end = max(y + height * 2, y + 80)

    elif(direction == 'bottom'):
        x_start = max(0, x - height * 5)
        x_end = x
        y_start = max(0, y - height)
        y_end = max(y + height * 4, y + 80)

    # Extract the search region
    region = hsv[y_start:y_end, x_start:x_end]
    
    if region.size == 0:
        print(f"Region at {x} is not valid")
        return False
    
    # Create red mask for this region
    red_mask= cv2.inRange(region, red_lower, red_upper)
    
    red_ratio = np.sum(red_mask > 0) / red_mask.size
    return red_ratio > 0.01

roadCount = 1
# Process horizontal blocks
print("\n=== Processing Horizontal Blocks ===")
for i, block in enumerate(horizontalBlocks):
    x, y, w, h = block['x'], block['y'], block['w'], block['h']
    
    # Calculate center
    center_x = x + w // 2
    center_y = y + h // 2

    
    cv2.circle(result, (center_x, center_y), dotRadius, (255, 0, 0), -1)
    
    # Find min and max x coordinates
    x_min = x
    x_max = x + w
    
    # Check left side - search area next to left edge
    if is_yellow_in_area(x_min, center_y, w, h, 'left'):
        # Add node to the left
        new_x = center_x - w
        horizontalExtensionNodes.append({
            'block_id': f"H{i+1}",
            'direction': 'left',
            'x': new_x,
            'y': center_y,
            'isExit': False,
            'road': roadCount,
            'to': (0,0),
            'from': (0,0)
        })

        centerBlock.append({
        'block_id': f"H{i+1}",
        'type': 'horizontal',
        'x': center_x,
        'y': center_y,
        'bounds': {'x': x, 'y': y, 'width': w, 'height': h},
        'yellowOn': 'left',
        'isExit': True,
        'road': roadCount,
        'to': (0,0),
        'from': (0,0)
        })
        roadCount += 1
        
        cv2.circle(result, (new_x, center_y), dotRadius, (255, 0, 0), -1) 
        print(f"Block H{i+1}: Added LEFT node at ({new_x}, {center_y})")
    
    # Check right side - search area next to right edge
    if is_yellow_in_area(x_max, center_y, w, h, 'right'):
        # Add node to the right
        new_x = center_x + w
        horizontalExtensionNodes.append({
            'block_id': f"H{i+1}",
            'direction': 'right',
            'x': new_x,
            'y': center_y,
            'to': None,
            'from': (0,0),
            'isExit': False,
            'road': roadCount,
            'to': (0,0),
            'from': (0,0)
        })
        centerBlock.append({
        'block_id': f"H{i+1}",
        'type': 'horizontal',
        'x': center_x,
        'y': center_y,
        'bounds': {'x': x, 'y': y, 'width': w, 'height': h},
        'yellowOn': 'right',
        'isExit': True,
        'road': roadCount,
        'to': (0,0),
        'from': (0,0)
        })
        roadCount +=1
        cv2.circle(result, (new_x, center_y), dotRadius, (255, 0, 0), -1)  
        print(f"Block H{i+1}: Added RIGHT node at ({new_x}, {center_y})")
    

# Process vertical blocks
print("\n=== Processing Vertical Blocks ===")
visitedBlocks = set()
for i, block in enumerate(verticalBlocks):
    x, y, w, h = block['x'], block['y'], block['w'], block['h']
    
    # Calculate center
    center_x = x + w // 2
    center_y = y + h // 2
    
    cv2.circle(result, (center_x, center_y), dotRadius, (255, 0, 0), -1)
    
    # Find min and max y coordinates
    y_min = y
    y_max = y + h
    
    # Check top side - search area above top edge
    if is_yellow_in_area(center_x, y_min, w, h, 'top'):
        # Add node to the top
        new_y = center_y - h
        verticalExtensionNodes.append({
            'block_id': f"V{i+1}",
            'direction': 'top',
            'x': center_x,
            'y': new_y,
            'h': h,
            'to': (0,0),
            'from': (0,0),
            'isExit': False,
            'road': roadCount,
        })
        centerBlock.append({
        'block_id': f"V{i+1}",
        'type': 'vertical',
        'x': center_x,
        'y': center_y,
        'bounds': {'x': x, 'y': y, 'width': w, 'height': h},
        'yellowOn': 'top',
        'isExit': True,
        'road': roadCount,
        'to': (0,0),
        'from': (0,0)
        })
        roadCount +=1


        if is_Intersection(center_x, y_min, w, h,'top'):
            x_start = x
            x_end = max(x + 150, x + height * 3)
            y_start = max(0, y - height * 2)
            y_end = max(y + height * 2, y + 80)
            for vB in verticalBlocks:
                if x_start <= vB['x'] <= x_end and y_start <= vB['y'] <= y_end:
                    visitedBlocks.add((vB['x'],vB['y']))

            midpoint_y = (center_y + new_y) // 2
            midpoint_x = (max(x + 38, x + h + 10))
            cv2.circle(result, (midpoint_x, midpoint_y), dotRadius, (255,0,0), -1) #Blue
            intersectionNodes.append({
                'x': midpoint_x,
                'y': midpoint_y,
                'height': h
            })

        cv2.circle(result, (center_x, new_y), dotRadius, (255, 0, 0), -1) 
        print(f"Block V{i+1}: Added TOP node at ({center_x}, {new_y})")
    
    # Check bottom side - search area below bottom edge
    if is_yellow_in_area(center_x, y_max, w, h, 'bottom'):
        # Add node to the bottom
        new_y = center_y + h
        verticalExtensionNodes.append({
            'block_id': f"V{i+1}",
            'direction': 'bottom',
            'x': center_x,
            'y': new_y,
            'h': h,
            'road': roadCount,
            'to': (0,0),
            'from': (0,0),
            'isExit': True,
        })
        centerBlock.append({
            'block_id': f"V{i+1}",
            'type': 'vertical',
            'x': center_x,
            'y': center_y,
            'bounds': {'x': x, 'y': y, 'width': w, 'height': h},
            'yellowOn': 'bottom',
            'isExit': True,
            'road': roadCount,
            'to': (0,0),
            'from': (0,0)
        })
        roadCount+=1
        if is_Intersection(center_x, y_min, w, h,'bottom') and (block['x'],block['y']) not in visitedBlocks:
            x_start = max(x - 150, x - height * 5)
            x_end = x
            y_start = max(0, y - height)
            y_end = max(y + height * 4, y + 80)
            for vB in verticalBlocks:
                if x_start <= vB['x'] <= x_end and y_start <= vB['y'] <= y_end:
                    visitedBlocks.add((vB['x'],vB['y']))
            midpoint_y = (center_y + new_y) // 2
            midpoint_x = (max(x - 38, x - h - 10))
            cv2.circle(result, (midpoint_x, midpoint_y), dotRadius, (255,0,0), -1) #Blue
            intersectionNodes.append({
                'x': midpoint_x,
                'y': midpoint_y,
                'height': h
            })

        cv2.circle(result, (center_x, new_y), dotRadius, (255, 0, 0), -1)  
        print(f"Block V{i+1}: Added BOTTOM node at ({center_x}, {new_y})")

#Testing
# NEW CODE: Create edges from intersection nodes
edges = []

def calculate_distance(x1, y1, x2, y2):
    """Calculate Euclidean distance between two points"""
    return math.sqrt((x2 - x1)**2 + (y2 - y1)**2)

def draw_arrow_with_weight(img, pt1, pt2, weight, color=(0, 255, 0), thickness = 1):
    """Draw an arrow from pt1 to pt2 with weight label"""
    # Draw arrow
    cv2.arrowedLine(img, pt1, pt2, color, thickness, tipLength=0.2)
    
    if(draw_weights):
        # Calculate midpoint for weight label
        mid_x = (pt1[0] + pt2[0]) // 2
        mid_y = (pt1[1] + pt2[1]) // 2
        
        # Draw weight text with background for readability
        weight_text = f"{weight:.1f}"
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.26
        font_thickness = 1
        
        # Get text size for background rectangle
        (text_width, text_height), baseline = cv2.getTextSize(weight_text, font, font_scale, font_thickness)
        
        # Draw white background rectangle
        cv2.rectangle(img, 
                    (mid_x - text_width // 2 , mid_y - text_height - 2),
                    (mid_x + text_width // 2, mid_y + 2),
                    (255, 255, 255), -1)
        
        # Draw text
        cv2.putText(img, weight_text, 
                    (mid_x - text_width // 2, mid_y), 
                    font, font_scale, (0, 0, 0), font_thickness)
        
print("\n=== Creating Edges from Intersection Nodes ===")
for intersection in intersectionNodes:
    int_x = intersection['x']
    int_y = intersection['y']
    searchRadius = intersection['height'] + 10
    
    # Create a square search area with intersection as center
    search_radius = searchRadius + 5 # Size of the square from center
    square_x_start = max(0, int_x - search_radius)
    square_x_end = min(cropped.shape[1], int_x + search_radius)
    square_y_start = max(0, int_y - search_radius)
    square_y_end = min(cropped.shape[0], int_y + search_radius)
    print(f"\nSearching around intersection at ({int_x}, {int_y})")
    
    # Search for vertical extension nodes
    for v_ext in verticalExtensionNodes:
        if (square_x_start <= v_ext['x'] <= square_x_end and 
            square_y_start <= v_ext['y'] <= square_y_end):
            weight = int(calculate_distance(int_x, int_y, v_ext['x'], v_ext['y']))
            edges.append({
                'from': (int(intersection['x']),int(intersection['y'])),
                'to': (int(v_ext['x']),int(v_ext['y'])),
                'weight': int(weight)
            })
            v_ext['from'] = (int_x,int_y)
            draw_arrow_with_weight(result, (int_x, int_y), (v_ext['x'], v_ext['y']), weight)
            print(f"  Edge to vertical extension {v_ext['block_id']}, weight: {weight:.2f}")
    
    # Search for horizontal extension nodes
    for h_ext in horizontalExtensionNodes:
        if (square_x_start <= h_ext['x'] <= square_x_end and 
            square_y_start <= h_ext['y'] <= square_y_end):
            weight = int(calculate_distance(int_x, int_y, h_ext['x'], h_ext['y']))
            edges.append({
                'from': (int(intersection['x']),int(intersection['y'])),
                'to': (int(h_ext['x']),int(h_ext['y'])),
                'weight': weight
            })
            h_ext['from'] = (int_x,int_y)
            draw_arrow_with_weight(result, (int_x, int_y), (h_ext['x'], h_ext['y']), weight)
            print(f"  Edge to horizontal extension {h_ext['block_id']}, weight: {weight:.2f}")
    
    # Search for center nodes
    for center in centerBlock:
        if (square_x_start <= center['x'] <= square_x_end and 
            square_y_start <= center['y'] <= square_y_end):
            weight = int(calculate_distance(center['x'], center['y'], int_x, int_y))
            edges.append({
                'from': (int(center['x']),int(center['y'])),
                'to': (int(intersection['x']),int(intersection['y'])),
                'weight': weight
            })
            center['to'] = (int_x,int_y)
            draw_arrow_with_weight(result, (center['x'], center['y']), (int_x, int_y), weight)
            print(f"  Edge from center {center['block_id']}, weight: {weight:.2f}")


# Bounds for visualisation  
# cv2.circle(result, (node_x, node_y), 25, (0,0,255))  
#      
# cv2.rectangle(
# result,
# (square_x_start, square_y_start),  # top-left corner
# (square_x_end, square_y_end),      # bottom-right corner
# color=(0, 0, 255),                 # red rectangle
# thickness=2
# )
def crosses_border(p1, p2, border_mask): 
    line_mask = np.zeros_like(border_mask) 
    cv2.line(line_mask, p1, p2, 255, thickness=3) 
    return np.any(cv2.bitwise_and(line_mask, border_mask))



def direction_ok(node_x, node_y, x_search, y_search, dir_vec, min_dot=0.3):
    vx = x_search - node_x
    vy = y_search - node_y

    mag = np.hypot(vx, vy)
    if mag == 0:
        return False

    vx /= mag
    vy /= mag

    dot = vx * dir_vec[0] + vy * dir_vec[1]
    return dot > min_dot

def findClose_withoutBorder(x_start, x_end, y_start, y_end, x_search, y_search,node_x, node_y, border_mask,dir_vec=0):
    if not (x_start <= x_search <= x_end and
        y_start <= y_search <= y_end):
            return
    # if not direction_ok(node_x,node_y,x_search,y_search,dir_vec):
    #         return 
    weight = calculate_distance(x_search, y_search, node_x, node_y)
    
    if(foundNode['weight'] == 0):
        foundNode['x'] = x_search
        foundNode['y'] = y_search
        foundNode['weight'] = weight
    elif(foundNode['weight'] != 0 and foundNode['weight'] > weight):
        foundNode['x'] = x_search
        foundNode['y'] = y_search
        foundNode['weight'] = weight

def findClose(x_start, x_end, y_start, y_end, x_search, y_search,node_x, node_y, border_mask,dir_vec=0):
    if not (x_start <= x_search <= x_end and
        y_start <= y_search <= y_end):
            return
    # if not direction_ok(node_x,node_y,x_search,y_search,dir_vec):
    #         return 

    if crosses_border((node_x,node_y),(x_search,y_search),resultBorder):
        weight = calculate_distance(x_search,y_search,node_x,node_y)
        return True 
    
    weight = calculate_distance(x_search, y_search, node_x, node_y)

    
    if(foundNode['weight'] == 0):
        foundNode['x'] = x_search
        foundNode['y'] = y_search
        foundNode['weight'] = weight
    elif(foundNode['weight'] != 0 and foundNode['weight'] > weight):
        foundNode['x'] = x_search
        foundNode['y'] = y_search
        foundNode['weight'] = weight

def markPoint(point_x, point_y ,turningPointNodes,x,y,to):
    if(any(node['point'][0] == point_x and node['point'][1] == point_y for node in turningPointNodes)):
        for node in turningPointNodes:
            if node['point'] == (point_x,point_y) and node[to] == (0,0):
                node[to] = (x,y)
                return True
    return False

foundNode = {
            'x': 0,
            'y': 0,
            'weight': 0
        }
doneCycle = False
count = 0

print("\n=== Creating Edges from Vertical Nodes ===")
for node in centerBlock:
    if node['type'] == 'vertical' and node['yellowOn'] == 'bottom':
        node_x = node['x']
        node_y = node['y']
        
        search_radius = (node['bounds']['height'] // 2) + 10
        square_y_start = max(0, node_y - search_radius)
        square_y_end = min(cropped.shape[0], node_y + search_radius)

        for i in range(2):
            #search left
            if(i == 0):
                square_x_start = 0
                square_x_end = node_x + 5

                #cv2.rectangle(result,(square_x_start,square_y_start),(square_x_end,square_y_end),(0,0,255),1)
                # for verExt in verticalExtensionNodes:
                #     verExt_x = verExt['x']
                #     verExt_y = verExt['y']
                #     findClose(square_x_start,square_x_end,square_y_start,square_y_end,verExt_x,verExt_y, node_x,node_y, resultBorder)
                #     #Search for turning point
                
                # if(foundNode['weight'] != 0):
                #     edges.append({
                #             'from': (int(node_x),int(node_y)),
                #             'to': (int(foundNode['x']),int(foundNode['y'])),
                #             'weight': int(foundNode['weight'])
                #         })
                #     for verExt in verticalExtensionNodes:
                #         if(verExt['x'] == foundNode['x']) and (verExt['y'] == foundNode['y']):
                #             verExt['from'] = (node_x,node_y)
                #             node['to'] = (verExt_x,verExt_y)
                #             if(verExt['road'] >= node['road']):
                #                 verExt['road'] = node['road']
                #             else: 
                #                 node['road'] = verExt['road']
                #     draw_arrow_with_weight(result,  (node_x, node_y),(foundNode['x'], foundNode['y']), foundNode['weight'])
                #     foundNode['weight'] = 0  
                # foundNode['weight'] = 0
            #search right
            else:
                square_x_start = node_x
                square_x_end = cropped.shape[1]

                for verExt in verticalExtensionNodes:
                    verExt_x = verExt['x']
                    verExt_y = verExt['y']
                    findClose(square_x_start,square_x_end,square_y_start,square_y_end,verExt_x,verExt_y, node_x,node_y, resultBorder)

                #cv2.rectangle(result,(square_x_start,square_y_start),(square_x_end,square_y_end),(0,0,255),1)
                if(foundNode['weight'] != 0): 
                    edges.append({
                            'from': (int(foundNode['x']),int(foundNode['y'])),
                            'to': (int(node_x),int(node_y)),
                            'weight': int(foundNode['weight'])
                        })
                    for verExt in verticalExtensionNodes:
                        if(verExt['x'] == foundNode['x']) and (verExt['y'] == foundNode['y']):
                                verExt['to'] = (node_x,node_y)
                                node['from'] = (verExt_x,verExt_y)
                                if(verExt['road'] >= node['road']):
                                    verExt['road'] = node['road']
                                else: 
                                    node['road'] = verExt['road']
                    draw_arrow_with_weight(result,(foundNode['x'], foundNode['y']), (node_x, node_y), foundNode['weight'])                    
                    foundNode['weight'] = 0   
                else: 
                    foundNode['weight'] = 0    
                    crossed_x = 0
                    crossed_y = 0
                    crossed = False
                    #add search for turning points
                    for point in turningPointNodes:
                        x = int(point['point'][0])
                        y = int(point['point'][1])

                        crossed = findClose(square_x_start,square_x_end,square_y_start,square_y_end,x,y, node_x,node_y, resultBorder)
                        if(crossed):
                            crossed_x = x
                            crossed_y = y
                    #cv2.rectangle(result,(square_x_start, square_y_start),(square_x_end,square_y_end),(0,255,255),1)
                    if(crossed_x != 0):
                        findClose_withoutBorder(square_x_start,square_x_end,square_y_start,square_y_end,crossed_x,crossed_y,node_x,node_y,resultBorder)
                    #cv2.circle(result,(foundNode['x'],foundNode['y']),dotRadius + 20,(0,0,255),1)
                    print(f"{crossed} with point {foundNode['x'],foundNode['y']}")
                    if(foundNode['weight'] != 0):
                        marked = markPoint(foundNode['x'],foundNode['y'],turningPointNodes,node_x,node_y,'to')
                        if(marked):
                            edges.append({
                                'from': (int(foundNode['x']),int(foundNode['y'])),
                                'to': (int(node_x),int(node_y)),
                                'weight': int(foundNode['weight'])
                            })
                            for point in turningPointNodes:
                                if(int(point['point'][0]) == foundNode['x']) and (int(point['point'][1]) == foundNode['y']):
                                    #cv2.circle(result,(point['point'][0],point['point'][1]),dotRadius+20,(0,0,255),1)
                                    node['from'] = (int(point['point'][0]),int(point['point'][1]))
                                    if(node['road'] >= point['road'] and point['road'] != 0):
                                        node['road'] = point['road']
                                    else: 
                                        point['road'] = node['road']
                                    #print(f"from: {point['from']} to {point['to']} on {point['point']}")
                            draw_arrow_with_weight(result,(foundNode['x'], foundNode['y'] ),  (node_x, node_y), foundNode['weight'])
                        foundNode['weight'] = 0   
                    foundNode['weight'] = 0
                foundNode['weight'] = 0
   
    elif(node['type'] == 'vertical' and node['yellowOn'] == 'top'):
        node_x = node['x']
        node_y = node['y']
        
        search_radius = (node['bounds']['height'] // 2) + 10
        square_y_start = max(0, node_y - search_radius)
        square_y_end = min(cropped.shape[0], node_y + search_radius)

        for i in range(2):
            #search left
            if(i == 0):
                square_x_start = 0
                square_x_end = node_x + 5
                #cv2.rectangle(result,(square_x_start,square_y_start),(square_x_end,square_y_end),(0,0,255),1)
                for verExt in verticalExtensionNodes:
                    verExt_x = verExt['x']
                    verExt_y = verExt['y']         
                    findClose(square_x_start,square_x_end,square_y_start,square_y_end,verExt_x,verExt_y, node_x,node_y, resultBorder)

                if(foundNode['weight'] != 0):
                    # print(f"left node found at {node_x}, {node_y}")
                    edges.append({
                            'from': (int(foundNode['x']),int(foundNode['y'])),
                            'to': (int(node_x),int(node_y)),
                            'weight': int(foundNode['weight'])
                        })
                    for verExt in verticalExtensionNodes:
                        if(verExt['x'] == foundNode['x']) and (verExt['y'] == foundNode['y']):
                                verExt['to'] = (node_x,node_y)
                                node['from'] = (verExt_x,verExt_y)
                                if(verExt['road'] >= node['road']):
                                    verExt['road'] = node['road']
                                else: 
                                    node['road'] = verExt['road']
                    draw_arrow_with_weight(result,(foundNode['x'], foundNode['y']),  (node_x, node_y), foundNode['weight'])
                    foundNode['weight'] = 0      
                else: 
                    if(square_y_start != 0):
                            square_y_start -= 10
                    foundNode['weight'] = 0   
                    crossed_x = 0
                    crossed_y = 0
                    crossed = False
                    #add search for turning points
                    for point in turningPointNodes:
                        x = int(point['point'][0])
                        y = int(point['point'][1])

                        crossed = findClose(square_x_start,square_x_end,square_y_start,square_y_end,x,y, node_x,node_y, resultBorder)
                        if(crossed):
                            crossed_x = x
                            crossed_y = y                   
                    #cv2.rectangle(result,(square_x_start,square_y_start),(square_x_end,square_y_end),(0,0,255),1)
                    if(crossed_x != 0):
                        findClose_withoutBorder(square_x_start,square_x_end,square_y_start,square_y_end,crossed_x,crossed_y, node_x,node_y, resultBorder)
                        #cv2.circle(result,(crossed_x,crossed_y),dotRadius + 20,(0,0,255),1)
                    if(foundNode['weight'] != 0):
                        marked = markPoint(foundNode['x'],foundNode['y'],turningPointNodes,node_x,node_y, 'to')
                        if(marked):
                            #print(f"Marked with {foundNode['x'],foundNode['y']}")
                            edges.append({
                                'from': (int(foundNode['x']),int(foundNode['y'])),
                                'to': (int(node_x),int(node_y)),
                                'weight': int(foundNode['weight'])
                            })
                            for point in turningPointNodes:
                                if(int(point['point'][0]) == foundNode['x']) and int(point['point'][1]) == foundNode['y']:
                                    #print(f"{foundNode['x'],foundNode['y']} entered")
                                    #cv2.circle(result,(foundNode['x'],foundNode['y']),dotRadius+20,(0,0,255),1)
                                    node['from'] = (int(point['point'][0]),int(point['point'][1]))
                                    if(node['road'] >= point['road'] and point['road'] != 0):
                                        node['road'] = point['road']
                                    else: 
                                        point['road'] = node['road']
                                    #print(f"from: {point['from']} to {point['to']} on {point['point']}")
                            draw_arrow_with_weight(result,(foundNode['x'], foundNode['y'] ),  (node_x, node_y), foundNode['weight'])
                        foundNode['weight'] = 0
                    foundNode['weight'] = 0
            #search right
            else:
                square_x_start = node_x
                square_x_end = cropped.shape[1]

                # for verExt in verticalExtensionNodes:
                #     verExt_x = verExt['x']
                #     verExt_y = verExt['y']

                #     findClose(square_x_start,square_x_end,square_y_start,square_y_end,verExt_x,verExt_y, node_x,node_y, resultBorder)

                # if(foundNode['weight'] != 0):
                #     edges.append({
                #             'from': (int(node_x),int(node_y)),
                #             'to': (int(foundNode['x']),int(foundNode['y'])),
                #             'weight': int(foundNode['weight'])
                #         })
                #     for verExt in verticalExtensionNodes:
                #         if(verExt['x'] == foundNode['x']) and (verExt['y'] == foundNode['y']):
                #                 verExt['from'] = (node_x,node_y)
                #                 node['to'] = (verExt_x,verExt_y)
                #                 if(verExt['road'] >= node['road']):
                #                     verExt['road'] = node['road']
                #                 else: 
                #                     node['road'] = verExt['road']
                #     draw_arrow_with_weight(result,  (node_x, node_y),(foundNode['x'], foundNode['y']), foundNode['weight'])
                #     foundNode['weight'] = 0    

print("\n=== Creating Edges from Horizontal Nodes ===")
for node in centerBlock:
    if node['type'] == 'horizontal' and node['yellowOn'] == 'right':
        node_x = node['x']
        node_y = node['y']

        search_radius = node['bounds']['width'] // 2 + 5
        square_x_start = max(0, node_x - search_radius)
        square_x_end = min(cropped.shape[1], node_x + search_radius)

        for i in range(2):
            #search up
            if(i == 0):
                square_y_start = 0
                square_y_end = node_y + 5  
                #search for horizontal extension nodes
                for horExt in horizontalExtensionNodes:
        
                    horExt_x = horExt['x']
                    horExt_y = horExt['y']

                    # Add nodes into found nodes
                    findClose(square_x_start,square_x_end,square_y_start,square_y_end,horExt_x,horExt_y,node_x,node_y, resultBorder)
           
                if(foundNode['weight'] != 0):
                    edges.append({
                            'from': (int(foundNode['x']),int(foundNode['y'])),
                            'to': (int(node_x),int(node_y)),
                            'weight': int(foundNode['weight'])
                        })
                    for horExt in horizontalExtensionNodes:
                        if(horExt['x'] == foundNode['x']) and (horExt['y'] == foundNode['y']):
                                horExt['to'] = (node_x,node_y)
                                node['from'] = (horExt_x,horExt_y)
                                if(horExt['road'] >= node['road']):
                                    horExt['road'] = node['road']
                                else: 
                                    node['road'] = horExt['road']
                    draw_arrow_with_weight(result, (foundNode['x'], foundNode['y']), (node_x, node_y), foundNode['weight'])
                    foundNode['weight'] = 0
                else:
                    foundNode['weight'] = 0    
                    crossed = False
                    crossed_x = 0
                    crossed_y = 0
                    if(square_x_start!= 0):
                        square_x_start -= 10

                    #cv2.rectangle(result,(square_x_start,square_y_start), (square_x_end,square_y_end),(0,0,255),1)
                    #add search for turning points
                    for point in turningPointNodes:
                        x = int(point['point'][0])
                        y = int(point['point'][1])

                        crossed = findClose(square_x_start,square_x_end,square_y_start,square_y_end,x,y, node_x,node_y, resultBorder)
                        if(crossed):
                            crossed_x = x
                            crossed_y = y
                    if(crossed_x != 0):
                        findClose_withoutBorder(square_x_start,square_x_end,square_y_start,square_y_end,crossed_x,crossed_y, node_x,node_y, resultBorder)
                    if(foundNode['weight'] != 0):
                        marked = markPoint(foundNode['x'],foundNode['y'],turningPointNodes,node_x,node_y,'to')
                        if(marked):
                            edges.append({
                                'from': (int(foundNode['x']),int(foundNode['y'])),
                                'to': (int(node_x),int(node_y)),
                                'weight': int(foundNode['weight'])
                            })
                            for point in turningPointNodes:
                                if(int(point['point'][0])== foundNode['x']) and int(point['point'][1]) == foundNode['y']:
                                    node['from'] = (int(point['point'][0]),int(point['point'][1]))
                                    if(horExt['road'] >= point['road'] and point['road'] != 0):
                                        horExt['road'] = point['road']
                                    else: 
                                        point['road'] = horExt['road']
                                    #print(f"from: {point['from']} to {point['to']} on {point['point']}")
                                    #cv2.circle(result,(point['point']),dotRadius+20,(0,0,255),1)
                            draw_arrow_with_weight(result,(foundNode['x'], foundNode['y'] ),  (node_x, node_y), foundNode['weight'])
                            foundNode['weight'] = 0   
                        foundNode['weight'] = 0
                foundNode['weight'] = 0
            #search down
            # else:
            #     square_y_start = node_y
            #     square_y_end = cropped.shape[0]  
            #     #search for horizontal extension nodes
            #     for horExt in horizontalExtensionNodes:
        
            #         horExt_x = horExt['x']
            #         horExt_y = horExt['y']

            #         # Add nodes into found nodes
            #         findClose(square_x_start,square_x_end,square_y_start,square_y_end,horExt_x,horExt_y,node_x,node_y, resultBorder)

                            
            #     if(foundNode['weight'] != 0):
            #         edges.append({
            #                 'from': (int(node_x),int(node_y)),
            #                 'to': (int(foundNode['x']),int(foundNode['y'])),
            #                 'weight': int(foundNode['weight'])
            #             })
            #         for horExt in horizontalExtensionNodes:
            #             if(horExt['x'] == foundNode['x']) and (horExt['y'] == foundNode['y']):
            #                     horExt['from'] = (node_x,node_y)
            #                     node['to'] = (horExt_x,horExt_y)
            #                     if(horExt['road'] >= node['road']):
            #                         horExt['road'] = node['road']
            #                     else: 
            #                         node['road'] = horExt['road']
            #         draw_arrow_with_weight(result, (node_x, node_y), (foundNode['x'], foundNode['y']), foundNode['weight'])
            #         foundNode['weight'] = 0
            #     foundNode['weight'] = 0
    elif(node['type'] == 'horizontal' and node['yellowOn'] == 'left'):
        node_x = node['x']
        node_y = node['y']

        searchRadius = node['bounds']['width'] // 2 + 5
        # Create a rectangle search area with intersection as center
        search_radius = searchRadius # Size of the square from centers
        square_x_start = max(0, node_x - search_radius - 10)
        square_x_end = min(cropped.shape[1], node_x + search_radius)

        for i in range(2):
            #search up
            if(i == 0):
                square_y_start = 0
                square_y_end = node_y + 5  
                #search for horizontal extension nodes
                # for horExt in horizontalExtensionNodes:
        
                #     horExt_x = horExt['x']
                #     horExt_y = horExt['y']

                #     # Add nodes into found nodes
                #     findClose(square_x_start,square_x_end,square_y_start,square_y_end,horExt_x,horExt_y,node_x,node_y, resultBorder)

                            
                # if(foundNode['weight'] != 0):
                #     edges.append({
                #             'from': (int(node_x),int(node_y)),
                #             'to': (int(foundNode['x']),int(foundNode['y'])),
                #             'weight': int(foundNode['weight'])
                #         })
                #     for horExt in horizontalExtensionNodes:
                #             if(horExt['x'] == foundNode['x']) and (horExt['y'] == foundNode['y']):
                #                 horExt['from'] = (node_x,node_y)
                #                 node['to'] = (horExt_x,horExt_y)
                #                 if(horExt['road'] >= node['road']):
                #                     horExt['road'] = node['road']
                #                 else: 
                #                     node['road'] = horExt['road']
                #     draw_arrow_with_weight(result, (node_x, node_y), (foundNode['x'], foundNode['y']), foundNode['weight'])
                #     foundNode['weight'] = 0  
                # foundNode['weight'] = 0

            #search down
            else:
                
                square_y_start = node_y
                square_y_end = cropped.shape[0]  
                #search for horizontal extension nodes
                for horExt in horizontalExtensionNodes:
        
                    horExt_x = horExt['x']
                    horExt_y = horExt['y']

                    # Add nodes into found nodes
                    findClose(square_x_start,square_x_end,square_y_start,square_y_end,horExt_x,horExt_y,node_x,node_y, resultBorder)
                    
                if(foundNode['weight'] != 0):
                    edges.append({
                            'from': (int(foundNode['x']),int(foundNode['y'])),
                            'to': (int(node_x),int(node_y)),
                            'weight': int(foundNode['weight'])
                        })
                    for horExt in horizontalExtensionNodes:
                            if(horExt['x'] == foundNode['x']) and (horExt['y'] == foundNode['y']):
                                horExt['to'] = (node_x,node_y)
                                node['from'] = (horExt_x,horExt_y)
                                if(horExt['road'] >= node['road']):
                                    horExt['road'] = node['road']
                                else: 
                                    node['road'] = horExt['road']
                    draw_arrow_with_weight(result, (foundNode['x'], foundNode['y']), (node_x, node_y),foundNode['weight'])
                    foundNode['weight'] = 0
                else:
                    foundNode['weight'] = 0    
                    if(square_x_end!= cropped.shape[1]):
                        square_x_end += 20

                    #cv2.rectangle(result,(square_x_start,square_y_start), (square_x_end,square_y_end),(0,0,255),1)
                    #add search for turning points
                    crossed_x = 0
                    crossed_y = 0
                    crossed = False
                    for point in turningPointNodes:
                        x = int(point['point'][0])
                        y = int(point['point'][1])

                        crossed = findClose(square_x_start,square_x_end,square_y_start,square_y_end,x,y, node_x,node_y, resultBorder)
                        if(crossed):
                            crossed_x = x
                            crossed_y = y
                    if(crossed_x != 0):
                    #comeback
                        #print(f"{crossed} at point {crossed_x,crossed_y} with weight {foundNode['weight']}")
                        findClose_withoutBorder(square_x_start,square_x_end,square_y_start,square_y_end,crossed_x,crossed_y, node_x,node_y, resultBorder)
                        #cv2.circle(result,(crossed_x,crossed_y),dotRadius + 20,(0,0,255),1)
                        #print(f"drawn")  
                    if(foundNode['weight'] != 0 ):
                        marked = markPoint(foundNode['x'],foundNode['y'],turningPointNodes,node_x,node_y,'to')
                        if(marked):
                            edges.append({
                                'from': (int(foundNode['x']),int(foundNode['y'])),
                                'to': (int(node_x),int(node_y)),
                                'weight': int(foundNode['weight']) 
                            })
                            for point in turningPointNodes:
                                if(int(point['point'][0]) ==foundNode['x']) and int(point['point'][1] == foundNode['y']):
                                    node['from'] = (point['point'][0]),int(point['point'][1])
                                    if(horExt['road'] >= point['road'] and point['road'] != 0):
                                            horExt['road'] = point['road']
                                    else: 
                                            point['road'] = horExt['road']
                            draw_arrow_with_weight(result,(foundNode['x'], foundNode['y'] ),  (node_x, node_y), foundNode['weight'])
                        foundNode['weight'] = 0    
                    foundNode['weight'] = 0    
                foundNode['weight'] = 0
def searchForTurningPoints(x,y,direction):
    x_bound = 20
    y_bound = 20
    if(direction) == 'top':
        square_x_start = max(0 , x - x_bound)
        square_x_end = min(cropped.shape[1], x + x_bound)
        square_y_start = 0 
        square_y_end = y 
    elif(direction) == 'down':
        square_x_start = max(0 , x - 10)
        square_x_end = min(cropped.shape[1], x + 10)
        square_y_start = y
        square_y_end = cropped.shape[0]
    elif(direction) == 'right':
        square_x_start = x
        square_x_end = cropped.shape[1]
        square_y_start = max(0,y - 10)
        square_y_end = min(cropped.shape[0],y + 10)
    elif(direction) == 'left':
        square_x_start = 0
        square_x_end = x
        square_y_start = max(0,y - y_bound)
        square_y_end = min(cropped.shape[0],y + y_bound)
    else:
        return
    return square_x_start,square_x_end,square_y_start,square_y_end

print("=== Creating Edges from Vertical Extension Nodes to Turning Points ===")
#comebackHere 11/01
for verExt in verticalExtensionNodes:
    if verExt['direction'] == 'top' and (verExt['to'] == (0,0)):
        search_radius = (verExt['h'] // 2) + 5
        verExt_x = verExt['x']
        verExt_y = verExt['y']

        search_radius = (verExt['h'] // 2) + 5
        square_y_start = max(0, verExt_y - search_radius)
        square_y_end = min(cropped.shape[0], verExt_y + search_radius)

        square_x_start = 0
        square_x_end = verExt_x + 5

        crossed_x = 0
        crossed_y = 0
        crossed = False
        #cv2.rectangle(result,(square_x_start,square_y_start),(square_x_end,square_y_end),(0,0,255),1)
        for point in turningPointNodes:
            x = int(point['point'][0])
            y = int(point['point'][1])

            crossed = findClose(square_x_start,square_x_end,square_y_start,square_y_end,x,y, verExt_x,verExt_y, resultBorder)
            if(crossed):
                crossed_x = x
                crossed_y = y
        if(crossed_x != 0):
            findClose_withoutBorder(square_x_start,square_x_end,square_y_start,square_y_end,crossed_x,crossed_y, verExt_x,verExt_y, resultBorder)
            #cv2.circle(result,(crossed_x,crossed_y),dotRadius + 20,(0,0,255),1)
        if(foundNode['weight'] != 0): #found node weight = 0
            marked = markPoint(foundNode['x'],foundNode['y'],turningPointNodes,verExt_x,verExt_y,'from')
            if(marked):#foundnodewiehgt = 0
                edges.append({
                    'from': (int(verExt_x),int(verExt_y)),
                    'to': (int(foundNode['x']),int(foundNode['y'])),
                    'weight': int(foundNode['weight'])
                })
                for point in turningPointNodes:
                    if(int(point['point'][0]) ==foundNode['x']) and int(point['point'][1]) == foundNode['y']:
                            verExt['to'] = (int(point['point'][0]),int(point['point'][1]))
                            if(verExt['road'] >= point['road'] and point['road'] != 0):
                                verExt['road'] = point['road']
                            else: 
                                point['road'] = verExt['road']
                draw_arrow_with_weight(result,  (verExt_x,verExt_y), (foundNode['x'], foundNode['y']),foundNode['weight'])
            foundNode['weight'] = 0 
                # for point in turningPointNodes:
                #     if(crossed_x == point['point'][0] and point['from'] == (0,0) and point['to'] == (0,0)):
                #         cv2.circle(result,(crossed_x,crossed_y),dotRadius + 20,(0,0,255),1)
    elif(verExt['direction'] == 'bottom') and (verExt['to'] == (0,0)):
        verExt_x = verExt['x']
        verExt_y = verExt['y']
        crossed_x = 0
        crossed_y = 0 
        crossed = False

        search_radius = (verExt['h'] // 2) + 5
        square_y_start = max(0, verExt_y - search_radius)
        square_y_end = min(cropped.shape[0], verExt_y + search_radius)

        square_x_start = verExt_x
        square_x_end = cropped.shape[1]

        for point in turningPointNodes:
            x = int(point['point'][0])
            y = int(point['point'][1])
            crossed = findClose(square_x_start,square_x_end,square_y_start,square_y_end,x,y, verExt_x,verExt_y, resultBorder)

            if(crossed):
                crossed_x = x
                crossed_y = y
        if(crossed_x != 0):
                #print(f"{crossed} at point {crossed_x,crossed_y} with weight {foundNode['weight']}")
            findClose_withoutBorder(square_x_start,square_x_end,square_y_start,square_y_end,crossed_x,crossed_y, verExt_x,verExt_y, resultBorder)

        #cv2.rectangle(result,(square_x_start,square_y_start),(square_x_end,square_y_end),(0,0,255),1)
        if(foundNode['weight'] != 0):
            #cv2.circle(result,(foundNode['x'],foundNode['y']),dotRadius + 20,(0,0,255),1)
            #comeback
            marked = markPoint(foundNode['x'],foundNode['y'],turningPointNodes,verExt_x,verExt_y,'from')
            if(marked):
                edges.append({
                    'from': (int(verExt_x),int(verExt_y)),
                    'to': (int(foundNode['x']),int(foundNode['y'])),
                    'weight': int(foundNode['weight'])
                })
                for point in turningPointNodes:
                    if(int(point['point'][0]) ==foundNode['x']) and int(point['point'][1]) == foundNode['y']:
                        verExt['to'] = (int(point['point'][0]),int(point['point'][1]))
                        if(verExt['road'] >= point['road'] and point['road'] != 0):
                            verExt['road'] = point['road']
                        else: 
                            point['road'] = verExt['road']
                        #print(f"from: {point['from']} to {point['to']} on {point['point']}")
                draw_arrow_with_weight(result, (verExt_x,verExt_y), (foundNode['x'], foundNode['y'] ),foundNode['weight'])
                foundNode['weight'] = 0    
            foundNode['weight'] = 0 
        foundNode['weight'] = 0

print("=== Creating Edges from Horizontal Extension Nodes to Turning Points ===")
for horExt in horizontalExtensionNodes:
    if horExt['direction'] == 'left' and (horExt['to'] == (0,0)):
        search_radius = (verExt['h'] // 2) + 5
        horExt_x = horExt['x']
        horExt_y = horExt['y']
        crossed = False
        crossed_x = 0
        crossed_y = 0

        square_x_start = max(0, horExt_x - search_radius)
        square_x_end = min(cropped.shape[1], horExt_x + search_radius)  
        square_y_start = horExt_y
        square_y_end = cropped.shape[0]

        #cv2.rectangle(result,(square_x_start,square_y_start),(square_x_end,square_y_end),(0,0,255),1)
     #search down
        #cv2.rectangle(result,(square_x_start,square_y_start),(square_x_end,square_y_end),(0,0,255),1)
        for point in turningPointNodes:
            x = int(point['point'][0])
            y = int(point['point'][1])
            crossed = findClose(square_x_start,square_x_end,square_y_start,square_y_end,x,y, horExt_x,horExt_y, resultBorder)
            if(crossed): crossed_x, crossed_y = x,y 
        #comeback later 11/01 FIX THISSS PLEASE
        if(crossed_x != 0):
            findClose_withoutBorder(square_x_start,square_x_end,square_y_start,square_y_end,crossed_x,crossed_y, horExt_x,horExt_y, resultBorder)
        if(foundNode['weight'] != 0):
            marked = markPoint(foundNode['x'],foundNode['y'],turningPointNodes,horExt_x,horExt_y,'from')
            #print(f"{foundNode['x'],foundNode['y']}")
            #cv2.circle(result,(foundNode['x'],foundNode['y']),dotRadius + 20,(0,0,255),1)
            if(marked):
                #cv2.circle(result,(foundNode['x'],foundNode['y']),dotRadius + 20, (0,0,255),1)
                edges.append({
                    'from': (int(horExt_x),int(horExt_y)),
                    'to': (int(foundNode['x']),int(foundNode['y'])),
                    'weight': int(foundNode['weight'])
                })
                for point in turningPointNodes:
                    if(int(point['point'][0]) ==foundNode['x']) and int(point['point'][1] == foundNode['y']):
                            horExt['to'] = (int(point['point'][0]),int(point['point'][1]))
                            if(horExt['road'] >= point['road'] and point['road'] != 0):
                                horExt['road'] = point['road']
                            else: 
                                point['road'] = horExt['road']
                    # cv2.circle(result, (483,404),dotRadius + 20,(255,0,0),1)
                    # cv2.circle(result, (389,46),dotRadius + 20,(0,255,0),1)
                    # cv2.circle(result, (496,159),dotRadius + 20,(0,0,255),1)
                    # cv2.circle(result, (horExt_x,horExt_y),dotRadius + 20,(0,255,255),1)
                    draw_arrow_with_weight(result, (horExt_x,horExt_y), (foundNode['x'], foundNode['y'] ),foundNode['weight'])
                foundNode['weight'] = 0    
            foundNode['weight'] = 0 
    elif horExt['direction'] == 'right' and (horExt['to'] == (0,0)):
        search_radius = (verExt['h'] // 2) + 5
        horExt_x = horExt['x']
        horExt_y = horExt['y']

        square_x_start = max(0, horExt_x - search_radius)
        square_x_end = min(cropped.shape[1], horExt_x + search_radius)
        square_y_start = 0
        square_y_end = horExt_y + 5

        crossed = False
        crossed_x = 0
        crossed_y = 0

        #cv2.rectangle(result,(square_x_start,square_y_start),(square_x_end,square_y_end),(0,0,255),1) 
        for point in turningPointNodes:
            x = int(point['point'][0])
            y = int(point['point'][1])
            crossed = findClose(square_x_start,square_x_end,square_y_start,square_y_end,x,y, horExt_x,horExt_y, resultBorder)
            if(crossed):
                crossed_x = x
                crossed_y = y
        if(crossed_x != 0):
            findClose_withoutBorder(square_x_start,square_x_end,square_y_start,square_y_end,crossed_x,crossed_y,horExt_x,horExt_y,resultBorder)
        #cv2.circle(result,(foundNode['x'],foundNode['y']),dotRadius + 20,(0,0,255),1)
        if(foundNode['weight'] != 0):
            marked = markPoint(foundNode['x'],foundNode['y'],turningPointNodes,horExt_x,horExt_y,'from')
            if(marked):
                edges.append({
                    'from': (int(horExt_x),int(horExt_y)),
                    'to': (int(foundNode['x']),int(foundNode['y'])),
                    'weight': int(foundNode['weight'])
                })
                for point in turningPointNodes:
                    if(int(point['point'][0]) ==foundNode['x']) and int(point['point'][1] == foundNode['y']):
                            horExt['to'] = (int(point['point'][0]),int(point['point'][1]))
                            if(horExt['road'] >= point['road'] and point['road'] != 0):
                                horExt['road'] = point['road']
                            else: 
                                point['road'] = horExt['road']
                    draw_arrow_with_weight(result, (horExt_x,horExt_y), (foundNode['x'], foundNode['y'] ),foundNode['weight'])
                foundNode['weight'] = 0    
            foundNode['weight'] = 0 

def getTurningPointBoundary(node_x,node_y,direction):
    search_radius = 40
    if(direction == 'right'):
        square_y_start = max(0, node_y - search_radius)
        square_y_end = min(cropped.shape[0], node_y + search_radius)

        square_x_start = node_x
        square_x_end = cropped.shape[1]
    elif(direction == 'left'):
        square_y_start = max(0, node_y - search_radius)
        square_y_end = min(cropped.shape[0], node_y + search_radius)

        square_x_start = 0
        square_x_end = node_x
    elif(direction == 'top'):
        square_x_start = max(node_x - search_radius, 0)
        square_x_end = min(node_x + search_radius, width)
        square_y_start = 0
        square_y_end  = node_y - 5
    elif(direction == 'down'):
        square_y_start = node_y + 5
        square_y_end = cropped.shape[1]

        square_x_start = max(node_x - search_radius, 0)
        square_x_end = min(node_x + search_radius, width)
    return square_x_start,square_x_end,square_y_start,square_y_end



counterO = 0
counterI = 0
print("\n=== Creating Edges from Turning Points ===")
for i, node in enumerate(turningPointNodes):
    print(i)
    if(node['type'] == 'outer'):
        counterO += 1
        print(f"Outer number {counterO}")
    if(node['type'] == 'inner'):
        counterI += 1
        print(f"inner number {counterI}")

for i, node in enumerate(turningPointNodes):
    node_x = node['point'][0]
    node_y = node['point'][1]
    crossed_x = 0 
    crossed_y = 0
    crossed = False
    check = []
    directionCheck = []
    directions = ['top','down','left','right']
    if(node['type'] == 'outer'):
        if(node['from'] == (0,0)) and node['to'] == (0,0): 
            directionCheck.append('to')
            directionCheck.append('from')
        elif(node['to'] == (0,0)) and (node['from'] != (0,0)):
            directionCheck.append('from')
        elif(node['from'] == (0,0)) and node['to'] != (0,0):
            directionCheck.append('to')

        if(node['top'] == False):
            check.append('top')
        if(node['left'] == False):
            check.append('left')
        if(node['right'] == False):
            check.append('right')
        if(node['down'] == False):
            check.append('down')
        for j in range(len(directionCheck)):
            for k in range (len(check)):
                square_x_start,square_x_end,square_y_start,square_y_end = getTurningPointBoundary(node_x,node_y,check[k])
                # cv2.rectangle(result,(square_x_start,square_y_start),(square_x_end,square_y_end),(0,0,255),1)
                for point in turningPointNodes:
                    if(point['type'] == 'outer'):
                        x = point['point'][0]
                        y = point['point'][1]
                        crossed = findClose(square_x_start,square_x_end,square_y_start,square_y_end,x,y, node_x,node_y, resultBorder)
                        if(crossed):
                            crossed_x = x
                            crossed_y = y
                if(crossed_x != 0):
                    findClose_withoutBorder(square_x_start,square_x_end,square_y_start,square_y_end,crossed_x,crossed_y, node_x,node_y, resultBorder)
                if(foundNode['weight'] != 0):
                    marked = markPoint(foundNode['x'],foundNode['y'],turningPointNodes,node_x,node_y,directionCheck[j])
                    if(marked):
                        if(directionCheck[j] != 'from'): #from = (0,0)
                            edges.append({
                                'from': (int(foundNode['x']),int(foundNode['y'])),
                                'to': (int(node_x),int(node_y)),
                                'weight': int(foundNode['weight'])
                            })
                            node['from'] = (foundNode['x'],foundNode['y'])
                            draw_arrow_with_weight(result,(foundNode['x'],foundNode['y']),(node_x,node_y),foundNode['weight'],(0,255,0),1)
                            foundNode['weight'] = 0
                        elif(directionCheck[j] != 'to'):
                            edges.append({
                                'from': (int(node_x),int(node_y)),
                                'to': (int(foundNode['x']),int(foundNode['y'])),
                                'weight': int(foundNode['weight'])
                            })
                            node['to'] = (foundNode['x'],foundNode['y'])
                            draw_arrow_with_weight(result,(node_x,node_y),(foundNode['x'],foundNode['y']),foundNode['weight'],(0,255,0),1)
                            foundNode['weight'] = 0
                        break
                foundNode['weight'] = 0
    elif(node['type'] == 'inner'):
        if(node['from'] == (0,0)) and node['to'] == (0,0): 
            directionCheck.append('to')
            directionCheck.append('from')
        elif(node['to'] == (0,0)) and (node['from'] != (0,0)):
            directionCheck.append('from')
        elif(node['from'] == (0,0)) and node['to'] != (0,0):
            directionCheck.append('to')

        if(node['top'] == False):
            check.append('top')
        if(node['left'] == False):
            check.append('left')
        if(node['right'] == False):
            check.append('right')
        if(node['down'] == False):
            check.append('down')
        for j in range (len(directionCheck)):
            for k in range (len(check)):
                print(f"Node No. {i} {node_x,node_y} has {len(directionCheck)} checks with {directionCheck[j]} at {check[k]}")
                square_x_start,square_x_end,square_y_start,square_y_end = getTurningPointBoundary(node_x,node_y,check[k])
                #cv2.rectangle(result,(square_x_start,square_y_start),(square_x_end,square_y_end),(0,0,255),1)
                for point in turningPointNodes:
                    if(point['type'] == 'inner'):
                        x = point['point'][0]
                        y = point['point'][1]
                        crossed = findClose(square_x_start,square_x_end,square_y_start,square_y_end,x,y, node_x,node_y, resultBorder)
                        if(crossed):
                            crossed_x = x
                            crossed_y = y
                if(crossed_x != 0):
                    findClose_withoutBorder(square_x_start,square_x_end,square_y_start,square_y_end,crossed_x,crossed_y, node_x,node_y, resultBorder)
                if(foundNode['weight'] != 0):
                    marked = markPoint(foundNode['x'],foundNode['y'],turningPointNodes,node_x,node_y,directionCheck[j])
                    if(marked):
                        if(directionCheck[j] != 'from'): #from = (0,0)
                            edges.append({
                                'from': (int(foundNode['x']),int(foundNode['y'])),
                                'to': (int(node_x),int(node_y)),
                                'weight': int(foundNode['weight'])
                            })
                            node['from'] = (foundNode['x'],foundNode['y'])
                            print("drawn from")
                            draw_arrow_with_weight(result,(foundNode['x'],foundNode['y']),(node_x,node_y),foundNode['weight'],(0,255,0),1)
                            foundNode['weight'] = 0
                        elif(directionCheck[j] != 'to'):
                            edges.append({
                                'from': (int(node_x),int(node_y)),
                                'to': (int(foundNode['x']),int(foundNode['y'])),
                                'weight': int(foundNode['weight'])
                            })
                            node['to'] = (foundNode['x'],foundNode['y'])
                            print("drawn to")
                            draw_arrow_with_weight(result,(node_x,node_y),(foundNode['x'],foundNode['y']),foundNode['weight'],(0,255,0),1)
                            foundNode['weight'] = 0
                        break
                foundNode['weight'] = 0


edgeData = []
for edge in edges:
    edgeData.append(edge)

def convert_tuples(obj):
    if isinstance(obj, tuple):
        return list(obj)
    if isinstance(obj, list):
        return [convert_tuples(x) for x in obj]
    if isinstance(obj, dict):
        return {k: convert_tuples(v) for k, v in obj.items()}
    return obj

edgeData = convert_tuples(edgeData)

import json




("cropped.jpg", cropped)
cv2.imwrite("nodes.jpg", result)

print(f"\n✓ Output saved to 'nodes.jpg'")
print(f"✓ Center nodes: {len(centerBlock)}")
print(f"✓ Horizontal extension nodes: {len(horizontalExtensionNodes)}")
print(f"✓ Vertical extension nodes: {len(verticalExtensionNodes)}")
print(f"✓ Intersection nodes: {len(intersectionNodes)}")
print(f"✓ Total edges: {len(edges)}")
print(f"✓ Number of yellow points: {len(yellowBlockCoord)}")
print(f"✓ Number of turning points Nodes: {len(turningPointNodes)}")
print(f"✓ Number of Roads: {len(roads)}")
print(f"✓ Number of visitedVBlocks: {len(visitedBlocks)}")
print(f"✓ Number of turning point nodes: {len(turningPointNodes)}")

spots = []
neighbors = []
mapping = []
visited = []
j = 0
for i, edge in enumerate(edges):
    if edge['from'] not in visited:
        visited.append(edge['from'])
        neighbors.append([edge['to']])
        mapping.append((edge['from'],j))
        spots.append({
            'x': edge['from'][0],
            'y': edge['from'][1]
        })
        j+=1
    else:
        for m in mapping:
            if m[0] == edge['from']:
                neighbors[m[1]].append(edge['to'])

print("---------- spots -------------")
for s in spots:
    print(s)

print("---------- neighbour -------------")
for n in neighbors:
    for i, x in enumerate(n):
        for m in mapping:
            if m[0] == x:
                n[i] = m[1]
    print(n)

print("---------- mapping -------------")
for m in mapping:
    print(f"Spot: {m[0]}")
    print(f"Index: {m[1]}")


visited.clear()
# for edge in edges:
#     if edge['from'][0] != spots['x'] and edge['from'][1] != spots['y']:
#         spots.append({
#             "x": edge['from'][0],
#             "y": edge['from'][1]
#         })
#         testEdge.append({
#             'spot': edge['from']
#             })
#     else:


# spots = convert_tuples(spots)
# neighbours = convert_tuples(neighbours)
data = {
    'spots': spots,
    'neighbors': neighbors
}

with open("graph.json", "w") as f:
    json.dump(data, f, indent=4)

# with open("graph.json", "w") as f:
#     json.dump(edgeData, f, indent=4)







# # Creating contour to track red color 
# contours, hierarchy = cv2.findContours(red_mask, 
#                                     cv2.RETR_TREE, 
#                                     cv2.CHAIN_APPROX_SIMPLE) 

# for pic, contour in enumerate(contours): 
#     area = cv2.contourArea(contour) 
#     if(area > 300): 
#         x, y, w, h = cv2.boundingRect(contour) 
#         imageFrame = cv2.rectangle(imageFrame, (x, y), 
#                                 (x + w, y + h), 
#                                 (0, 0, 255), 2) 
        
#         cv2.putText(imageFrame, "Red Colour", (x, y), 
#                     cv2.FONT_HERSHEY_SIMPLEX, 1.0, 
#                     (0, 0, 255))

# gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# blur = cv2.GaussianBlur(gray, (5, 5), 0)

# edges = cv2.Canny(blur, 50, 150)

# lines = cv2.HoughLinesP(
#     edges,
#     rho=1,
#     theta=np.pi / 180,
#     threshold=150,
#     minLineLength=200,
#     maxLineGap=10
# )

# foam_lines = []

# if lines is not None:
#     for line in lines:
#         x1, y1, x2, y2 = line[0]

#         # Vertical line
#         if abs(x1 - x2) < 10:
#             foam_lines.append((x1, y1, x2, y2))

#         # Horizontal line
#         elif abs(y1 - y2) < 10:
#             foam_lines.append((x1, y1, x2, y2))

# for x1, y1, x2, y2 in foam_lines:
#     cv2.line(img, (x1, y1), (x2, y2), (0, 0, 255), 3)
# cv2.imshow("original", original)
# cv2.imshow("result red", resultRed)
# cv2.imshow("yellowBlocks", maskYel)
#imgT = cv2.imread("cropped.jpg", 0)

cv2.imwrite("test.jpg", resultWhite)
cv2.imwrite("test1.jpg", resultRed)
cv2.imwrite("test2.jpg", resultYellow)
cv2.imwrite("test3.jpg", resultBorder)


# cv2.imwrite("yellowMask.jpg", resultYellow)

cv2.waitKey(1)
cv2.destroyAllWindows()
