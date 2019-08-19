#include "maze.h"
#include <iostream>
//receves maze, start position, end position, link to sollution array
//recursive function looks for path toward start using changing ending coordinates.
//if no more valid directions, return 0
//if start is found, return distance to start from this position (end) and adds coordinate to path[distance]
int runPath(Maze& theMaze, const Coord& start, Coord& position, Coord path[], int& count);
int findPath(Maze& theMaze, const Coord& start, const Coord& end, Coord path[]);

int findPath(Maze& theMaze, const Coord& start, const Coord& end, Coord path[]) {
	Coord position(end.x, end.y);
	int count = 0;
	return runPath(theMaze, start, position, path, count);
}
int runPath(Maze& theMaze, const Coord& start, Coord& position, Coord path[], int& count) {
	//check if maze if solved
	if (start == position) {
		//mark first step in path
		theMaze.mark(position);
		path[0] = Coord(start.x, start.y);
		return 1;
	} else {
		//check North path
		if ((position.y - 1) >= 0) {
			if (theMaze.isEmpty(Coord(position.x, position.y - 1))) {
				//mark and follow path
				theMaze.mark(position);
				position.y -= 1;
				count = runPath(theMaze, start, position, path, count);
				position.y += 1;
				//if path comes back with solution add current position to array
				if (count > 0) {
					path[count] = Coord(position.x, position.y);
					return count + 1;
				} else {
					theMaze.unMark(position);
				}
			}
		}
		//check West path
		if ((position.x - 1) >= 0) {
			if (theMaze.isEmpty(Coord(position.x - 1, position.y))) {
				//mark and follow path
				theMaze.mark(position);
				position.x -= 1;
				count = runPath(theMaze, start, position, path, count);
				//if path comes back with solution add current position to array
				position.x += 1;
				if (count > 0) {
					path[count] = Coord(position.x, position.y);
					return count + 1;
				} else {
					theMaze.unMark(position);
				}
			}
		}
		//check East path
		if ((position.x + 1) < theMaze.width()) {
			if (theMaze.isEmpty(Coord(position.x + 1, position.y))) {
				//mark and follow path
				theMaze.mark(position);
				position.x += 1;
				count = runPath(theMaze, start, position, path, count);
				position.x -= 1;
				//if path comes back with solution add current position to array
				if (count > 0) {
					path[count] = Coord(position.x, position.y);
					return count + 1;
				} else {
					theMaze.unMark(position);
				}
			}
		}
		//check South path
		if ((position.y + 1) < theMaze.height()) {
			if (theMaze.isEmpty(Coord(position.x, position.y + 1))) {
				//mark and follow path
				theMaze.mark(position);
				position.y += 1;
				count = runPath(theMaze, start, position, path, count);
				position.y -= 1;
				//if path comes back with solution add current position to array
				if (count > 0) {
					path[count] = Coord(position.x, position.y);
					return count + 1;
				} else {
					theMaze.unMark(position);
				}
			}
		}
	}
	//no posible paths, return 0
	return 0;
}
