#ifndef MAZE_H
#define MAZE_H
#include <string>
using namespace std;

struct Coord{
	int x;
	int y;
	Coord(){
	}
	Coord(int a, int b){
		x=a;
		y=b;
	}
	void set(int a, int b){
		x=a;
		y=b;
	}
};
bool operator==(const Coord& left, const Coord& right);
bool operator!=(const Coord& left, const Coord& right);
const int MARKED = 2;
const int WALL = 1;
const int EMPTY = 0;
class Maze{
	int** theMaze_;
	int width_;
	int height_;
public:
	Maze(string mazeFile);
	//returns true if co is a wall
	bool isWall(const Coord& co) const;
	//returns true if co is empty (not wall, unmarked)
	bool isEmpty(const Coord& co) const;
	//returns true if co is marked
	bool isMarked(const Coord& co) const;
	//returns width of the maze
	int width() const{return width_;}
	//returns height of the maze
	int height() const{return height_;}
	//marks the coodinate co, returns true if successful
	bool mark(const Coord& co);
	//unmarks the coordinate co, returns true if successful
	bool unMark(const Coord& co);
	//prints the maze
	void print() const;
	~Maze();
};

#endif
