
///////////////////////////////////////////////////////////////////////
//
// triangles.cpp
//
///////////////////////////////////////////////////////////////////////

using namespace std;

#include "vgl.h"
#include "LoadShaders.h"
#include "glm\glm.hpp"
#include "glm\gtc\matrix_transform.hpp"
#include "glm\gtx\rotate_vector.hpp"
#include <iostream>
#include <fstream>
#include <string>
#include <vector>

enum VAO_IDs { Triangles, NumVAOs };
enum Buffer_IDs { ArrayBuffer, NumBuffers };
enum Attrib_IDs { vPosition = 0 };

GLuint VAOs[NumVAOs];
GLuint Buffers[NumBuffers];
GLuint location;
GLuint cam_mat_location;
GLuint proj_mat_location;

const GLuint NumVertices = 16 + 24461 + 529;
const GLuint NumFaces = 48918 + 1024;


//Player motion speed and key controls
float height = 0.8f;
float yaw_speed = 0.1f;
float travel_speed = 60.0f;
float mouse_sensitivity = 0.01f;

//Used for tracking mouse cursor position on screen
int x0 = 0;
int y_0 = 0;

//Transformation matrices and camera vectors
glm::mat4 model_view;
glm::vec3 cam_pos = glm::vec3(0.0f, 0.0f, height);
glm::vec3 forward_vector = glm::vec3(1, 1, 0);
glm::vec3 up_vector = glm::vec3(0, 0, 1);
glm::vec3 side_vector = glm::cross(up_vector, forward_vector);

//Used to measure time between two frames
int oldTimeSinceStart = 0;
int deltaTime;

//Creating and rendering bunch of objects on the scene to interact with
const int Num_Obstacles = 100;
float obstacle_data[Num_Obstacles][3];

GLubyte GLOBAL_FACES[1024][3];

void loadFile(const char * filename, GLfloat vertices[][3], GLubyte faces[][3]) {
	//do during init()

	int vArrSize = 0;
	int uvArrSize = 0;
	int viArrSize = 0;
	int uviArrSize = 0;
	std::fstream fin(filename, std::ios::in);

	//open file
	if (fin.is_open()) {

		std::string line;

		//find size of arrays
		while (getline(fin, line)) {
			if (line[0] == 'v') {
				if (line[1] == ' ')
					vArrSize++;
				else if (line[1] == 't')
					uvArrSize++;
			}
			if (line[0] == 'f') {
				viArrSize++;
			}
		}
		uviArrSize = viArrSize;

		//go back to begining of file
		fin.clear();
		fin.seekg(0, std::ios::beg);


		//create indicies
		int vArri = 0;
		int uvArri = 0;
		int viArri = 0;
		int uviArri = 0;

		//input values
		while (getline(fin, line)) {

			std::vector<std::string> row;
			std::vector<std::string> fRow;
			std::vector<std::string> ufRow;

			//read vector
			if (line[0] == 'v' && line[1] == ' ') {
				size_t index = 2; //start after identifier
				std::string field;
				while (index < line.size() + 1) {
					if (line[index] == ' ' || index == line.size()) { //read until end of field or line
						if (!field.empty() && field[0] != ' ') { //if field has data
							row.push_back(move(field));
						}
					}
					else {
						field += line[index]; //add character to field
					}
					index++;
				}


				//load into vector instead[]
				vertices[vArri + 8][0] = stof(row[0]);
				vertices[vArri + 8][1] = stof(row[1]);
				vertices[vArri + 8][2] = stof(row[2]);
				vArri++;				

			}
			//vector + UV faces
			//Swap back and forth between arrays
			if (line[0] == 'f') {
				size_t index = 2; //start after identifier
				std::string field;
				while (index < line.size() + 1) {
					if (line[index] == '/') { //read until end of Vector face
						if (!field.empty() && field[0] != ' ') { //if field has data
							fRow.push_back(move(field));
						}
					}
					else if (line[index] == ' ' || index == line.size()) { //read until end of texture face
						if (!field.empty() && field[0] != ' ') { //if field has data
							ufRow.push_back(move(field));
						}
					}
					else {
						field += line[index]; //add character to field
					}
					index++;
				}
				int temp;
				temp = stoi(ufRow[0]);
				faces[uviArri][0] = temp; 
				temp = stoi(ufRow[1]);
				faces[uviArri][1] = temp;
				temp = stoi(ufRow[2]);
				faces[uviArri][2] = temp;


				uviArri++;
			}
		}
		fin.close();
	}
	else {
		//TODO: return error
		return;
	}
	return;
}

//Helper function to generate a random float number within a range
float randomFloat(float a, float b)
{
	float random = ((float)rand()) / (float)RAND_MAX;
	float diff = b - a;
	float r = random * diff;
	return a + r;
}

// inititializing buffers, coordinates, setting up pipeline, etc.
void init(void)
{
	glEnable(GL_DEPTH_TEST);

	//Randomizing the position and scale of obstacles
	for (int i = 0; i < Num_Obstacles; i++)
	{
		obstacle_data[i][0] = randomFloat(-50, 50); //X
		obstacle_data[i][1] = randomFloat(-50, 50); //Y
		obstacle_data[i][2] = randomFloat(0.1, 10.0); //Scale
	}

	ShaderInfo shaders[] = {
		{ GL_VERTEX_SHADER, "triangles.vert" },
	{ GL_FRAGMENT_SHADER, "triangles.frag" },
	{ GL_NONE, NULL }
	};

	GLuint program = LoadShaders(shaders);
	glUseProgram(program);	//My Pipeline is set up

	GLfloat vertices[NumVertices][3] = {
	{ -100.0, -100.0, 0.0 }, //Ground and a sky
	{ 100.0, -100.0, 0.0 },
	{ 100.0, 100.0, 0.0 },
	{ -100.0, 100.0, 0.0 },
	{ -100.0, -100.0, 10.0 },
	{ 100.0, -100.0, 10.0 },
	{ 100.0, 100.0, 10.0 },
	{ -100.0, 100.0, 10.0 }

	//[8]is where the rest of the models go
	};

	GLfloat colorData[NumVertices][3] = {
	{ 0,1,0 }, // color for plane vertices
	{ 0,1,0 },
	{ 0,1,0 },
	{ 0,1,0 },
	{ 0,0,1 },
	{ 0,0,1 },
	{ 0,0,1 },
	{ 0,0,1 },
	};

	//	{1, 1, 1} //white for obsticales
	std::fill_n(colorData[8], NumVertices - 8, 1);

	GLubyte faces[1024][3];
	loadFile("Teapot.obj", vertices, faces);
	for (int i = 0; i < 1024; i++) {
		for (int j = 0; j < 3; j++) {
			GLOBAL_FACES[i][j] = faces[i][j];
		}
	}

	glGenBuffers(2, Buffers);
	glBindBuffer(GL_ARRAY_BUFFER, Buffers[0]);
	glBufferData(GL_ARRAY_BUFFER, sizeof(vertices), vertices, GL_STATIC_DRAW);
	glBindAttribLocation(program, 0, "vPosition");
	glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 0, BUFFER_OFFSET(0));
	glEnableVertexAttribArray(0);

	glBindBuffer(GL_ARRAY_BUFFER, Buffers[1]);
	glBufferData(GL_ARRAY_BUFFER, sizeof(colorData), colorData, GL_STATIC_DRAW);
	glBindAttribLocation(program, 1, "vertexColor");
	glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 0, BUFFER_OFFSET(0));
	glEnableVertexAttribArray(1);

	location = glGetUniformLocation(program, "model_matrix");
	cam_mat_location = glGetUniformLocation(program, "camera_matrix");
	proj_mat_location = glGetUniformLocation(program, "projection_matrix");
}

//Helper function to draw a cube
void drawObstacle(float scale, GLubyte faces[][3])
{
	//model_view = glm::scale(model_view, glm::vec3(scale, scale, scale));
	model_view = glm::scale(model_view, glm::vec3(0.1, 0.1, 0.1));
	glUniformMatrix4fv(location, 1, GL_FALSE, &model_view[0][0]);

	for (int i = 0; i < 1024; i++) {
		glDrawElements(GL_TRIANGLES, 3, GL_UNSIGNED_BYTE, faces[i]);
	}
}

//Renders level
void draw_level()
{
	//Drawing the floor and the sky
	GLubyte ground[] = { 1,2,3,4 };
	GLubyte sky[] = { 5,6,7,8 };
	glDrawElements(GL_QUADS, 4, GL_UNSIGNED_BYTE, ground);
	glDrawElements(GL_QUADS, 4, GL_UNSIGNED_BYTE, sky);

	//Rendering obstacles obstacles
	for (int i = 0; i < 1; i++)
	{
		model_view = glm::translate(model_view, glm::vec3(obstacle_data[i][0], obstacle_data[i][1], 0.0));
		glUniformMatrix4fv(location, 1, GL_FALSE, &model_view[0][0]);
		drawObstacle(obstacle_data[i][2], GLOBAL_FACES);
		model_view = glm::mat4(1.0);
	}
}

//---------------------------------------------------------------------
//
// display
//
void display(void)
{
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

	//Camera controls
	model_view = glm::mat4(1.0);
	up_vector = glm::normalize(up_vector);
	forward_vector = glm::normalize(forward_vector);
	glm::vec3 look_at = glm::vec3(cam_pos.x + forward_vector.x, cam_pos.y + forward_vector.y, height);
	glm::mat4 camera_matrix = glm::lookAt(glm::vec3(cam_pos.x, cam_pos.y, cam_pos.z), glm::vec3(look_at.x, look_at.y, look_at.z), up_vector);
	glUniformMatrix4fv(cam_mat_location, 1, GL_FALSE, &camera_matrix[0][0]);
	glm::mat4 proj_matrix = glm::frustum(-0.01f, +0.01f, -0.01f, +0.01f, 0.01f, 100.0f);
	glUniformMatrix4fv(proj_mat_location, 1, GL_FALSE, &proj_matrix[0][0]);


	draw_level();
	glFlush();
}


void keyboard(unsigned char key, int x, int y)
{
	if (key == 'a')
	{
		cam_pos += glm::cross(up_vector, forward_vector) * travel_speed * ((float)deltaTime) / 1000.0f;
	}
	if (key == 'd')
	{
		cam_pos -= glm::cross(up_vector, forward_vector) * travel_speed * ((float)deltaTime) / 1000.0f;
	}
	if (key == 'w')
	{
		cam_pos += forward_vector * travel_speed * ((float)deltaTime) / 1000.0f;
	}
	if (key == 's')
	{
		cam_pos -= forward_vector * travel_speed * ((float)deltaTime) / 1000.0f;
	}
}

void mouse(int x, int y)
{
	int delta_x = x - x0;
	forward_vector = glm::rotate(forward_vector, -delta_x * mouse_sensitivity, up_vector);
	side_vector = glm::cross(up_vector, forward_vector);
	x0 = x;

}

void idle()
{
	//Calculate momentum
	int timeSinceStart = glutGet(GLUT_ELAPSED_TIME);
	deltaTime = timeSinceStart - oldTimeSinceStart;
	oldTimeSinceStart = timeSinceStart;

	glutPostRedisplay();
}

//---------------------------------------------------------------------
//
// main
//

int
main(int argc, char** argv)
{
	glutInit(&argc, argv);
	glutInitDisplayMode(GLUT_RGBA);
	glutInitWindowSize(1024, 1024);
	glutCreateWindow("Camera and Projection");

	glewInit();	//Initializes the glew and prepares the drawing pipeline.

	init();

	glutDisplayFunc(display);

	glutKeyboardFunc(keyboard);

	glutIdleFunc(idle);

	glutPassiveMotionFunc(mouse);

	glutMainLoop();



}
