
///////////////////////////////////////////////////////////////////////
//
// triangles.cpp
//
///////////////////////////////////////////////////////////////////////

using namespace std;

#include <iostream>
#include "vgl.h"
#include "LoadShaders.h"
#include "glm\glm.hpp"
#include "glm\gtc\matrix_transform.hpp"

enum VAO_IDs { Triangles, NumVAOs };
enum Buffer_IDs { ArrayBuffer, NumBuffers };
enum Attrib_IDs { vPosition = 0 };

GLuint VAOs[NumVAOs];
GLuint Buffers[NumBuffers];
GLuint location;

GLuint cam_matrix_location_in_vram;
GLuint projection_matrix_location_in_vram;

const GLuint NumVertices = 8;

float translate_value = 0;
float rotate_value = 0;
float alpha = 0;

float camz = 0.5f;
float camx = 0.0f;
float camy = 0.0f;

void drawCube()
{
	GLubyte top_face[] = { 0, 1, 2, 3 };
	GLubyte bottom_face[] = { 4, 5, 6, 7 };
	GLubyte left_face[] = { 0, 4, 7, 3 };
	GLubyte right_face[] = { 1, 5, 6, 2 };
	GLubyte front_face[] = { 2, 3, 7, 6 };
	GLubyte back_face[] = { 2, 3, 7, 6 };
	glDrawElements(GL_LINE_LOOP, 4, GL_UNSIGNED_BYTE, top_face);
	glDrawElements(GL_LINE_LOOP, 4, GL_UNSIGNED_BYTE, bottom_face);
	glDrawElements(GL_LINE_LOOP, 4, GL_UNSIGNED_BYTE, left_face);
	glDrawElements(GL_LINE_LOOP, 4, GL_UNSIGNED_BYTE, right_face);
	glDrawElements(GL_LINE_LOOP, 4, GL_UNSIGNED_BYTE, front_face);
	glDrawElements(GL_LINE_LOOP, 4, GL_UNSIGNED_BYTE, back_face);
}


//---------------------------------------------------------------------
// Setting up our pipeline and preparing to draw 
void init(void)
{
	//Defining the name of our shader files
	ShaderInfo shaders[] = {
		{ GL_VERTEX_SHADER, "triangles.vert" },
		{ GL_FRAGMENT_SHADER, "triangles.frag" },
		{ GL_NONE, NULL }
	};

	//Loading and attaching shaders to our pipeline
	GLuint program = LoadShaders(shaders);
	glUseProgram(program);	//My Pipeline is set up

	// Coordinates of vertices (Square)
	GLfloat vertices[NumVertices][3] = {
		{ -0.45, 0.45, -0.45 },
		{ 0.45, 0.45, -0.45 },
		{ 0.45, -0.45, -0.45 },
		{-0.45, -0.45, -0.45 },
		

		{ -0.45, 0.45, 0.45 },
		{ 0.45 , 0.45 , 0.45 },
		{ 0.45, -0.45, 0.45 },
		{ -0.45, -0.45, 0.45 },
	};

	// Colors for vertices in {R, G, B} mode
	GLfloat colorData[NumVertices][3] = {
		{ 1,0,0 }, //Red
		{ 0,1,0 }, //Green
		{ 0,0,1 }, //Blue
		{ 1,1,1 },  //White
		{ 1,0,0 }, //Red
		{ 0,1,0 }, //Green
		{ 0,0,1 }, //Blue
		{ 1,1,1 } //White
	};

	glGenBuffers(2, Buffers);
	glBindBuffer(GL_ARRAY_BUFFER, Buffers[0]);
	glBufferData(GL_ARRAY_BUFFER, sizeof(vertices), vertices, GL_STATIC_DRAW);
	glBindAttribLocation(program, 0, "vPosition");
	glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 0, BUFFER_OFFSET(0));
	glEnableVertexAttribArray(0);

	glBindBuffer(GL_ARRAY_BUFFER, Buffers[1]);
	glBufferData(GL_ARRAY_BUFFER, sizeof(colorData), colorData, GL_STATIC_DRAW);
	glBindAttribLocation(program, 1, "vertexColor");
	glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 0, BUFFER_OFFSET(0));
	glEnableVertexAttribArray(1);


	//Retrieving the location of the matrices from VRAM and storing them inside these variables
	location = glGetUniformLocation(program, "model_matrix");
	cam_matrix_location_in_vram = glGetUniformLocation(program, "camera_matrix");
	projection_matrix_location_in_vram = glGetUniformLocation(program, "projection_matrix");
	
}


//---------------------------------------------------------------------
//
// The following function is named display function in OpenGL. The task of the display function is to render the scene
// The name of the display function is defined by the user (in this case, it is called drawScene).
// In the main function, you will need to register the name of the display function in main() method.
//This is done by using glutDisplayFunc function. Look at the main method
void drawScene(void)
{
	//Clear the screen and preparing to draw
	glClear(GL_COLOR_BUFFER_BIT);

	glm::mat4  model_view = glm::rotate(glm::mat4(1.0), 0.0f, glm::vec3(0.0f, 0.0f, 1.0f));

	model_view = glm::scale(model_view, glm::vec3(1.0, 1.0, 1.0));

	glUniformMatrix4fv(location, 1, GL_FALSE, &model_view[0][0]);

	//Setting up camera matrix and initialize the camera_matrix in VRAM	
	glm::mat4 camera_matrix = glm::lookAt(glm::vec3(camx, camy, camz), glm::vec3(0, 0, 0), glm::vec3(0, 1, 0));
	glUniformMatrix4fv(cam_matrix_location_in_vram, 1, GL_FALSE, &camera_matrix[0][0]);

	//Setting up projection matrix and initialize the projection_matrix in VRAM
	glm::mat4 projection_matrix = glm::frustum(-0.01f, +0.01f, -0.02f, +0.02f, 0.01f, 10.0f);
	glUniformMatrix4fv(projection_matrix_location_in_vram, 1, GL_FALSE, &projection_matrix[0][0]);


	//Draw Sun
	glUniformMatrix4fv(location, 1, GL_FALSE, &model_view[0][0]);
	drawCube();
	//glDrawArrays(GL_LINE, 0, 8);

	glFlush();

}

//This function is called "idle" function. This function is invoked by OpenGL ever frame (60 frame per second).
//This function is used for animation, since you can animate your scene for every frame.
//You will need to register the name of your idle function to OpenGL (It is "runEveryFrame" in here).
//The registration happens in the main() function using glutIdleFunc(runEveryFrame) function.


void keyboard(unsigned char key, int x, int y)
{
	if (key == '+')
	{
		camz -= 0.01f;
	}
	if (key == '-')
	{
		camz += 0.01f;
	}

	if (key == 'a')
	{
		camx -= 0.01;
	}

	if (key == 'd')
	{
		camx += 0.01;
	}

	if (key == 'w')
	{
		camy -= 0.01;
	}

	if (key == 's')
	{
		camy += 0.01;
	}

	glutPostRedisplay();
}

void idle()
{

}

//---------------------------------------------------------------------
//
// main
//
int main(int argc, char** argv)
{
	//Initializing to draw
	glutInit(&argc, argv);
	glutInitDisplayMode(GLUT_RGBA);
	glutInitWindowSize(1024, 1024);
	glutCreateWindow("Hello World");

	glewInit();	

	init();
	glutDisplayFunc(drawScene);
	glutIdleFunc(idle);
	
	glutKeyboardFunc(keyboard);

	glutMainLoop();

}
