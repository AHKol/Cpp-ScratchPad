
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

enum VAO_IDs { Triangles, NumVAOs };
enum Buffer_IDs { ArrayBuffer, NumBuffers };
enum Attrib_IDs { vPosition = 0 };

GLuint VAOs[NumVAOs];
GLuint Buffers[NumBuffers];
GLuint location;

const GLuint NumVertices = 14;

float rotate_value = 0;

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

	//Defining (in RAM NOT VRAM) the coordinates and colors of the vertices.
	//The coordinates are specified on X-Y plane
	//In the following section, we will transfer these data from RAM into VRAM

	// Coordinates of vertices (Square)
	GLfloat vertices[NumVertices + 360][2] = {
		//Sun
		{ 0, -1.00 },
		{ -0.95, -0.31 },
		{ -0.59, 0.81 },
		{ 0.59, 0.81 },
		{ 0.95, -0.31 },

		//Earth
		{ -0.3, -0.3 }, 
		{ 0.3, -0.3 },
		{ 0.3, 0.3 },
		{ -0.3, 0.3 },
		//Jupiter
		{ -0.0, -1.0 },
		{ -0.87, 0.5 },
		{ 0.87, 0.5 },
		//Moon
		{0.0, 0.0}
	};
	for (int i = 0; i < 360; i++) {
		vertices[NumVertices + i][0] = cos(i);
		vertices[NumVertices + i][0] = sin(i);
	}

	// Colors for vertices in {R, G, B} mode
	GLfloat colorData[NumVertices][3] = {
		{ 1,0,0 }, //Red
		{ 0,1,0 }, //Green
		{ 0,0,1 }, //Blue
		{ 1,1,1 },  //White
		{ 1,0,0 }, //Red
		{ 0,1,0 }, //Green
		{ 0,0,1 }, //Blue
		{ 1,1,1 },  //White
		{ 1,0,0 }, //Red
		{ 0,1,0 }, //Green
		{ 0,0,1 }, //Blue
		{ 1,1,1 },  //White
		{ 1,1,1 },  //White
		{ 1,0,0 } //Red
	};

	//Allocating buffers in VRAM and pushing vertex data (position and color) into the data.
	//In the following section, we will transfer the vertex data from RAM into VRAM using OpenGL library
	//Do NOT go inside the details. I will teach them in details next week.
	glGenBuffers(2, Buffers);
	glBindBuffer(GL_ARRAY_BUFFER, Buffers[0]);
	glBufferData(GL_ARRAY_BUFFER, sizeof(vertices),	vertices, GL_STATIC_DRAW);
	glBindAttribLocation(program, 0, "vPosition");
	glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 0, BUFFER_OFFSET(0));
	glEnableVertexAttribArray(0);

	glBindBuffer(GL_ARRAY_BUFFER, Buffers[1]);
	glBufferData(GL_ARRAY_BUFFER, sizeof(colorData), colorData, GL_STATIC_DRAW);
	glBindAttribLocation(program, 1, "vertexColor");
	glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 0, BUFFER_OFFSET(0));
	glEnableVertexAttribArray(1);
	location = glGetUniformLocation(program, "model_matrix");
	
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

	//Position Sun
	glm::mat4 model_view = glm::scale(glm::mat4(1.0), glm::vec3(0.1, 0.1, 0.1));
	//model_view = glm::rotate(model_view, rotate_value, glm::vec3(0.0f, 0.0f, 1.0f));
	
	//Draw Sun
	glUniformMatrix4fv(location, 1, GL_FALSE, &model_view[0][0]);
	glDrawArrays(GL_POLYGON, 0, 5);

	//Position Earth
	model_view = glm::scale(glm::mat4(1.0), glm::vec3(0.1, 0.1, 1));
	model_view = glm::rotate(model_view, rotate_value / 2, glm::vec3(0.0f, 0.0f, 1.0f));
	model_view = glm::translate(model_view, glm::vec3(5.0f, 0.0f, 0.0f));
	model_view = glm::rotate(model_view, 2 * rotate_value, glm::vec3(0.0f, 0.0f, 1.0f));

	//Draw Earth
	glUniformMatrix4fv(location, 1, GL_FALSE, &model_view[0][0]);
	glDrawArrays(GL_POLYGON, 5, 4);


	//Position Earth Moon
	model_view = glm::rotate(model_view, rotate_value * -5 , glm::vec3(0.0f, 0.0f, 1.0f));
	model_view = glm::translate(model_view, glm::vec3(1.0f, 0.0f, 0.0f));
	model_view = glm::rotate(model_view, rotate_value * -5, glm::vec3(0.0f, 0.0f, 1.0f));
	model_view = glm::scale(model_view, glm::vec3(0.5, 0.5, 1));

	//Draw Earth Moon
	glUniformMatrix4fv(location, 1, GL_FALSE, &model_view[0][0]);
	glDrawArrays(GL_POLYGON, 5, 4);

	/*
	//Position Jupiter
	model_view = glm::scale(glm::mat4(1.0), glm::vec3(0.05, 0.05, 0.05));
	model_view = glm::rotate(model_view, rotate_value / 5, glm::vec3(0.0f, 0.0f, 1.0f));
	model_view = glm::translate(model_view, glm::vec3(15.0f, 0.0f, 0.0f));
	model_view = glm::rotate(model_view, 2 * rotate_value, glm::vec3(0.0f, 0.0f, 1.0f));

	//Draw Jupiter
	glUniformMatrix4fv(location, 1, GL_FALSE, &model_view[0][0]);
	glDrawArrays(GL_POLYGON, 9, 3);
	*/

	/*
	//Position Moon 1
	model_view = glm::scale(glm::mat4(1.0), glm::vec3(0.05, 0.05, 0.05));
	model_view = glm::rotate(model_view, rotate_value / 5, glm::vec3(0.0f, 0.0f, 1.0f));
	model_view = glm::translate(model_view, glm::vec3(15.0f, 0.0f, 0.0f));
	model_view = glm::rotate(model_view, 2 * rotate_value, glm::vec3(0.0f, 0.0f, 1.0f));
	*/


	//GL_TRIANGLEFAN

	
	//Flush the image onto the window (screen)
	glFlush();

}

//This function is called "idle" function. This function is invoked by OpenGL ever frame (60 frame per second).
//This function is used for animation, since you can animate your scene for every frame.
//You will need to register the name of your idle function to OpenGL (It is "runEveryFrame" in here).
//The registration happens in the main() function using glutIdleFunc(runEveryFrame) function.
void runEveryFrame()
{
	//Increasing our rotation angle
	rotate_value += 0.001;

	//Distance between shapes
	//jump_value == 0.5 ? jump_value = -0.5 : jump_value = 0.5;

	//glutpostRedisplay() finction calls the display function in order to re-draw the scene
	//In here, glutPostRedisplay function is used inside the idle function to re-draw the scene for every frame.
	//So, in here, we increase the rotation value and then redraw the scene with the new value for the rotation.
	glutPostRedisplay();

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
	glutInitWindowSize(512, 512);
	glutCreateWindow("Hello World");
	glewInit();	

	//init function is defined above
	init();

	//Registering the display function
	//Read the comments for the drawScene function above
	glutDisplayFunc(drawScene);

	//Registering the idle function
	//Read the comments for the runEveryFrame function above
	glutIdleFunc(runEveryFrame);

	//glutMainLoop enters the event processing loop
	glutMainLoop();
	
	

}
