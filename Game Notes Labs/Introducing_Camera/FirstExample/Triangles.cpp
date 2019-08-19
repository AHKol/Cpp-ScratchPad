
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

//Variables to store the location of matrices inside VRAM
GLuint cam_matrix_location_in_vram;
GLuint projection_matrix_location_in_vram;

const GLuint NumVertices = 8;

float translate_value = 0;
float rotate_value = 0;
float alpha = 0;

float orbit = 0;

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
//
// init
//


void
init(void)
{

	ShaderInfo shaders[] = {
		{ GL_VERTEX_SHADER, "triangles.vert" },
		{ GL_FRAGMENT_SHADER, "triangles.frag" },
		{ GL_NONE, NULL }
	};

	GLuint program = LoadShaders(shaders);
	glUseProgram(program);	//My Pipeline is set up

	GLfloat vertices[NumVertices][3] = {
		{ -0.45, 0.45, -0.45 },
		{ 0.45, 0.45, -0.45 },
		{ 0.45, -0.45, -0.45 },
		{ -0.45, -0.45, -0.45 },

		{ -0.45, 0.45, 0.45 },
		{ 0.45 , 0.45 , 0.45 },
		{ 0.45, -0.45, 0.45 },
		{ -0.45, -0.45, 0.45 },
	};

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
	glBufferData(GL_ARRAY_BUFFER, sizeof(vertices),	vertices, GL_STATIC_DRAW);
	glBindAttribLocation(program, 0, "vPosition");
	glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 0, BUFFER_OFFSET(0));
	glEnableVertexAttribArray(0);

	glBindBuffer(GL_ARRAY_BUFFER, Buffers[1]);
	glBufferData(GL_ARRAY_BUFFER, sizeof(colorData), colorData, GL_STATIC_DRAW);
	glBindAttribLocation(program, 0, "vPosition");
	glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 0, BUFFER_OFFSET(0));
	glEnableVertexAttribArray(1);


	//Retrieving the location of the matrices from VRAM and storing them inside these variables
	location = glGetUniformLocation(program, "model_matrix");
	cam_matrix_location_in_vram = glGetUniformLocation(program, "camera_matrix");
	projection_matrix_location_in_vram = glGetUniformLocation(program, "projection_matrix");
}


//---------------------------------------------------------------------
//
// display
//

void
display(void)
{
	glClear(GL_COLOR_BUFFER_BIT);

	//DRAW CAMERA

	//Setting up camera matrix and initialize the camera_matrix in VRAM	
	glm::mat4 camera_matrix = glm::lookAt(glm::vec3(camx, camy, camz), glm::vec3(0, 0, 0), glm::vec3(0, 1, 0));
	glUniformMatrix4fv(cam_matrix_location_in_vram, 1, GL_FALSE, &camera_matrix[0][0]);

	//Setting up projection matrix and initialize the projection_matrix in VRAM
	glm::mat4 projection_matrix = glm::frustum(-0.01f, +0.01f, -0.01f, +0.01f, 0.01f, 10.0f);
	glUniformMatrix4fv(projection_matrix_location_in_vram, 1, GL_FALSE, &projection_matrix[0][0]);


	//DRAW SCENE

	glm::mat4 model_view;

	//sun
	model_view = glm::scale(glm::mat4(1.0), glm::vec3(0.1, 0.1, 0.1));
	glUniformMatrix4fv(location, 1, GL_FALSE, &model_view[0][0]);
	drawCube();

	//Position Earth
	model_view = glm::scale(glm::mat4(1.0), glm::vec3(0.1, 0.1, 0.1));
	model_view = glm::rotate(model_view, orbit / 2, glm::vec3(0.0f, 0.0f, 1.0f));
	model_view = glm::translate(model_view, glm::vec3(5.0f, 0.0f, 0.0f));
	model_view = glm::rotate(model_view, 2 * orbit, glm::vec3(0.0f, 0.0f, 1.0f));

	//Draw Earth
	glUniformMatrix4fv(location, 1, GL_FALSE, &model_view[0][0]);
	drawCube();

	//Position Earth Moon
	model_view = glm::rotate(model_view, orbit * -5, glm::vec3(0.0f, 0.0f, 1.0f));
	model_view = glm::translate(model_view, glm::vec3(1.0f, 0.0f, 0.0f));
	model_view = glm::rotate(model_view, orbit * -5, glm::vec3(0.0f, 0.0f, 1.0f));
	model_view = glm::scale(model_view, glm::vec3(0.5, 0.5, 1));

	//Draw Earth Moon
	glUniformMatrix4fv(location, 1, GL_FALSE, &model_view[0][0]);
	drawCube();

	glFlush();
}

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
}

void mouse(int state, int button, int x, int y)
{

}


void idle()
{
	orbit += 0.001;
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
	glutCreateWindow("Introducing Camera");

	glewInit();	//Initializes the glew and prepares the drawing pipeline.

	init();

	glutDisplayFunc(display);

	glutIdleFunc(idle);

	glutKeyboardFunc(keyboard);

	glutMouseFunc(mouse);

	glutMainLoop();
	
	

}
