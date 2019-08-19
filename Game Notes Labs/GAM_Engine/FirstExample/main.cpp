//functions to standardize reading file
//consider classes to standardize objects
	//classes that have methods to facilitate tranformation
//put in seperate files	

//Every object will require 3 buffers.
//When app is loading, send vectors to buffer, object V object F object T

//TODO, vertex 1d array

#pragma once

#include "vgl.h"
#include "LoadShaders.h"
#include "glm\glm.hpp"
#include "glm\gtc\matrix_transform.hpp"
#include "SOIL.h"
#include <vector>
#include <fstream>
#include <iostream>
#include <string>


enum VAO_IDs { Triangles, NumVAOs };
enum Buffer_IDs { ArrayBuffer, NumBuffers };
enum Attrib_IDs { vPosition = 0 };

GLuint VAOs[NumVAOs];


GLuint Buffers[3];	//Buffer pointers used to push data into our shaders: vertexPositions, vertexColors and vertex Coordinates. Look at triangles.vert shader
GLuint location;	//Will be used to stire the location of our model_view matrix in VRAM
GLuint location2;	//Will be used to stire the location of our camera matrix in VRAM
GLuint location3;	//Will be used to stire the location of our projection matrix in VRAM

GLuint texture[2];	//Array of pointers to textrure data in VRAM. We will use two different textures in this example

const GLuint NumVertices = 2446;
float translate_value = 0;
float rotate_value = 0;
float camera_distance = 0.1;

GLuint program;

class Model {
private:
	std::vector <glm::vec3 > vArr;
	std::vector <glm::vec3 > uvArr;
	std::vector <GLuint> viArr;
	std::vector <GLuint> uviArr;
public:
	Model() {

	}
	std::vector <glm::vec3 > getvArr() {
		return vArr;
	};
	std::vector <glm::vec3> getuvArr() {
		return uvArr;
	};
	std::vector <GLuint> getviArr() {
		return viArr;
	};
	std::vector <GLuint> getuviArr() {
		return uviArr;
	};

	void loadFile(const char * filename) {
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

			//Set Vectors to proper size
			vArr.resize(vArrSize);
			uvArr.resize(uvArrSize);
			viArr.resize(viArrSize * 3);
			uviArr.resize(uviArrSize * 3);

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

					//parse row for the 3 elements
					/*
					std::vector <GLfloat> glRow;
					glRow.push_back(stof(row[0]));
					glRow.push_back(stof(row[1]));
					glRow.push_back(stof(row[2]));

					vArr[vArri] = move(glRow);
					*/
					vArr[vArri].x = stof(row[0]);
					vArr[vArri].y = stof(row[1]);
					vArr[vArri].z = stof(row[2]);
					vArri++;
				}
				//UVs
				if (line[0] == 'v' && line[1] == 't') {
					size_t index = 2; //start after identifier
					std::string field;
					while (index < line.size() + 1) {
						if (line[index] == ' ' || index == line.size()) { //read until end of field or line
							if (!field.empty() && field[0] != ' ') { //if field has data
								fRow.push_back(move(field));
							}
						}
						else {
							field += line[index]; //add character to field
						}
						index++;
					}
					/*
					//parse row for the 3 elements
					std::vector <GLfloat> glRow;
					glRow.push_back(stof(fRow[0]));
					glRow.push_back(stof(fRow[1]));
					glRow.push_back(stof(fRow[2]));

					tArr[tArri] = move(glRow);

					tArri++;
					*/
					uvArr[uvArri].x = stof(fRow[0]);
					uvArr[uvArri].y = stof(fRow[1]);
					uvArr[uvArri].z = stof(fRow[2]);
					uvArri++;
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
					/*
					//parse row for the 3 elements
					std::vector <GLuint> fglRow;
					fglRow.push_back(stod(fRow[0]));
					fglRow.push_back(stod(fRow[1]));
					fglRow.push_back(stod(fRow[2]));

					fArr[fArri] = move(fglRow);

					std::vector <GLuint> ufglRow;
					ufglRow.push_back(stod(ufRow[0]));
					ufglRow.push_back(stod(ufRow[1]));
					ufglRow.push_back(stod(ufRow[2]));

					fvArr[fvArri] = move(ufglRow);

					fArri++;
					fvArri++;
					*/
					viArr[viArri * 3] = stod(fRow[0]);
					viArr[viArri * 3 + 1] = stod(fRow[1]);
					viArr[viArri * 3 + 2] = stod(fRow[2]);
					viArri++;

					uviArr[uviArri * 3] = stod(ufRow[0]);
					uviArr[uviArri * 3 + 1] = stod(ufRow[1]);
					uviArr[uviArri * 3 + 2] = stod(ufRow[2]);
					uviArri++;
				}

			}
			fin.close();
		}
		else {
			//TODO: return error
		}
	}
	void loadModel() {
		//do during display()

		//define buffer
		glBindBuffer(GL_ARRAY_BUFFER, Buffers[2]);

		//Pushing the texture coordinates into the buffer
		glBufferData(GL_ARRAY_BUFFER, uvArr.size() * sizeof(glm::vec3) , &uvArr, GL_STATIC_DRAW);
		glBindAttribLocation(program, 2, "vTexCoord");
		glVertexAttribPointer(2, 2, GL_FLOAT, GL_FALSE, 0, BUFFER_OFFSET(0));
		glEnableVertexAttribArray(2);

		//pick the texture relevent to this model
		glBindTexture(GL_TEXTURE_2D, texture[0]);
		//Rendering the faces
		for (int i = 0; i < uviArr.size(); i++) {
			glDrawElements(GL_TRIANGLES, 3, GL_UNSIGNED_INT, &uviArr);
		}

	}
	void loadTexture() {
		//do during init()

		GLint width, height;
		unsigned char* modelTexture = SOIL_load_image("Earth.jpg", &width, &height, 0, SOIL_LOAD_RGB);

		//get texture coordinates

		//Set the type of the free buffer as "TEXTURE_2D"
		glBindTexture(GL_TEXTURE_2D, texture[0]);

		//Loading the second texture into the second allocated buffer:
		glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, width, height, 0, GL_RGB, GL_UNSIGNED_BYTE, modelTexture);

		//Setting up parameters for the texture that recently pushed into VRAM
		glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT);
		glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT);
		glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
		glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
		
	}
};

Model modelOne;

//---------------------------------------------------------------------
//
// init
//


void init(void)
{
	//get model
	modelOne.loadFile("Earth.Obj");

	



	//Creating our second texture:
	//This texture is loaded from file. To do this, I will use the SOIL (Simple OpenGL Imaging Library) library to import the texture.
	//When using the SOIL_load_image() function, make sure the you are using correct patrameters, or else, your image will NOT be loaded properly, or will not be loaded at all.
	GLint width, height;
	unsigned char* textureData2 = SOIL_load_image("apple.png", &width, &height, 0, SOIL_LOAD_RGB);

	//These are the texture coordinates for the second texture
	GLfloat textureCoordinates2[4][2] = {
		0.0f, 10.0f,
		10.0f, 10.0f,
		10.0f, 0.0f,
		0.0f, 0.0f
	};



	//Once we set up our textures, we need to push the texture data for my two textures into VRAM
	//Note: The texture coordinates will be sent to shaders as variable vTexCoord. Take a look at the vertex shader for more details.

	//Allocating two buffers in VRAM
	glGenTextures(2, texture);
	modelOne.loadTexture();

	///////////////////////SECOND TEXTURE////////////////////////

	//Set the type of the second buffer as "TEXTURE_2D"
	glBindTexture(GL_TEXTURE_2D, texture[1]);

	//Loading the second texture into the second allocated buffer:
	glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, width, height, 0, GL_RGB, GL_UNSIGNED_BYTE, textureData2);

	//Setting up parameters for the texture that recently pushed into VRAM
	glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT);
	glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT);
	glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
	glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
	//////////////////////////////////////////////////////////////


	//Of courese, following the same procedure, you mah load as many textures as your Graphic Card allows you to.

	////////////////////////////////////DONE WITH SETTING-UP THE TEXTURES///////////////////////////

	ShaderInfo shaders[] = {
		{ GL_VERTEX_SHADER, "triangles.vert" },
	{ GL_FRAGMENT_SHADER, "triangles.frag" },
	{ GL_NONE, NULL }
	};

	program = LoadShaders(shaders);
	glUseProgram(program);	//My Pipeline is set up

	GLfloat vertices[NumVertices][2] = {
		{ -0.90, -0.90 }, // Square
		{ 0.9, -0.9 },
		{ 0.90, 0.9 },
		{ -0.9, 0.9 }
	};

	GLfloat colorData[NumVertices][3] = {
		{ 1,0,0 }, // color for vertices
		{ 0,1,0 },
		{ 0,0,1 },
		{ 1,1,1 }
	};

	glGenBuffers(3, Buffers);

	glBindBuffer(GL_ARRAY_BUFFER, Buffers[0]);
	//glBufferData(GL_ARRAY_BUFFER, sizeof(vertices), vertices, GL_STATIC_DRAW);
	glBufferData(GL_ARRAY_BUFFER, modelOne.getuviArr().size() * sizeof(glm::vec3), &modelOne.getuviArr(), GL_STATIC_DRAW);
	glBindAttribLocation(program, 0, "vPosition");
	glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 0, BUFFER_OFFSET(0));
	glEnableVertexAttribArray(0);


	glBindBuffer(GL_ARRAY_BUFFER, Buffers[1]);
	glBufferData(GL_ARRAY_BUFFER, sizeof(colorData), colorData, GL_STATIC_DRAW);
	glBindAttribLocation(program, 1, "vertexColor");
	glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 0, BUFFER_OFFSET(0));
	glEnableVertexAttribArray(1);

	location = glGetUniformLocation(program, "model_matrix");
	location2 = glGetUniformLocation(program, "camera_matrix");
	location3 = glGetUniformLocation(program, "projection_matrix");

}


//---------------------------------------------------------------------
//
// display
//

void
display(void)
{
	glClear(GL_COLOR_BUFFER_BIT);

	modelOne.loadModel();

	glm::mat4 model_view = glm::translate(glm::mat4(1.0), glm::vec3(5, 0, 0));
	model_view = glm::scale(model_view, glm::vec3(2.0, 2.0, 1.0));
	glUniformMatrix4fv(location, 1, GL_FALSE, &model_view[0][0]);

	glm::mat4 camera_matrix = glm::lookAt(glm::vec3(0.0, 0.0, camera_distance), glm::vec3(0.0, 0.0, -1.0), glm::vec3(0.0, 1.0, 0.0));
	glUniformMatrix4fv(location2, 1, GL_FALSE, &camera_matrix[0][0]);

	glm::mat4 projection_matrix = glm::frustum(-1.0, +1.0, -1.0, +1.0, 0.01, 100.0);
	glUniformMatrix4fv(location3, 1, GL_FALSE, &projection_matrix[0][0]);


	//The texture coordinates for the first square:
	GLfloat textureCoordinates[4][2] = {
		0.0f, 10.0f,
		10.0f, 10.0f,
		10.0f, 0.0f,
		0.0f, 0.0f
	};

	//Setting up the buffer in VRAM to pass the texture coordinates

	//Remember that we defined three buffers: position, color and texture coordinates
	//Selecting the buffer that will contain texture coordinates (Buffers[2])
	glBindBuffer(GL_ARRAY_BUFFER, Buffers[2]);

	//Pushing the texture coordinates into the buffer
	glBufferData(GL_ARRAY_BUFFER, sizeof(textureCoordinates), textureCoordinates, GL_STATIC_DRAW);
	glBindAttribLocation(program, 2, "vTexCoord");
	glVertexAttribPointer(2, 2, GL_FLOAT, GL_FALSE, 0, BUFFER_OFFSET(0));
	glEnableVertexAttribArray(2);

	//We have two textures defined in VRAM, which one would you like to be applied?
	//Lets say the first one.
	glBindTexture(GL_TEXTURE_2D, texture[0]);
	GLuint tmp[4] = {0, 1, 2, 3};
	//Rendering the first square
	glDrawElements(GL_QUADS, 4, GL_UNSIGNED_INT, tmp);
	//Done !!!

	//Now, rendering the second square with the second texture applied. (The texture that was loaded from file)

	//Definig the texture coordinates that will be pushed into the buffer and from there to shaders
	GLfloat textureCoordinates2[4][2] = {
		0.0f, 1.0f,
		1.0f, 1.0f,
		1.0f, 0.0f,
		0.0f, 0.0f
	};
	
	//Pushing the texture coordinated for the second geometry into the buffer. Note: these data will be pushed into the last buffer that is bound. A few lines above, we set: glBindBuffer(GL_ARRAY_BUFFER, Buffers[2]);
	//So, these data will be pushed into the Buffers[2] which contains the texture coordinates.
	//In other words, we are re-initializing texture coordinnate values inside the buffer.(Previously, the texture copordinates for our first geometry was here).
	glBufferData(GL_ARRAY_BUFFER, sizeof(textureCoordinates2), textureCoordinates2, GL_STATIC_DRAW);

	model_view = glm::translate(glm::mat4(1.0), glm::vec3(-5, 0, 0));
	model_view = glm::scale(model_view, glm::vec3(2.0, 2.0, 1.0));
	glUniformMatrix4fv(location, 1, GL_FALSE, &model_view[0][0]);

	//We have two textures defined in VRAM, which one would you like to be applied?
	//Now, we use the second one.
	glBindTexture(GL_TEXTURE_2D, texture[1]);
	//Rendering the first square
	glDrawElements(GL_QUADS, 4, GL_UNSIGNED_INT, tmp);
	//Done !!!

	glFlush();
}

void idle()
{
	
}

//---------------------------------------------------------------------
//
// main
//

void keyboardHandler(unsigned char key, int x, int y)
{
	if (key == 'a')
	{
		translate_value -= 0.1;
	}
	else if (key == 'd')
	{
		translate_value += 0.1;
	}
	else if (key == 'l')
	{
		rotate_value += 0.1;
	}
	else if (key == 'k')
	{
		rotate_value -= 0.1;
	}
	else if (key == 'f')
	{
		camera_distance += 0.1;
	}
	else if (key == 'j')
	{
		camera_distance -= 0.1;
	}

	glutPostRedisplay();
}

int
main(int argc, char** argv)
{
	glutInit(&argc, argv);
	glutInitDisplayMode(GLUT_RGBA);
	glutInitWindowSize(512, 512);
	glutCreateWindow("Hello World");

	glewInit();	//Initializes the glew and prepares the drawing pipeline.

	init();

	glutDisplayFunc(display);

	glutKeyboardFunc(keyboardHandler);

	glutIdleFunc(idle);

	glutMainLoop();
	
	

}
