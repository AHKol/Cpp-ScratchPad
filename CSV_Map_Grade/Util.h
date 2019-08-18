#pragma once
#include <iostream>
#include <fstream>
#include <string>
#include <vector>
using namespace std;
void trimSpaces(std::string& inputString)
{
	while (!inputString.empty() && inputString[0] == ' ')
		inputString.erase(0, 1);

	while (!inputString.empty() && inputString[inputString.size() - 1] == ' ')
		inputString.erase(inputString.size() - 1, 1);
}
void csvRead(const std::string& filename, char delimiter, std::vector<std::vector<std::string>>& csvData)
{
	std::fstream fin(filename, std::ios::in);
	if (fin.is_open()) {
		std::string line;
		std::vector<std::string> fields;
		while (getline(fin, line)) {
			size_t index = 0;
			std::string field;
			while (index < line.size() + 1) {
				if (line[index] == delimiter || index == line.size()) { //read until delim or endloop
					trimSpaces(field);
					if (!field.empty() && field[0] != delimiter) {  //clean and add if not empty
						fields.push_back(move(field));
					}
				}
				else {
					field += line[index];
				}
				index++;
			}
			if (!fields.empty())
				csvData.push_back(move(fields));
		}
		fin.close();
	}
	else {
		throw std::string("Cannot open file '") + filename + "'";
	}
}
void csvPrint(std::vector<std::vector<std::string>>& csvData)
{
	std::cout << "Range-based for:\n";
	for (auto line : csvData) {
		for (auto field : line) {
			std::cout << "(" << field << ")";
		}
		std::cout << "\n";
	}
	std::cout << "\n";
}

//Workin progress

//csvmap class
//input vector vector as constructor
//string argument used for formating, use fprint, %d%c%s%f
//2d dynamic array, [0] being size of row?

//Should there be an empty array of every type of map?

class MapCSV {
	vector<void*> dataAll;

	bool columnToData(char type, vector<vector<string>> data, int iColumn) {
		vector<double>* columnD = new vector<double>;
		vector<double>* columnF = new vector<double>;
		vector<char>* columnC = new vector<char>;
		vector<string>* columnS = new vector<string>;
		switch (type)
		{
		case 'd':
			for (auto& row : data) {
				try {
					columnD->push_back(stod(row[iColumn]));
				}
				catch (...) {
					delete columnD;
					return false;
				}
			}
			dataAll.push_back(columnD);
			break;
		case 'f':
			for (auto& row : data) {
				try {
					columnF->push_back(stod(row[iColumn]));
				}
				catch (...) {
					delete columnF;
					return false;
				}
			}
			dataAll.push_back(columnF);
			break;
		case 'c':
			for (auto& row : data) {
				//test if string can convert to char
				if (row.size() != 1) {
					delete columnC;
					return false;
				}
				columnC->push_back(row[iColumn][0]);
			}
			dataAll.push_back(columnC);
			break;
		case 's':
			for (auto& row : data) {
				columnS->push_back(row[iColumn]);
			}
			dataAll.push_back(columnS);
			break;
		default:
			return false;
			break;
		}
		return true;
	}
public:
	MapCSV() {

	}
	MapCSV(vector<vector<string>> &args, string format) {
		for (size_t column = 0; column < args.size(); column++) {
			columnToData(format[column], args, column);
		}
	}
	~MapCSV() {

	}
	vector<void> getRow(int rowID) {

	}
	void* getElement(int rowID, int column) {
		int* i = new int(ONE);
		return i;
	}
	void doThing() {
		int foo = (int)getElement(1, 1);
	}
};