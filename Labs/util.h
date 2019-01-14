#pragma once
#include <iostream>
#include <fstream>
#include <string>
#include <vector>

namespace oop345 {
   void trimSpaces(std::string& inputString);
   void csvRead(std::string& filename, char delimiter, std::vector<std::vector<std::string>>& csvData);
   void csvPrint(std::vector<std::vector<std::string>>& csvData);
}