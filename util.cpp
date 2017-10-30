#pragma once
#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include "util.h"
namespace oop345 {
   void trimSpaces(std::string& inputString)
   {
      while (!inputString.empty() && inputString[0] == ' ')
         inputString.erase(0, 1);

      while (!inputString.empty() && inputString[inputString.size() - 1] == ' ')
         inputString.erase(inputString.size() - 1, 1);
   }
   void csvRead(std::string& filename, char delimiter, std::vector<std::vector<std::string>>& csvData)
   {
      std::fstream fin(filename, std::ios::in);
      if (fin.is_open()) {
         std::string line;
         std::vector<std::string> fields;
         while (getline(fin, line)) {
            auto cr = line.find('\r');
            if (cr != std::string::npos) {
               line.erase(cr);
            }
            size_t index = 0;
            std::string field;
            while (index < line.size() + 1) {
               if (line[index] == delimiter || index == line.size()) { //read until delim or endloop
                  oop345::trimSpaces(field);
                  if (!field.empty() && field[0] != delimiter) {  //clean and add if not empty
                     fields.push_back(move(field));
                  }
               } else {
                  field += line[index];
               }
               index++;
            }
            if (!fields.empty())
               csvData.push_back(move(fields));
         }
         fin.close();
      } else {
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

      std::cout << "iterator for:\n";
      for (auto line = csvData.begin(); line != csvData.end(); line++) {
         for (auto field = line->begin(); field != line->end(); field++) {
            std::cout << "[" << *field << "]";
         }
         std::cout << "\n";
      }
      std::cout << "\n";

      std::cout << "indices for:\n";
      for (size_t line = 0; line < csvData.size(); line++) {
         for (size_t field = 0; field < csvData[line].size(); field++) {
            std::cout << "{" << csvData[line][field] << "}";
         }
         std::cout << "\n";
      }
      std::cout << "\n";
   }
}