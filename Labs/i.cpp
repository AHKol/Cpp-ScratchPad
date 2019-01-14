#pragma once
#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include "util.h"
#include "i.h"
namespace oop345 {
      bool Item::validName(std::string& s)
      {
         if (s.empty()) return false;
         for (auto& c : s) //check each character 'c' in string 'c'
            if (!(isalnum(c) || c == ' '))
               return false;
         return true;
      }
      bool Item::validSlot(std::string& s)
      {
         if (s.empty()) return false;
         for (auto& c : s) //check each character 'c' in string 'c'
            if (!isdigit(c))
               return false;
         return true;
      }
      Item::Item(std::vector<std::string>& line)
      {
         switch (line.size()) {
         case 5:
            description = line[4];
         case 4:
            if (validSlot(line[3]))
               sequential_code = line[3];
            else
               throw std::string("Expected a sequential_code, found ") + line[3];
         case 3:
            if (validName(line[2]))
               remover_task = line[2];
            else
               throw std::string("Expected a remover task name, found ") + line[2];
         case 2:
            if (validName(line[1]))
               installer_task = line[1];
            else
               throw std::string("Expected a installer task, found ") + line[1];
         case 1:
            if (validName(line[0]))
               item_name = line[0];
            else
               throw std::string("Expected a fail task name, found ") + line[0];
            break;
         default:
            throw std::string("Expected 1, 2, 3, 4, 5 fields, not ") + std::to_string(line.size());
            break;
         }
      }
      void Item::print()
      {
         std::cout << "name=" << item_name << '\n';
         std::cout << "installer task=" << installer_task << '\n';
         std::cout << "remover task=" << remover_task << '\n';
         std::cout << "sequential_code=" << sequential_code << '\n';
         std::cout << "description=" << description << '\n';
         std::cout << "\n";
      }
      void Item::graph(std::fstream& gv)
      {
         if (!installer_task.empty()) {
            gv << '"' << item_name << '"';
            gv << "->";
            gv << '"' << installer_task << '"';
            gv << "[color=green];\n";
         }
         if (!remover_task.empty()) {
            gv << '"' << item_name << '"';
            gv << "->";
            gv << '"' << remover_task << '"';
            gv << "[color=red];\n";
         }
         if (installer_task.empty() && remover_task.empty()) {
            gv << '"' << item_name << '"';
            gv << ";\n";
         }
      }
      ItemManager::ItemManager(std::vector<std::vector<std::string>>& csvData)
      {
         for (auto& line : csvData) {
            itemList.push_back(Item(line));
         }
      }
      Item * ItemManager::find(std::string & i)
      {
         for (int item = 0; item < itemList.size(); item++) {
            if (itemList[item].name() == i)
               return&itemList[item];
         }
         return nullptr;
      }

      Item * ItemManager::findRemover(std::string & i)
      {
         for (int item = 0; item < itemList.size(); item++) {
            if (itemList[item].remover() == i)
               return&itemList[item];
         }
         return nullptr;
      }

      Item * ItemManager::findInstaller(std::string & i)
      {
         for (int item = 0; item < itemList.size(); item++) {
            if (itemList[item].installer() == i)
               return&itemList[item];
         }
         return nullptr;
      }

      bool ItemManager::validate()
      {
         int errors = 0;
         for (auto& item : itemList) {
            std::string installer = item.installer();
            if (findInstaller(installer) == nullptr) {
               errors++;
               std::cerr << "Cannot find installer " << installer << "\n";
            }
            std::string remover = item.remover();
            if (findRemover(remover) == nullptr) {
               errors++;
               std::cerr << "Cannot find remover " << remover << "\n";
            }
         }
         if (errors) std::cerr << "Error count: " << errors << "\n";
         return !errors;
      }

      void ItemManager::print()
      {
         for (auto& t : itemList) {
            t.print();
         }
      }
      void ItemManager::graph(const std::string& file)
      {
         std::fstream fout(file + ".gv", std::ios::out | std::ios::trunc);
         if (fout.is_open()) {
            fout << "digraph itemgraph { \n";
            for (auto& t : itemList) {
               t.graph(fout);
            }
            fout << "}";
            fout.close();
         }
      }
}