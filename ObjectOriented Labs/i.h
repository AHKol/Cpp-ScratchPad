#pragma once
#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include "util.h"
namespace oop345 {
   class Item
   {
   private:
      bool validName(std::string& s);
      bool validSlot(std::string& s);
      std::string item_name, installer_task, remover_task, sequential_code, description;
   public:
      Item() {}
      Item(std::vector<std::string>& line);
      void print();
      void graph(std::fstream& gv);
      std::string name() { return item_name; }
      std::string installer() { return installer_task; }
      std::string remover() { return remover_task; }

   };
   class ItemManager
   {
   private:
      std::vector <Item> itemList;
   public:
      ItemManager() {}
      ItemManager(std::vector<std::vector<std::string>>& csvData);
      int size() { return itemList.size(); }
      Item* find(std::string& i);
      Item* findRemover(std::string& i);
      Item* findInstaller(std::string& i);
      bool validate();
      void print();
      void graph(const std::string& file);
   };
}