#pragma once
#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include "i.h"

namespace oop345 {

   class Order
   {
   private:
      bool validName(std::string& s);
      bool validSlot(std::string& s);
      std::string customer_name, product_name;
      std::vector<std::string> items;
   public:
      Order() {}
      Order(std::vector<std::string>& line);
      void print();
      void graph(std::fstream& gv);
      int itemSize() { return items.size(); }
      std::string itemName(int i) { return items[i]; }
      std::string custName() { return customer_name; }

   };
   class OrderManager
   {
   private:
      std::vector <Order> orderList;
   public:
      OrderManager() {}
      OrderManager(std::vector<std::vector<std::string>>& csvDataTask);
      int size() { return orderList.size(); }
      bool validate(ItemManager& items);
      void print();
      void graph(const std::string& file);
   };
}