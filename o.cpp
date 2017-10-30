#pragma once
#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include "util.h"
#include "i.h"
#include "o.h"
namespace oop345 {

   bool Order::validName(std::string& s)
   {
      if (s.empty()) return false;
      for (auto& c : s) //check each character 'c' in string 'c'
         if (!(isalnum(c) || c == ' '))
            return false;
      return true;
   }
   bool Order::validSlot(std::string& s)
   {
      if (s.empty()) return false;
      for (auto& c : s) //check each character 'c' in string 'c'
         if (!isdigit(c))
            return false;
      return true;
   }
   Order::Order(std::vector<std::string>& line)
   {
      if (line.size() >= 3) {
         if (validName(line[1])) {
            product_name = line[1];
         } else {
            throw std::string("Expected product name, not " + line[1]);
         }
         if (validName(line[0])) {
            customer_name = line[0];
         } else {
            throw std::string("Expected customer name, not " + line[0]);
         }
         for (int i = 2; i < line.size(); i++) {
            items.push_back(move(line[i]));
         }
      } else {
         throw std::string("Expected 3 or more fields, not ") + std::to_string(line.size());
      }
   }
   void Order::print()
   {
      std::cout << "customer=" << customer_name << '\n';
      std::cout << "product=" << product_name << '\n';
      for (int i = 0; i < items.size(); i++) {
         std::cout << "item " << i + 1 << " =" << items[i] << '\n';
      }
      std::cout << "\n";
   }
   void Order::graph(std::fstream& gv)
   {
      if (!customer_name.empty()) {
         for (int i = 0; i < items.size(); i++) {
            gv << '"' << customer_name << "\n" << product_name << '"';
            gv << "->" << '"' << items[i] << '"' << '\n';
         }
      }
   }
   OrderManager::OrderManager(std::vector<std::vector<std::string>>& csvDataTask)
   {
      for (auto& line : csvDataTask) {
         //for (int line = 0; line < csvDataTask.size(); line++) {
         orderList.push_back(Order(line));
      }
   }
   bool OrderManager::validate(ItemManager & items)
   {
      int errors = 0;
      for (int i = 0; i < orderList.size(); i++) {
         for (int j = 0; j < orderList[i].itemSize(); j++) {
            std::string item = orderList[i].itemName(j);
            if (items.find(item) == nullptr) {
               errors++;
               std::cerr << "Could not find " << item << " from " << orderList[i].custName() << " order" << '\n';
            }
         }
      }
      if (errors) std::cerr << "Number of errors: " << errors << '\n';
      return !errors;
   }
   void OrderManager::print()
   {
      for (auto& t : orderList) {
         t.print();
      }
   }
   void OrderManager::graph(const std::string& file)
   {
      std::fstream fout(file + ".gv", std::ios::out | std::ios::trunc);
      if (fout.is_open()) {
         fout << "digraph taskgraph { \n";
         for (auto& t : orderList) {
            t.graph(fout);
         }
         fout << "}";
         fout.close();
      }
   }
}