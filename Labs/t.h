#pragma once
#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include "util.h"
namespace oop345 {

   class Task
   {
   private:
      bool validTaskName(std::string& s);
      bool validSlot(std::string& s);
      std::string taskName, taskSlots, taskPass, taskFail;
   public:
      Task() {}
      Task(std::vector<std::string>& line);
      Task(const Task* t);
      std::string name() { return taskName; }
      std::string pass() { return taskPass; }
      std::string fail() { return taskFail; }

      void print();
      void graph(std::fstream& gv);

   };
   class TaskManager
   {
   private:
      std::vector <Task> taskList;
   public:
      TaskManager() {}
      TaskManager(std::vector<std::vector<std::string>>& csvDataTask);
      Task* find(std::string& t);
      bool validate();
      int size() { return taskList.size(); }
      void print();
      void graph(const std::string& file);
      Task* task(int i) { return& taskList[i]; }
   };
}