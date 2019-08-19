#pragma once
#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include "util.h"
#include "t.h"
namespace oop345 {

      bool Task::validTaskName(std::string& s)
      {
         if (s.empty()) return false;
         for (auto& c : s) //check each character 'c' in string 'c'
            if (!(isalnum(c) || c == ' '))
               return false;
         return true;
      }
      bool Task::validSlot(std::string& s)
      {
         if (s.empty()) return false;
         for (auto& c : s) //check each character 'c' in string 'c'
            if (!isdigit(c))
               return false;
         return true;
      };
      Task::Task(std::vector<std::string>& line)
      {
         switch (line.size()) {
         case 4:
            if (validTaskName(line[3]))
               taskFail = line[3];
            else
               throw std::string("Expected a fail task name, found ") + line[3];
         case 3:
            if (validTaskName(line[2]))
               taskPass = line[2];
            else
               throw std::string("Expected a fail task name, found ") + line[2];
         case 2:
            if (validSlot(line[1]))
               taskSlots = line[1];
            else
               throw std::string("Expected a slot, found ") + line[1];
         case 1:
            if (validTaskName(line[0]))
               taskName = line[0];
            else
               throw std::string("Expected a fail task name, found ") + line[0];
            break;
         default:
            throw std::string("Expected 1, 2, 3, 4 fields, not ") + std::to_string(line.size());
            break;
         }
      }
      Task::Task(const Task * t)
      {
         taskName = t->taskName;
         taskSlots = t->taskSlots;
         taskPass = t->taskPass;
         taskFail = t->taskFail;
      }
      void Task::print()
      {
         std::cout << "name=" << taskName << '\n';
         std::cout << "slot=" << taskSlots << '\n';
         std::cout << "pass=" << taskPass << '\n';
         std::cout << "fail=" << taskFail << '\n';
         std::cout << "\n";
      }
      void Task::graph(std::fstream& gv)
      {
         if (!taskPass.empty()) {
            gv << '"' << taskName << '"';
            gv << "->";
            gv << '"' << taskPass << '"';
            gv << "[color=green];\n";
         }
         if (!taskFail.empty()) {
            gv << '"' << taskName << '"';
            gv << "->";
            gv << '"' << taskFail << '"';
            gv << "[color=red];\n";
         }
         if (taskPass.empty() && taskFail.empty()) {
            gv << '"' << taskName << '"';
            gv << ";\n";
         }
      }
      TaskManager::TaskManager(std::vector<std::vector<std::string>>& csvDataTask)
      {
         for (auto& line : csvDataTask) {
            //for (int line = 0; line < csvDataTask.size(); line++) {
            taskList.push_back(Task(line));
         }
      }
      Task * TaskManager::find(std::string & t)
      {
         for (size_t i = 0; i < taskList.size(); i++) {
            if (taskList[i].name() == t) {
               return& taskList[i];
            }
         }
         return nullptr;
      }
      bool TaskManager::validate()
      {
         int errors = 0;
         for(auto& task : taskList) {
            std::string pass = task.pass();
            if(!pass.empty() && (find(task.pass()) == nullptr)) {
               errors++;
               std::cerr << "Cannot find pass task" << pass << "\n";
            }
            std::string fail = task.fail();
            if(!fail.empty() && (find(task.fail()) == nullptr)) {
               errors++;
               std::cerr << "Cannot find fail task" << fail << "\n";
            }
         }
         if(errors) std::cerr << "Error count: " << errors << "\n";
         return !errors;
      }
      void TaskManager::print()
      {
         for (auto& t : taskList) {
            t.print();
         }
      }
      void TaskManager::graph(const std::string& file)
      {
         std::fstream fout(file + ".gv", std::ios::out | std::ios::trunc);
         if (fout.is_open()) {
            fout << "digraph taskgraph { \n";
            for (auto& t : taskList) {
               t.graph(fout);
            }
            fout << "}";
            fout.close();
         }
      }
}