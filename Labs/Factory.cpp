//TODO: MS5
#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include "util.h"
#include "t.h"
#include "i.h"
#include "o.h"

using namespace std;


namespace oop345 {
   class Machine : public Task {
      int incoming = 0;
   public:
      Machine() {}
      Machine(const Task* t) : Task(t) {}
      bool isSource() {}
      void incomingRoutes() { incoming++; }
   };

   class Factory {
      vector<Machine> machineList;
   public:
      Factory(TaskManager tm) {

         //Make all machines
         for (int m = 0; m < tm.size(); m++) {
            machineList.push_back(Machine(tm.task(m)));
         }
         for (auto& machine : machineList) {
            cout << "Machine created named: " << machine.name() << endl;
         }
         //mark sing source and singletons
         for (auto& machine : machineList) {
            string outgoing = machine.pass();
            Task* outgoingTask = tm.find(outgoing);
         }
      }
   };


}
using namespace oop345;
int main() {

   //read and create from files
   vector<vector<string>> csvOrderData;
   vector<vector<string>> csvItemData;
   vector<vector<string>> csvTaskData;
   std::string orderFileName = "SimpleOrder.dat";
   std::string itemFileName = "SimpleItem.dat";
   std::string taskFileName = "SimpleTask.dat";

   csvRead(orderFileName, '|', csvOrderData);
   csvRead(itemFileName, '|', csvItemData);
   csvRead(taskFileName, '|', csvTaskData);

   try {
      TaskManager taskMan(csvTaskData);
      //taskMan.print();
      taskMan.graph("taskGraph");

      OrderManager orderMan(csvOrderData);
      //orderData.print();
      orderMan.graph("orderGraph");

      ItemManager itemMan(csvItemData);
      //ItemData.print();
      itemMan.graph("itemGraph");

      //check consistency
      //find find task pass and fail must be in task data
      if (
         !taskMan.validate() ||
         !itemMan.validate() ||
         !orderMan.validate(itemMan)
         ) {
         cout << "Please repair job files" << endl;
      }
      
      Factory simulation(taskMan);
   }
   catch (const std::string& e) {
      std::cout << e;
   };
   cout << "***************" << endl;
   cout << "Program Exiting" << endl;
   cout << "***************" << endl << endl;
   return 0;
}