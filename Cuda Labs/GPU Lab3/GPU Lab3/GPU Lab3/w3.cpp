// Device Query and Selection - Workshop 3
// w3.cpp

#include <iostream>
#include <cstdlib>
#include <cstring>
#include <cuda_runtime.h> // CUDA run-time header file


int main(int argc, char** argv) {
   bool selectADevice = argc == 3;
   bool listAllDevices = argc == 1;
   int rc = 0;

   if (selectADevice) {
      int device;
      int major = std::atoi(argv[1]); // major version - compute capability 
      int minor = std::atoi(argv[2]); // minor version - compute capability

      cudaDeviceProp findProp;
      findProp.major = major;
      findProp.minor = minor;
                                             // choose a device close to compute capability maj.min
                                             // - fill the properties struct with the user-requested capability
      cudaChooseDevice(&device, &findProp);  // - retrieve the device that is the closest match
                                             // - retrieve the properties of the selected device

      std::cout << "Device with compute capability " << major << '.' <<
         minor << " found (index " << device << ')' << std::endl;
   } else if (listAllDevices) {
      int noDevices;

      cudaGetDeviceCount(&noDevices);  // retrieve the number of installed devices

      for (int device = 0; device < noDevices; ++device) {

         cudaDeviceProp properties; // retrieve the properties of device i_dev
         cudaGetDeviceProperties(&properties, device);

         std::cout << "Name:                " << properties.name
            << std::endl;
         std::cout << "Compute Capability:  " << properties.major
            << '.' << properties.minor
            << std::endl;
         std::cout << "Total Global Memory: " << properties.totalGlobalMem
            << std::endl;
      }
      if (noDevices == 0) {
         std::cout << "No Device found " << std::endl;
      }
   } else {
      std::cout << "***Incorrect number of arguments***\n";
      rc = 1;
   }

   system("pause");
   return rc;
}