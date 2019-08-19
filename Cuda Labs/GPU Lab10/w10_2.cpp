 // Workshop 10 - Matrix Multiply using OpenCL
 // w10.cpp

 #include <iostream>
 #include <iomanip>
 #include <fstream>
 #include <cstdlib>
 // add OpenCL header file

 using namespace std;

 const int ntpb = 16;  // number of work units per workgroup

 inline void checkError(cl_int status, const char* name) {
    if (status != CL_SUCCESS) {
        std::cout << "Error: " << name << " (" << status << ") " << std::endl; 
        switch (status) {
            case CL_SUCCESS:
                std::cout << "Success!"; break;
            case CL_DEVICE_NOT_FOUND:
                std::cout << "Device not found."; break;
            case CL_DEVICE_NOT_AVAILABLE:
                std::cout << "Device not available"; break;
            case CL_COMPILER_NOT_AVAILABLE:
                std::cout << "Compiler not available"; break;
            case CL_MEM_OBJECT_ALLOCATION_FAILURE:
                std::cout << "Memory object allocation failure"; break;
            case CL_OUT_OF_RESOURCES:
                std::cout << "Out of resources"; break;
            case CL_OUT_OF_HOST_MEMORY:
                std::cout << "Out of host memory"; break;
            case CL_PROFILING_INFO_NOT_AVAILABLE:
                std::cout << "Profiling information not available"; break;
            case CL_MEM_COPY_OVERLAP:
                std::cout << "Memory copy overlap"; break;
            case CL_IMAGE_FORMAT_MISMATCH:
                std::cout << "Image format mismatch"; break;
            case CL_IMAGE_FORMAT_NOT_SUPPORTED:
                std::cout << "Image format not supported"; break;
            case CL_BUILD_PROGRAM_FAILURE:
                std::cout << "Program build failure"; break;
            case CL_MAP_FAILURE:
                std::cout << "Map failure"; break;
            case CL_INVALID_VALUE:
                std::cout << "Invalid value"; break;
            case CL_INVALID_DEVICE_TYPE:
                std::cout << "Invalid device type"; break;
            case CL_INVALID_PLATFORM:
                std::cout << "Invalid platform"; break;
            case CL_INVALID_DEVICE:
                std::cout << "Invalid device"; break;
            case CL_INVALID_CONTEXT:
                std::cout << "Invalid context"; break;
            case CL_INVALID_QUEUE_PROPERTIES:
                std::cout << "Invalid queue properties"; break;
            case CL_INVALID_COMMAND_QUEUE:
                std::cout << "Invalid command queue"; break;
            case CL_INVALID_HOST_PTR:
                std::cout << "Invalid host pointer"; break;
            case CL_INVALID_MEM_OBJECT:
                std::cout << "Invalid memory object"; break;
            case CL_INVALID_IMAGE_FORMAT_DESCRIPTOR:
                std::cout << "Invalid image format descriptor"; break;
            case CL_INVALID_IMAGE_SIZE:
                std::cout << "Invalid image size"; break;
            case CL_INVALID_SAMPLER:
                std::cout << "Invalid sampler"; break;
            case CL_INVALID_BINARY:
                std::cout << "Invalid binary"; break;
            case CL_INVALID_BUILD_OPTIONS:
                std::cout << "Invalid build options"; break;
            case CL_INVALID_PROGRAM:
                std::cout << "Invalid program"; break;
            case CL_INVALID_PROGRAM_EXECUTABLE:
                std::cout << "Invalid program executable"; break;
            case CL_INVALID_KERNEL_NAME:
                std::cout << "Invalid kernel name"; break;
            case CL_INVALID_KERNEL_DEFINITION:
                std::cout << "Invalid kernel definition"; break;
            case CL_INVALID_KERNEL:
                std::cout << "Invalid kernel"; break;
            case CL_INVALID_ARG_INDEX:
                std::cout << "Invalid argument index"; break;
            case CL_INVALID_ARG_VALUE:
                std::cout << "Invalid argument value"; break;
            case CL_INVALID_ARG_SIZE:
                std::cout << "Invalid argument size"; break;
            case CL_INVALID_KERNEL_ARGS:
                std::cout << "Invalid kernel arguments"; break;
            case CL_INVALID_WORK_DIMENSION:
                std::cout << "Invalid work dimension"; break;
            case CL_INVALID_WORK_GROUP_SIZE:
                std::cout << "Invalid work group size"; break;
            case CL_INVALID_WORK_ITEM_SIZE:
                std::cout << "Invalid work item size"; break;
            case CL_INVALID_GLOBAL_OFFSET:
                std::cout << "Invalid global offset"; break;
            case CL_INVALID_EVENT_WAIT_LIST:
                std::cout << "Invalid event wait list"; break;
            case CL_INVALID_EVENT:
                std::cout << "Invalid event"; break;
            case CL_INVALID_OPERATION:
                std::cout << "Invalid operation"; break;
            case CL_INVALID_GL_OBJECT:
                std::cout << "Invalid OpenGL object"; break;
            case CL_INVALID_BUFFER_SIZE:
                std::cout << "Invalid buffer size"; break;
            case CL_INVALID_MIP_LEVEL:
                std::cout << "Invalid mip-map level"; break;
            default: cout << "Unknown";
        }
        std::cout << std::endl;
        exit (EXIT_FAILURE);
    }
 }

 int main(int argc, char* argv[]) {
     if (argc != 3) {
         std::cerr << "***Incorrect number of arguments***\n";
         return 1;
     }
     int   n = atoi(argv[1]) * ntpb;
     int  nb = n * n * sizeof(float);
     float run_time_gpu;
     // allocate host memory
     float* a = new float[n * n];
     float* b = new float[n * n];
     float* c = new float[n * n];
     // initialize host memory
     for (int i = 0; i < n * n; i++)
         a[i] = b[i] = 0;
     for (int i = 0; i < n * n; i += n + 1)
         a[i] = b[i] = 1.0f;

     // Load Device Program from argv[2]
     ifstream f(argv[2]);
     char cc;
     size_t size = 0;
     while (f) {
         f.get(cc);
         size++;
     }
     f.clear();
     f.seekg(0);
     char* src = new char[size+1];
     size = 0;
     while (f)
         f.get(src[size++]);
     f.close();
     if (size) src[--size] = '\0'; // overwrite eof

     // Platform Model
     //===============

     // get platform info

     // get device info

     // Execution Model
     //================

     // create context

     // create command queue for the device

     // create memory buffers on the device

     // Program Model
     //==============

     // create program from src[]

     // build program

     // if errors encountered build log and send to output

     // create kernel

     // set kernel arguments

     // Execute
     //========

     // copy to buffers on the device

     // define execution configuration

     // launch kernel

     // extract profiling information (run_time_gpu)

     // copy to host memory (c) from the device buffer

     // Release OpenCL Resources
     //=========================

     // add code here

     // output errors only
     int ne = 0;
     std::cout << fixed << setprecision(6);
     for (int i = 0; i < n * n; i += n + 1)
         if (c[i] != 1.0f)
             std::cout << setw(3) << ++ne << ' ' <<
                  c[i] << endl;
     if (ne)
         std::cout << ne << " Errors encountered" << endl;
     else
         std::cout << "No Errors encountered" << endl;
         std::cout << argv[2] << " kernel took " <<
          run_time_gpu << " microsecs" << endl;

     // deallocate host memory
     delete [] a;
     delete [] b;
     delete [] c;
     delete [] src;
 }