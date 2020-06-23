/* Controller Class Header File
 *
 * 200613
 * author = @ugurc
 * ugurcan.cakal@gmail.com
 */

#ifdef __CUDACC__
#define CUDA_CALLABLE_MEMBER __host__ __device__
#else
#define CUDA_CALLABLE_MEMBER
#endif 

#ifndef CONTROLLER_H
#define CONTROLLER_H

#include <string>
#include <iostream>
#include <time.h>
#include "Board.cuh"
#include "Value.cuh"
#include "Memory.cuh"
#include "Explore.cuh"

struct VAL {
    int fitness;
    float activity;
};

class Controller {
protected:

    int n_neuron;
    Board* board;
    Value* value;
    Explore* explore;
    Memory* memory;
    int* chromosome;

    std::vector<VAL> valueRec;
    std::vector<std::vector<int>> choromosomeRec;

    void step_CPU();
    void step_GPU();
    
    //Methods
    std::string dateTimeStamp(const char* filename);
    void saveChCSV(char* filename, int stop = -1, int start = 0);
    void saveValueCSV(char* filename, int stop = -1, int start = 0);
    void saveInfo(char* filename);
public:
    
    void runFor_CPU(int stepSize);
    void runFor_GPU(int stepSize);
    Controller(int n = 8);
    ~Controller();
    std::string toString();
    //CUDA_CALLABLE_MEMBER std::string toString();
    void saveLog();
    void connect_CPU();
    void connect_GPU();


    static void POC_CPU();
    static void POC_GPU();

    static Controller getControllerCPU(int n);
    static Controller getControllerGPU(int n);
};

#endif // CONTROLLER_H