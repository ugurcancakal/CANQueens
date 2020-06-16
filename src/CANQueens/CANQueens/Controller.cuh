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

class Controller {
private:

    int n_neuron;
    Board* board;
    Value* value;
    Explore* explore;
    Memory* memory;
    int* chromosome;

    void step();
    
public:
    
    void runFor(int stepSize);
    Controller(int n = 8);
    ~Controller();
    std::string toString();
    //CUDA_CALLABLE_MEMBER std::string toString();
};

#endif // CONTROLLER_H