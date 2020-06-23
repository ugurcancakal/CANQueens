/* Explore Class Header File
 *
 * 200516
 * author = @ugurc
 * ugurcan.cakal@gmail.com
 */

#ifdef __CUDACC__
#define CUDA_CALLABLE_MEMBER __host__ __device__
#else
#define CUDA_CALLABLE_MEMBER
#endif 

#ifndef EXPLORE_H
#define EXPLORE_H

#include "FLIF.cuh"

class Explore : public FLIF{
private:
    static int d_n_neuron;
    static float d_activity;
    static int d_available;    

public:
    Explore(int n = d_n_neuron,
            float activity = d_activity);
    ~Explore();

    // Running
    void runFor_CPU(int timeStep, int available = d_available);
    void update_CPU(float act = -1.0f);

    void runFor_GPU(int timeStep, int available = d_available);
    void update_GPU(float act = -1.0f);
    
    // Proof of Concept
    static void POC_CPU();
    static void POC_GPU();
    void initExploreGPU();
};

#endif // EXPLORE_H