/* Memory Class Header File
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

#ifndef MEMORY_H
#define MEMORY_H

#include "Synapse.cuh"

class Memory : public Synapse{

private:
    static int d_n_neuron;
    static float d_activity;
    static float d_connectivity;
    static float d_inhibitory;
    static float d_alpha;
    static int d_available;

public:
    Memory(int n = d_n_neuron,
           float activity = d_activity,
           float connectivity = d_connectivity,
           float inhibitory = d_inhibitory,
           float alpha = d_alpha);
    ~Memory();
    
    // Running
    void runFor_CPU(int timeStep, int available = d_available);
    void update_CPU(float act = -1.0f);

    void runFor_GPU(int timeStep, int available = d_available);
    void update_GPU(float act = -1.0f);

    // Proof of Concept
    static void POC_CPU();
    static void POC_GPU();
    void initMemoryGPU();
};

#endif // MEMORY_H