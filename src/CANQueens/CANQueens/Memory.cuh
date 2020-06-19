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

#include "FLIF.cuh"

class Memory : public FLIF{

private:
    static int d_n_neuron;
    static float d_activity;
    static float d_connectivity;
    static float d_inhibitory;
    static float d_alpha;
    static int d_available;

protected:
    float alpha; // learning rate
    float w_average; // constant representing average total synaptic strength of the pre-synaptic neuron.
    float w_current; // current total synaptic strength

    // Updates
    void updateFlags(std::vector<bool>& flag_vec,
                     const float& activity);
    void updateWeights(std::vector<std::vector<float>>& weight_vec,
                       const std::vector<bool>& pre_vec,
                       const std::vector<bool>& post_vec,
                       const float& alpha,
                       const float& w_average,
                       const float& w_current);
    
public:
    Memory(int n = d_n_neuron,
           float activity = d_activity,
           float connectivity = d_connectivity,
           float inhibitory = d_inhibitory,
           float alpha = d_alpha);
    ~Memory();
    
    // Running
    void runFor(int timeStep, int available = d_available);
    void update(float act = -1.0f);

    // Set
    void setActivity(float act);

    // Proof of Concept
    static void POC();
};

#endif // MEMORY_H