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

protected:    

    // Updates
    void updateFlags(std::vector<bool>& flag_vec,
                     const float& activity);

public:
    Explore(int n = d_n_neuron,
            float activity = d_activity);
    ~Explore();

    // Running
    void runFor(int timeStep, int available = d_available);
    void update(float act = -1.0f);

    // Set
    void setActivity(float act);
    
    // Proof of Concept
    static void POC();
};

#endif // EXPLORE_H