/* Cell Assembly Class Header File
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

#ifndef CA_H
#define CA_H

#include <string>
#include <iostream>
#include <vector>
#include <fstream>
#include <time.h>

class CA {

private:
    int n_neuron;
    int n_inhibitory;
    int n_excitatory;
    int n_threshold;
    bool ignition;
    int ID;
    std::vector<std::vector<bool>> activityRecord; //activityRecord[timeStep][n_id]
    
    // Constant Parameters
    float theta; // firing threshold
    float c_decay; // decay constant d
    float f_recover; // recovery constant F^R
    float f_fatigue; // fatigue constant F^C
    float alpha; // learning rate
    float w_average; // constant representing average total synaptic strength of the pre-synaptic neuron.
    float w_current; // current total synaptic strength

    // Neurons
    std::vector<bool> flags; // firing flags phi
    std::vector<float> energy; // energy levels
    std::vector<float> fatigue; // fatigue levels
    std::vector<std::vector<float>> weights; // connection weights weights[n_id][to_neuron]
    std::vector<float> pre_synaptic; // Pre-synaptic Firing Flags
    std::vector<float> post_synaptic; // Post-synaptic Firing Flags

    // Updates
    
    void updateFlags();
    void updateEF();
    void updateWeights();

    // Inits
    void initWeights(bool print);
    void initFlags();

    //Methods
    float dotP(std::vector<float>& weight, std::vector<bool>& flag);
    //int firingStatus(int timeStep);
    int firingStatus(std::vector<bool>& firings);
    std::string dateTimeStamp(const char* filename);

protected:
    static int d_n_neuron;
    static float d_inh;
    static float d_threshold;
    static float d_C[7];
    static int nextID;

public:
    CA(int n = d_n_neuron, 
        float threshold = d_threshold, 
        float inh = d_inh, 
        float* C = d_C, 
        bool print = false);
    ~CA();
    std::string toString(int timeStep);
    //CUDA_CALLABLE_MEMBER std::string toString();
    bool getIgnition();
    int getID();
    void runFor(int timeStep);
    // Formatting
    std::string getRaster(int timeStep);
    void saveRaster(char* filename, int timeStep);
    
    void update();
};

#endif // CA_H