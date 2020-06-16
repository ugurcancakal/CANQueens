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

struct record {
    std::vector<bool> flags;
    std::vector<std::vector<float>> weights;
    std::vector<float> energy;
    std::vector<float> fatigue;
};

class CA {

private:
    // Default values for CA intiation
    static int d_n_neuron;
    static float d_inh;
    static float d_conn;
    static float d_threshold;
    static float d_C[7];
    static int nextID;

protected:
    // Data
    int n_neuron;
    int n_threshold;
    int n_inhibitory;
    int n_excitatory;
    int ID;
    bool ignition;

    //activityRecord[timeStep][n_id]
    std::vector<std::vector<bool>> activityRecord; 
    std::vector<record> activity;

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
    void updateE();
    void updateF();
    void updateWeights();

    // Inits
    void initWeights(int n, float connectivity, bool print);
    void initFlags(int n);

    //Methods
    float dotP(std::vector<float>& weight, std::vector<bool>& flag);
    int num_fire(std::vector<bool>& firings);
    std::string dateTimeStamp(const char* filename);


    template <typename T>
    std::string vectorToString(std::vector<T>& vec) {
        std::string temp = "";
        typename std::vector<T>::iterator it;
        for (it = vec.begin(); it < vec.end(); it++) {
            temp += std::to_string(*it) + " ";
        }
        return temp;
    }

public:
    // Constructors - Destructors
    CA(int n = d_n_neuron,
       float threshold = d_threshold,
       float inh = d_inh,
       float connectivity = d_conn,
       float* C = d_C, 
       bool print = false);
    ~CA();

    // Running
    void runFor(int timeStep);
    void update();

    // Printing
    std::string toString(int timeStep);
    std::string getRaster(int timeStep);
    void saveRaster(char* filename, int timeStep);

    std::string getActivity(int timeStep = 0);
    void saveActivity(char* filename, int timeStep);

    // GET
    bool getIgnition();
    int getID();       
};

#endif // CA_H