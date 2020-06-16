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

#include <string>
#include <iostream>
#include <vector>

struct record_m {
    std::vector<bool> flags;
    std::vector<std::vector<float>> weights;
};

class Memory {

private:
    static int d_n_neuron;
    static float d_act;
    static float d_alpha;
    static float d_inh;
    static float d_conn;

protected:
    float activation;
    int n_activation;
    int n_neuron;
    float alpha; // learning rate
    float w_average; // constant representing average total synaptic strength of the pre-synaptic neuron.
    float w_current; // current total synaptic strength
    float connectivity; // connectivity rate
    float inhibitory;
    int n_inh;

    std::vector<record_m> activity;

    // Neurons
    std::vector<bool> flags; // firing flags phi
    std::vector<std::vector<float>> weights; // connection weights weights[n_id][to_neuron]

    // Updates
    void updateFlags();
    void updateWeights();

    // Inits
    void initFlags(int n, int n_act);
    void initWeights(int n, float connectivity, bool print);

    template <typename T>
    std::string vectorToString(std::vector<T>& vec) {
        std::string temp = "";
        typename std::vector<T>::iterator it;
        for (it = vec.begin(); it < vec.end(); it++) {
            temp += std::to_string(*it) + " ";
        }
        return temp;
    }

    int num_fire(std::vector<bool>& firings);
    
public:
    Memory(int n = d_n_neuron,
        float act = d_act,
        float r_l = d_alpha,
        float inh = d_inh,
        float conn = d_conn, 
        bool print = false);
    ~Memory();
    std::string toString();
    
    // Running
    void runFor(int timeStep);
    void update();
    void updateA(float act);

    std::string getActivity(int timeStep = 0);


};

#endif // MEMORY_H