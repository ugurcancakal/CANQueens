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

#include <string>
#include <iostream>
#include <vector>

class Explore{
private:
    static int d_n_neuron;
    static float d_act;

protected:
    float activation;
    int n_activation;
    int n_neuron;

    std::vector<std::vector<bool>> activity;
    
    // Neurons
    std::vector<bool> flags; // firing flags phi

    // Updates
    void updateFlags();

    // Inits
    void initFlags(int n, int n_act);

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
    Explore(int n = d_n_neuron,
            float act = d_act,
            bool print = false);
    ~Explore();
    std::string toString();

    // Running
    void runFor(int timeStep);
    void update(float act);
    void updateA(float act);

    //Activity
    std::string getActivity(int timeStep = 0);
    
    //CUDA_CALLABLE_MEMBER std::string toString();
};

#endif // EXPLORE_H