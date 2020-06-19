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

#include "Synapse.cuh"

class CA : public Synapse{

private:
    // Default values for CA intiation
    static int d_n_neuron;
    static float d_activity;
    static float d_connectivity;
    static float d_inhibitory;
    static float d_threshold;
    static float d_C[7];
    static int d_available;

protected:
    // Data
    int n_threshold;
    int n_inhibitory;
    int n_excitatory;
    int n_activation;
    bool ignition;

    // Constant Parameters
    float theta; // firing threshold
    float c_decay; // decay constant d
    float f_recover; // recovery constant F^R
    float f_fatigue; // fatigue constant F^C

    // Updates
    void updateFlags(std::vector<bool>& flag_vec,
                     const std::vector<float>& energy_vec,
                     const std::vector<float>& fatigue_vec,
                     const float& theta);
    void updateE(std::vector<float>& energy_vec,
                 const std::vector<std::vector<float>>& weight_vec,
                 const std::vector<bool>& flag_vec,
                 const int& c_decay);
    void updateF(std::vector<float>& fatigue_vec,
                 const std::vector<bool>& flag_vec,
                 const float& f_fatigue,
                 const float& f_recover);

    //Methods
    float dotP(const std::vector<float>& weights_vec,
               const std::vector<bool>& flags_vec);

public:
    // Constructors - Destructors
    CA(int n = d_n_neuron,
       float activity = d_activity,
       float connectivity = d_connectivity,
       float inhibitory = d_inhibitory,
       float threshold = d_threshold,
       float* C = d_C);
    ~CA();

    // Running
    void runFor(int timeStep, int available = d_available);
    void update();

    // GET
    bool getIgnition();

    // Proof of Concept
    static void POC();
};

#endif // CA_H