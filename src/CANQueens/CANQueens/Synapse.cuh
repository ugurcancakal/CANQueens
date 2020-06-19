/* Synapse Class Header File
 *
 * 200619
 * author = @ugurc
 * ugurcan.cakal@gmail.com
 */

#ifdef __CUDACC__
#define CUDA_CALLABLE_MEMBER __host__ __device__
#else
#define CUDA_CALLABLE_MEMBER
#endif 

#ifndef SYNAPSE_H
#define SYNAPSE_H

#include "FLIF.cuh"

class Synapse : public FLIF{
private:

protected:
    float connectivity;
    float inhibitory;
    float alpha; // learning rate
    float w_average; // constant representing average total synaptic strength of the pre-synaptic neuron.
    float w_current; // current total synaptic strength

    // Synapse ==== FLIF
    std::vector<bool> pre_flags;
    std::vector<bool> post_flags;

    std::vector<FLIF*> incomingList;
    std::vector<FLIF*> outgoingList;

    // Init === FLIF
    void initWeights(int in, int out, float connectivity, float inhibitory,
        std::vector<std::vector<float>>& weight_vec);

    // Update
    void updateWeights(std::vector<std::vector<float>>& weight_vec,
        const std::vector<bool>& pre_vec,
        const std::vector<bool>& post_vec,
        const float& alpha,
        const float& w_average,
        const float& w_current);

    void updatePre(std::vector<bool>& pre_synaptic_flags,
        const std::vector<FLIF*>& incoming);

    void updatePost(std::vector<bool>& post_synaptic_flags,
        const std::vector<FLIF*>& outgoing);

    // Connect
    void addIncomingWeights(std::vector<std::vector<float>>& resting,
        const std::vector<std::vector<float>>& in);
    void addOutgoingWeights(std::vector<std::vector<float>>& resting,
        const std::vector<std::vector<float>>& out);
    

public:
    Synapse();
    ~Synapse();

    // Connecting
    void connectIn(FLIF* incoming,
        float strength,
        float inhibitory);
    void connectOut(FLIF* outgoing,
        float strength,
        float inhibitory);
    static void connect(Synapse* pre_synaptic, float pre_strength, float pre_inhibitory,
                        Synapse* post_synaptic, float post_strength, float post_inhibitory);
};

#endif // SYNAPSE_H