/* Synapse Class Source File
 *
 * 200619
 * author = @ugurc
 * ugurcan.cakal@gmail.com
 */

#include "Synapse.cuh"

Synapse::Synapse() {
    //std::cout << "Synapse constructed" << std::endl;
    connectivity = 0.0f;
    inhibitory = 0.0f;
    alpha = 0.0f; // learning rate
    w_average = 0.0f; // constant representing average total synaptic strength of the pre-synaptic neuron.
    w_current = 0.0f; // current total synaptic strength
}

Synapse::~Synapse() {
    //std::cout << "Synapse destructed" << std::endl;
}

void Synapse::initWeights(int in, int out, float connectivity, float inhibitory, std::vector<std::vector<float>>& weight_vec) {
    /* Initialize neuron weights randomly
     * Sign of the weigth is determined by the inhibitory neuron rate
     *
     * An example connection map: (10x10, 0.2 inhibitory, 1.0 connectivity)
     * -------------------
     * | - - + + + + + + | <-- incoming line
     * | - - + + + + + + |
     * | - - + + + + + + |
     * | - - + + + + + + |
     * | - - + + + + + + |
     * | - - + + + + + + |
     * | - - + + + + + + |
     * | - - + + + + + + |
     * -------------------
     * 0<w<1
     *
     * Parameters:
     *      in(int):
     *          incoming connections. in = 10 creates 10 rows
     *      out(int):
     *          outgoing connections. out = 10 creates 10 columns
     *      connectivity(float):
     *          connectivity ratio inside network.
     *          1.0 means fully connected.
     *      inhibitory(float):
     *          inhibitory neuron rate inside network.
     *          1.0 full inhibitory and 0.0 means full excitatory.
     *      weight_vec(std::vector<std::vector<float>>&):
     *          reference to weight vector to be filled.
     */
    int n_inh;
    if (inhibitory > 0) {
        n_inh = floor(1.0f / inhibitory);
    }
    else {
        n_inh = -1;
    }
    float sign = -1.0f;
    weight_vec.resize(in);
    std::vector<std::vector<float>>::iterator it;
    for (it = weight_vec.begin(); it < weight_vec.end(); it++) {
        (*it).resize(out);
    }

    // Connectivity range check
    if (connectivity < 0.0f) {
        connectivity = 0.0f;
    }
    else if (connectivity > 1.0f) {
        connectivity = 1.0f;
    }

    // Iterators
    std::vector<std::vector<float>>::iterator it_w;
    std::vector<float>::iterator it_weight;
    int counter = 0;

    for (it_w = weight_vec.begin(); it_w < weight_vec.end(); it_w++) {
        for (it_weight = (*it_w).begin(); it_weight < (*it_w).end(); it_weight++) {
            if (n_inh > 0) {
                sign = (rand() % n_inh) == 0 ? -1.0f : 1.0f;
            }
            else {
                sign = 1.0f;
            }
            if (static_cast <float> (rand()) / static_cast <float> (RAND_MAX) < connectivity) {
                *it_weight = sign * static_cast <float> (rand()) / static_cast <float> (RAND_MAX);
            }
            else {
                *it_weight = 0.0f;
            }
            counter++;
        }
        counter = 0;
    }
}

void Synapse::updateWeights(std::vector<std::vector<float>>& weight_vec,
    const std::vector<bool>& pre_vec,
    const std::vector<bool>& post_vec,
    const float& alpha,
    const float& w_average,
    const float& w_current) {
    /* Update weights of the FLIF neurons inside CA
     * according to hebbian learning rule.
     * That is, neurons fire together, wire together.
     * !! W-current W-average updates are to be done
     * !! w must be between 0<w<1
     * !! Changes incoming weights
     * !! it need to trace all incoming flags
     * !! it_f need to trace all outgoing flags rather than just internal
     * !! the weight between neurons fire together will increase in absolute value
     */
    float delta = 0.0f;
    float sign = 1.0f;
    // Size check
    if (weight_vec[0].size() != pre_vec.size()) {
        std::cout << "Weight matrix width is different than pre synaptic vector size!" << std::endl;
        //return;
    }
    if (weight_vec.size() != post_vec.size()) {
        std::cout << "Weight matrix height is different than post synaptic vector size!" << std::endl;
        //return;
    }

    // Iterators
    std::vector<bool>::const_iterator it_pre;
    std::vector<bool>::const_iterator it_post;
    std::vector<float>::iterator it_weight;
    std::vector<std::vector<float>>::iterator it_w;

    it_post = post_vec.begin();

    for (it_w = weight_vec.begin(); it_w < weight_vec.end(); it_w++) {
        it_pre = pre_vec.begin();
        for (it_weight = (*it_w).begin(); it_weight < (*it_w).end(); it_weight++) {
            if (*it_pre) {
                sign = (*it_weight) / abs(*it_weight);
                if (*it_post) {
                    delta = alpha * (1.0f - abs(*it_weight)) * exp(w_average - w_current);
                }
                else {
                    delta = (-1.0f) * alpha * abs(*it_weight) * exp(w_current - w_average);
                }
                *it_weight += sign * delta;
            }
            it_pre++;
        }
        it_post++;
    }
}

void Synapse::updatePre(std::vector<bool>& pre_synaptic_flags,
    const std::vector<FLIF*>& incoming)
{
    std::vector<FLIF*>::const_iterator it;
    std::vector<bool>::const_iterator it_f;
    pre_synaptic_flags.clear();

    //std::cout << "\nINCOMING SIZE: " << incoming.size() << std::endl;
    for (it = incoming.begin(); it < incoming.end(); it++) {
        //std::cout << "PRE SIZE: " << pre_synaptic_flags.size() << std::endl;
        pre_synaptic_flags.insert(pre_synaptic_flags.end(),
            (*it)->flags.begin(),
            (*it)->flags.end());
    }

    //std::cout << "Updated SIZE: " << pre_synaptic_flags.size() << std::endl;
    //std::cout << getID() << " PRE:\n" << vectorToString<bool>(pre_synaptic_flags) << std::endl;
}

void Synapse::updatePost(std::vector<bool>& post_synaptic_flags,
    const std::vector<FLIF*>& outgoing) {

    std::vector<FLIF*>::const_iterator it;
    std::vector<bool>::const_iterator it_f;
    post_synaptic_flags.clear();

    //std::cout << "\nOUTGOING SIZE: " << outgoing.size() << std::endl;
    for (it = outgoing.begin(); it < outgoing.end(); it++) {
        //std::cout << "POST SIZE: " << post_synaptic_flags.size() << std::endl;
        post_synaptic_flags.insert(post_synaptic_flags.end(),
            (*it)->flags.begin(),
            (*it)->flags.end());
    }

    //std::cout << "Updated SIZE: " << post_synaptic_flags.size() << std::endl;
    //std::cout << getID() << " PRE:\n" << vectorToString<bool>(post_synaptic_flags) << std::endl;
}

// Connect
void Synapse::addIncomingWeights(std::vector<std::vector<float>>& resting,
    const std::vector<std::vector<float>>& in) {

    std::vector<std::vector<float>>::iterator it_w = resting.begin();
    std::vector<std::vector<float>>::const_iterator it_in = in.begin();
    std::vector<float> zeros((*it_in).size(), 0.0);

    if (resting.size() >= in.size()) {
        for (it_in = in.begin(); it_in < in.end(); it_in++) {
            (*it_w).insert((*it_w).end(), (*it_in).begin(), (*it_in).end());
            it_w++;
        }

        for (it_w = it_w; it_w < resting.end(); it_w++) {
            (*it_w).insert((*it_w).end(), zeros.begin(), zeros.end());
        }
    }
    else {
        std::cout << "Smaller Outgoing Vector!" << std::endl;
        for (it_w = resting.begin(); it_w < resting.end(); it_w++) {
            (*it_w).insert((*it_w).end(), (*it_in).begin(), (*it_in).end());
            it_in++;
        }
    }
}

void Synapse::addOutgoingWeights(std::vector<std::vector<float>>& resting,
    const std::vector<std::vector<float>>& out) {

    std::vector<std::vector<float>>::const_iterator it_out;
    std::vector<float> zeros(resting[resting.size() - 1].size() - out[0].size(), 0.0);
    std::vector<float> temp;

    if (resting[resting.size() - 1].size() >= out[0].size()) {
        for (it_out = out.begin(); it_out < out.end(); it_out++) {
            temp = *it_out;
            temp.insert(temp.end(), zeros.begin(), zeros.end());
            resting.push_back(temp);
        }
    }
    else {
        std::cout << "Smaller Incoming Vector!" << std::endl;
        int maxSize = resting[resting.size() - 1].size();
        for (it_out = out.begin(); it_out < out.end(); it_out++) {
            std::vector<float> newVec((*it_out).begin(), (*it_out).begin() + maxSize);
            resting.push_back(newVec);
        }
    }
}

void Synapse::connectIn(FLIF* incoming,
    float strength,
    float inhibitory) {

    (this->incomingList).push_back(incoming);
    std::vector<std::vector<float>> inWeights;
    initWeights(getN(), incoming->getN(), strength, inhibitory, inWeights);
    addIncomingWeights(this->weights, inWeights);
}

void Synapse::connectOut(FLIF* outgoing,
    float strength,
    float inhibitory) {

    (this->outgoingList).push_back(outgoing);
    std::vector<std::vector<float>> outWeights;
    initWeights(outgoing->getN(), getN(), strength, inhibitory, outWeights);
    addOutgoingWeights(this->weights, outWeights);
}

void Synapse::connect(Synapse* pre_synaptic, float pre_strength, float pre_inhibitory,
    Synapse* post_synaptic, float post_strength, float post_inhibitory) {
    post_synaptic->connectIn(pre_synaptic, pre_strength, pre_inhibitory);
    pre_synaptic->connectOut(post_synaptic, post_strength, post_inhibitory);
}