/* Memory Class Source File
 *
 * 200517
 * author = @ugurc
 * ugurcan.cakal@gmail.com
 */

#include "Memory.cuh"

int Memory::d_n_neuron = 10;
float Memory::d_act = 0.5f;
float Memory::d_alpha = 0.2f;
float Memory::d_inh = 0.2f;
float Memory::d_conn = 1.0f;

void Memory::initFlags(int n, int n_act) {
    /* Initialize firing flags randomly
     */
    flags.resize(n);
    std::vector<bool>::iterator it;
    for (it = flags.begin(); it < flags.end(); it++) {
        *it = 0 == (rand() % n_activation);
    }
}

Memory::Memory(int n, float act, float r_l, float inh, float conn, bool print) {
    std::cout << "Memory constructed" << std::endl;

    n_neuron = n;
    updateA(act);
    alpha = r_l;
    connectivity = conn;
    inhibitory = inh;
    n_inh = floor(1.0f / inh);
    w_average = 1.0f;
    w_current = 1.0f;

    // Neurons
    initFlags(n, n_activation);
    initWeights(n, conn, print);

    //if (print) {
    //    std::cout << "\nExplore constructed with ID: " << ID << ": " << std::endl
    //        << n_excitatory << " excitatory, " << std::endl
    //        << n_inhibitory << " inhibitory neurons; " << std::endl
    //        << n_threshold << " of neurons active threshold\n" << std::endl;

    //    std::cout << "Constant Parameters " << std::endl
    //        << "firing threshold : " << theta << std::endl
    //        << "decay constant : " << c_decay << std::endl
    //        << "recovery constant F^R : " << f_recover << std::endl
    //        << "fatigue constant F^C : " << f_fatigue << std::endl
    //        << "learning rate : " << alpha << std::endl
    //        << "average total synaptic strength : " << w_average << std::endl
    //        << "current total synaptic strength : " << w_current << std::endl;
    //}
}

Memory::~Memory() {
    std::cout << "Memory destructed" << std::endl;
}

std::string Memory::toString() {
    return "\Memory activity: " + std::to_string(activation)
        + "\n(" + std::to_string(num_fire(flags)) +
        "/" + std::to_string(n_neuron) + ")\n" +
        vectorToString<bool>(flags);
}

void Memory::updateFlags() {
    std::vector<bool>::iterator it;
    for (it = flags.begin(); it < flags.end(); it++) {
        *it = 0 == (rand() % n_activation);
    }
}

void Memory::updateWeights() {
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
    if (weights.size() != flags.size()) {
        std::cout << "Weight matrix row size is different than flag vector size!" << std::endl;
        return;
    }

    // Iterators
    std::vector<bool>::iterator it;
    std::vector<bool>::iterator it_f;
    std::vector<float>::iterator it_weight;

    std::vector<std::vector<float>>::iterator it_w;
    it_w = weights.begin();

    // Update
    for (it = flags.begin(); it < flags.end(); it++) { // pre_synaptic
        //std::cout << "PRE " << *it << std::endl;
        if (*it) {
            it_f = flags.begin(); // post_synaptic

            for (it_weight = (*it_w).begin(); it_weight < (*it_w).end(); it_weight++) {
                //std::cout << "POST " << *it_f << std::endl;
                sign = (*it_weight) / abs(*it_weight);
                if (*it_f) {
                    delta = alpha * (1.0f - abs(*it_weight)) * exp(w_average - w_current);
                }
                else {
                    delta = (-1.0f) * alpha * abs(*it_weight) * exp(w_current - w_average);
                }
                *it_weight += sign * delta;
                it_f++;
            }
        }
        it_w++;
    }
}

// Inits
void Memory::initWeights(int n, float connectivity, bool print) {
    /* Initialize neuron weights randomly
     * ! all connected for now but it is required to
     * be defined by a parameter
     * Sign of the weigth is determined by the inhibitory
     * or exhibitory characteristic of the neuron
     *
     * An example connection map:
     * -------------------
     * | - - + + + + + + |
     * | - - + + + + + + |
     * | - - + + + + + + |
     * | - - + + + + + + |
     * | - - + + + + + + |
     * | - - + + + + + + |
     * | - - + + + + + + |
     * | - - + + + + + + |
     * -------------------
     * 0<w<1 initially
     *
     * Parameters:
     *      connectivity(float):
     *          connectivity ratio inside CA.
     *          1.0 means fully connected.
     *      print(bool):
     *          print the weights or not
     */

    float sign = -1.0f;
    weights.resize(n);
    std::vector<std::vector<float>>::iterator it;
    for (it = weights.begin(); it < weights.end(); it++) {
        (*it).resize(n);
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

    for (it_w = weights.begin(); it_w < weights.end(); it_w++) {
        for (it_weight = (*it_w).begin(); it_weight < (*it_w).end(); it_weight++) {
            sign = (rand() % n_inh) == 0 ? -1.0f : 1.0f;
            if (static_cast <float> (rand()) / static_cast <float> (RAND_MAX) < connectivity) {
                *it_weight = sign * static_cast <float> (rand()) / static_cast <float> (RAND_MAX);
            }
            else {
                *it_weight = 0.0f;
            }
            if (print) {
                std::cout << *it_weight << " " << std::endl;
            }
            counter++;
        }
        counter = 0;
    }
}

void Memory::updateA(float act) {
    activation = act;
    n_activation = floor(1.0f / act);
}

void Memory::update() {
    /* Update the CA by updating neuron related data structures
     * ! pre_synaptic and post_synaptic not in use
     */
    updateWeights();
    updateFlags();
}

// Running
void Memory::runFor(int timeStep) {
    record_m temp;
    for (int i = 0; i < timeStep; i++) {
        temp.flags = flags;
        temp.weights = weights;
        activity.push_back(temp);
        update();
    }
}

std::string Memory::getActivity(int timeStep) {
    std::string temp = "\n";
    std::vector<record_m>::iterator it_a;
    int count = 0;
    if (timeStep == 0) {
        timeStep = activity.size();
    }

    for (it_a = activity.begin(); it_a < activity.end(); it_a++) {
        temp += "timeStep " + std::to_string(count) +
            "(" + std::to_string(num_fire((*it_a).flags)) +
            "/" + std::to_string(n_neuron) + ")" + "\n";
        
        temp += "\nFlags \n";
        temp += vectorToString<bool>((*it_a).flags);

        temp += "\n\nWeights \n";
        std::vector<std::vector<float>>::iterator it_w;
        for (it_w = (*it_a).weights.begin(); it_w < (*it_a).weights.end(); it_w++) {
            temp += "|" + vectorToString<float>(*it_w) + "|\n";
        }

        temp += "\n\n";
        count++;
    }

    return temp;
}

int Memory::num_fire(std::vector<bool>& firings) {
    /* Number of neurons fired in a given flag vector.
     *
     * Parameters:
     *      firings(std::vector<bool>&):
     *          firing flag vector consisting of booleans
     *
     * Returns:
     *      num(int):
     *          total number of fire
     */
    int fire = 0;
    std::vector<bool>::iterator it;
    for (it = firings.begin(); it < firings.end(); it++) {
        if (*it) {
            fire++;
        }
    }
    return fire;
}