/* Memory Class Source File
 *
 * 200517
 * author = @ugurc
 * ugurcan.cakal@gmail.com
 */

#include "Memory.cuh"

int Memory::d_n_neuron = 10;
float Memory::d_activity = 0.5f;
float Memory::d_connectivity = 1.0f;
float Memory::d_inhibitory = 0.2f;
float Memory::d_alpha = 0.2f;
int Memory::d_available = 0b1001;


Memory::Memory(int n, float activity, float connectivity, float inhibitory, float alpha) {
    std::cout << "Memory constructed" << std::endl;

    this->n_neuron = n;
    this->activity = activity;
    this->connectivity = connectivity;
    this->inhibitory = inhibitory;

    this -> alpha = alpha;
    this -> w_average = 1.0f;
    this -> w_current = 1.0f;

    // Neurons
    initFlags(n, activity, this->flags);
    initWeights(n, n, connectivity, inhibitory, this->weights);
}

Memory::~Memory() {
    std::cout << "Memory destructed" << std::endl;
}

void Memory::updateFlags(std::vector<bool>& flag_vec,
                         const float& activity) {

    std::vector<bool>::iterator it;
    for (it = flag_vec.begin(); it < flag_vec.end(); it++) {
        *it = 0 == (rand() % static_cast<int>(floor(1.0f / activity)));
    }
}

void Memory::updateWeights(std::vector<std::vector<float>>& weight_vec,
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

// Set
void Memory::setActivity(float act) {
    this->activity = act;
}

void Memory::POC() {
    int timeStep = 10;
    Memory* myMEM;
    myMEM = new Memory(3);

    myMEM->runFor(timeStep);
    std::cout << myMEM->getActivity() << std::endl;

    myMEM->setActivity(0.1);

    myMEM->runFor(timeStep);
    std::cout << myMEM->getActivity() << std::endl;
}

void Memory::update(float act) {
    /* Update the CA by updating neuron related data structures
     * ! pre_synaptic and post_synaptic not in use
     */
    if (act >= 0.0f && act < 1.0f) {
        setActivity(act);
    }
    updateWeights(this->weights, this->flags, this->flags, this->alpha, this->w_average, this->w_current);
    updateFlags(this->flags, this->activity);
}

// Running
void Memory::runFor(int timeStep, int available) {
    /* Run the CA for defined timestep and record the activity
     * Implemented for raster plot drawing
     *
     * Parameters:
     *      timestep(int):
     *          number of steps to stop running
     */
    for (int i = 0; i < timeStep; i++) {
        record.push_back(setRecord(available));
        //std::cout << record[0].energy.size() << std::endl;
        update();
    }

}