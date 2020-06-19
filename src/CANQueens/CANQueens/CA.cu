/* Cell Assembly Class Source File
 * Parent class for Explore and Memory
 * Construct a Cell Assembly composed of FLIF neurons
 * and record the activity. An raster plot of an example
 * CA with 4 neurons 1 inhibitory and 3 excitatory 
 * having 1 neuron fire threshold is given below
 *
 *  N_ID    ||         SPIKE ACTIVITY
 *  --------------------------------------------
 *  0*      ||      |
 *  1       ||      |               |
 *  2       ||      |                       |
 *  3       ||
 *  --------------------------------------------
 *  TIME    ||      0       1       2       3
 *  --------------------------------------------
 *  FIRE    ||      2       0       1       1
 *  --------------------------------------------
 *  IGNIT   ||      1       0       1       1
 *
 * 200516
 * author = @ugurc
 * ugurcan.cakal@gmail.com
 */

#include "CA.cuh"

// Default values for CA initiation
int CA::d_n_neuron = 10;
float CA::d_activity = 0.5f;
float CA::d_connectivity = 1.0f;
float CA::d_inhibitory = 0.2f;
float CA::d_threshold = 0.3f;
float CA::d_C[7] = { 1.0, 4.0, 0.25, 1.0, 0.2, 1.0, 1.0 };
int CA::d_available = 0b1111;

// PROTECTED MEMBERS
// Updates
void CA::updateFlags(std::vector<bool>& flag_vec,
                     const std::vector<float>& energy_vec,
                     const std::vector<float>& fatigue_vec,
                     const float& theta) {
    /* Update the firing flags of the FLIF neurons inside CA
     * according tho energy levels and fatigueness
     */

    // Size check
    if (fatigue_vec.size() != energy_vec.size()) {
        std::cout << "Fatigue vector size is different than energy vector size!" << std::endl;
        return;
    }
    else if (flag_vec.size() != energy_vec.size()) {
        std::cout << "Flag vector size is different than energy and fatigue vectors size!" << std::endl;
        return;
    }

    // Iterators
    std::vector<bool>::iterator it;
    std::vector<float>::const_iterator it_f;
    std::vector<float>::const_iterator it_e;
    it_f = fatigue_vec.begin();
    it_e = energy_vec.begin();

    // Update
    for (it = flag_vec.begin(); it < flag_vec.end(); it++) {
        if (*it_e - *it_f > theta) {
            *it = true;
        }
        else {
            *it = false;
        }
        it_f++;
        it_e++;
    }
}

void CA::updateE(std::vector<float>& energy_vec,
                 const std::vector<std::vector<float>>& weight_vec,
                 const std::vector<bool>& flag_vec,
                 const int& c_decay) {
    /* Update energy levels of the FLIF neurons inside CA
     * according to weights and firing flags
     */

    // Size check
    /*if (weight_vec.size() != energy_vec.size()) {
        std::cout << "Weight matrix row size is different than energy vector size!" << std::endl;
    }*/
    //if (flag_vec.size() != energy_vec.size()) {
    //    std::cout << "flag vector size is different than energy and weight vectors size!" << std::endl;
    //    //return;
    //}
    // Iterators
    std::vector<float>::iterator it_e;
    std::vector<std::vector<float>>::const_iterator it_w;
    std::vector<bool>::const_iterator it_f;
    it_w = weight_vec.begin();
    it_f = flag_vec.begin();

    // Update
    for (it_e = energy_vec.begin(); it_e < energy_vec.end(); it_e++) {
        if (*it_f) {
            *it_e = dotP(*it_w, flag_vec);
        }
        else {
            *it_e = ((1.0f / c_decay) * (*it_e)) + dotP(*it_w, flag_vec);
        }
        it_w++;
        it_f++;
    }
}

void CA::updateF(std::vector<float>& fatigue_vec, 
                 const std::vector<bool>& flag_vec, 
                 const float& f_fatigue,
                 const float& f_recover) {
    /* Update fatigueness of the FLIF neurons inside CA
     * according to recover rate
     */
    // Size Check
    if (fatigue_vec.size() != flag_vec.size()) {
        std::cout << "Fatigue vector size is different than flag vector size!" << std::endl;
    }

    //Iterators
    std::vector<bool>::const_iterator it = flag_vec.begin();
    std::vector<float>::iterator it_f;

    //Update
    for (it_f = fatigue_vec.begin(); it_f < fatigue_vec.end(); it_f++) {
        if (*it) {
            *it_f += f_fatigue;
        }
        else {
            if (*it_f - f_recover > 0.0f) {
                *it_f = *it_f - f_recover;
            }
            else {
                *it_f = 0.0f;
            }
        }
        it++;
    }
}

void CA::updateWeights(std::vector<std::vector<float>>& weight_vec, 
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

void CA::updatePre(std::vector<bool>& pre_synaptic_flags, 
                   const std::vector<CA*>& incoming)
{
    std::vector<CA*>::const_iterator it;
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

void CA::updatePost(std::vector<bool>& post_synaptic_flags, 
                    const std::vector<CA*>& outgoing) {

    std::vector<CA*>::const_iterator it;
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

// Methods
float CA::dotP(const std::vector<float>& weights_vec, 
               const std::vector<bool>& flags_vec) {
    /* Dot product of two vectors
     * 
     * Parameters:
     *      weights(std::vector<float>&):
     *          weight vector consisting of floating point numbers
     *      flags(std::vector<bool>&):
     *          firing flag vector consisting of booleans
     *
     * Returns:
     *      sum(float):
     *          dot product result
     */
    // SIZE CHECK
    if (weights_vec.size() != flags_vec.size()) {
        std::cout << "ID " << getID() << std::endl;
        std::cout << "DOT PRODUCT REQUIRES SIZES TO BE EQUAL" << std::endl;
        std::cout << weights_vec.size() << std::endl;
        std::cout << flags_vec.size() << std::endl;
    }

    float sum = 0.0f;
    int i = 0;
    std::vector<bool>::const_iterator it_f;
    std::vector<float>::const_iterator it_w;
    it_w = weights_vec.begin();
    for (it_f = flags_vec.begin(); it_f < flags_vec.end(); it_f++) {
        if (*it_f) {
            sum += *it_w;
        }
        it_w++;
    }
    return sum;
}

// Connect
void CA::addIncomingWeights(std::vector<std::vector<float>>& resting,
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

void CA::addOutgoingWeights(std::vector<std::vector<float>>& resting,
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

void CA::connectIn(CA* incoming,
    float strength,
    float inhibitory) {

    (this->incomingList).push_back(incoming);
    std::vector<std::vector<float>> inWeights;
    initWeights(getN(), incoming->getN(), strength, inhibitory, inWeights);
    addIncomingWeights(this->weights, inWeights);
}

void CA::connectOut(CA* outgoing,
    float strength,
    float inhibitory) {

    (this->outgoingList).push_back(outgoing);
    std::vector<std::vector<float>> outWeights;
    initWeights(outgoing->getN(), getN(), strength, inhibitory, outWeights);
    addOutgoingWeights(this->weights, outWeights);
}



// PUBLIC MEMBERS
// Constructors - Destructors
CA::CA(int n, 
       float activity, 
       float connectivity,
       float inhibitory,
       float threshold, 
       float* C) {
    /* Constructor
     * Intitialize constant parameters of decision process
     * and data structures which stores information
     * related to neurons
     * 
     * Parameters:
     *      n(int):
     *          number of neurons inside CA
     *      threshold(float):
     *          CA activation threshold 0<t<1
     *      inh(float):
     *          inhibitory neuron rate
     *      connectivity(float):
     *          connectivity ratio inside CA.
     *          1.0 means fully connected.
     *      C[7](float*):
     *          1D array consisting of constant parameters
     *          - theta; // firing threshold
     *          - c_decay; // decay constant d
     *          - f_recover; // recovery constant F^R
     *          - f_fatigue; // fatigue constant F^C
     *          - alpha; // learning rate
     *          - w_average; // constant representing average total synaptic strength of the pre-synaptic neuron.
     *          - w_current; // current total synaptic strength
     *      print(bool):
     *          print the CA data or not
     */
    
    // Parent
    this->n_neuron = n;
    this->activity = activity;
    this->connectivity = connectivity;
    this->inhibitory = inhibitory;

    // CA Specific
    n_threshold = ceil(n * threshold);
    n_inhibitory = static_cast<int>(ceil(n * inhibitory));
    n_excitatory = n - static_cast<int>(ceil(n * inhibitory));
    n_activation = floor(1.0f / activity);
    ignition = (0 == (rand() % static_cast<int>(floor(1.0f / activity))));

    // Constant Parameters
    theta = C[0]; // firing threshold
    c_decay = C[1]; // decay constant d
    f_recover = C[2]; // recovery constant F^R
    f_fatigue = C[3]; // fatigue constant F^C
    alpha = C[4]; // learning rate
    w_average = C[5]; // constant representing average total synaptic strength of the pre-synaptic neuron.
    w_current = C[6]; // current total synaptic strength

    // Neurons
    initFlags(n, activity, this->flags);
    initWeights(n, n, connectivity, inhibitory, this->weights);
    initEF(n, C[0]+C[3], 0.0f, this->energy);
    initEF(n, C[3], 0.0f, this->fatigue);
    pre_flags = this->flags;
    post_flags = this->flags;

    this->incomingList.push_back(this);
    this->outgoingList.push_back(this);
}

CA::~CA() {
    /* Destructor
     * Not in use for now
     */
    //std::cout << "CA destructed" << std::endl;
}

// Running
void CA::runFor(int timeStep, int available) {
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

void CA::update() {
    /* Update the CA by updating neuron related data structures
     * ! pre_synaptic and post_synaptic not in use
     */
    updatePre(this->pre_flags, this->incomingList);
    updatePost(this->post_flags, this->outgoingList);
    updateE(this -> energy, this -> weights, this -> pre_flags, this -> c_decay);
    updateF(this -> fatigue, this -> flags, this -> f_fatigue, this -> f_recover);
    // UPDATE W_AVERAGE and W_CURRENT
    updateWeights(this -> weights, this -> pre_flags, this -> post_flags, this -> alpha, this -> w_average, this -> w_current);
    updateFlags(this -> flags, this -> energy, this -> fatigue, this -> theta);
    this -> ignition = num_fire(this -> flags) > (this -> n_threshold);
    
    //std::cout << "Size\n" <<activity.size() << std::endl;
    //std::cout << activityRecord.size() << std::endl;
}

void CA::connect(CA* pre_synaptic, float pre_strength, float pre_inhibitory,
                 CA* post_synaptic, float post_strength, float post_inhibitory) {
    post_synaptic->connectIn(pre_synaptic, pre_strength, pre_inhibitory);
    pre_synaptic->connectOut(post_synaptic, post_strength, post_inhibitory);
    (post_synaptic->pre_flags).resize(7);
}

// GET
bool CA::getIgnition() {
    /* Ignition getter
     *
     * Returns:
     *      ignition(bool):
     *          ignition status of the CA
     */
    return ignition;
}

void CA::POC() {
    CA* myCA1;
    CA* myCA2;
    CA* myCA3;

    myCA1 = new CA(3);
    myCA2 = new CA(4);
    myCA3 = new CA(5);

    myCA1->runFor(1);
    myCA2->runFor(1);
    myCA3->runFor(1);

    CA::connect(myCA1, 0.2, 0.0, myCA2, 0.2, 0.0);
    CA::connect(myCA3, 0.2, 0.0, myCA1, 0.2, 0.0);

    myCA1->runFor(1);
    myCA2->runFor(1);
    myCA3->runFor(1);


    std::cout << myCA1->getActivity() << std::endl;
    std::cout << myCA2->getActivity() << std::endl;
    std::cout << myCA3->getActivity() << std::endl;
}
