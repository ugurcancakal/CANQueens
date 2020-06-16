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
int CA::nextID = 0;
int CA::d_n_neuron = 10;
float CA::d_inh = 0.2f;
float CA::d_conn = 1.0f;
float CA::d_threshold = 0.3f;
float CA::d_C[7] = { 1.0, 4.0, 0.25, 1.0, 0.2, 1.0, 1.0 };

// PROTECTED MEMBERS
// Updates
void CA::updateFlags() {
    /* Update the firing flags of the FLIF neurons inside CA
     * according tho energy levels and fatigueness
     */

    // Size check
    if (fatigue.size() != energy.size()) {
        std::cout << "Fatigue vector size is different than energy vector size!" << std::endl;
        return;
    }
    else if (flags.size() != energy.size()) {
        std::cout << "Flag vector size is different than energy and fatigue vectors size!" << std::endl;
        return;
    }

    // Iterators
    std::vector<bool>::iterator it;
    std::vector<float>::iterator it_f;
    std::vector<float>::iterator it_e;
    it_f = fatigue.begin();
    it_e = energy.begin();

    // Update
    for (it = flags.begin(); it < flags.end(); it++) {
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

void CA::updateE() {
    /* Update energy levels of the FLIF neurons inside CA
     * according to weights and firing flags
     */

    // Size check
    if (weights.size() != energy.size()) {
        std::cout << "Weight matrix row size is different than energy vector size!" << std::endl;
        return;
    }
    else if (flags.size() != energy.size()) {
        std::cout << "Flag vector size is different than energy and weight vectors size!" << std::endl;
        return;
    }
    // Iterators
    std::vector<float>::iterator it_e;
    std::vector<std::vector<float>>::iterator it_w;
    std::vector<bool>::iterator it_f;
    it_w = weights.begin();
    it_f = flags.begin();

    // Update
    for (it_e = energy.begin(); it_e < energy.end(); it_e++) {
        if (*it_f) {
            *it_e = dotP(*it_w, flags);
        }
        else {
            *it_e = ((1.0f / c_decay) * (*it_e)) + dotP(*it_w, flags);
        }
        it_w++;
        it_f++;
    }
}

void CA::updateF() {
    /* Update fatigueness of the FLIF neurons inside CA
     * according to recover rate
     */
    // Size Check
    // Size check
    if (fatigue.size() != flags.size()) {
        std::cout << "Fatigue vector size is different than flag vector size!" << std::endl;
        return;
    }

    //Iterators
    std::vector<bool>::iterator it;
    std::vector<float>::iterator it_f;
    it_f = fatigue.begin();

    //Update
    for (it = flags.begin(); it < flags.end(); it++) {
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
        it_f++;
    }
}

void CA::updateWeights() {
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
            
            for (it_weight = (*it_w).begin(); it_weight <(*it_w).end(); it_weight++) {
                //std::cout << "POST " << *it_f << std::endl;
                sign = (*it_weight) / abs(*it_weight);
                if (*it_f) {
                    delta = alpha* (1.0f - abs(*it_weight))* exp(w_average - w_current);
                }
                else {
                    delta = (-1.0f)* alpha* abs(*it_weight) * exp(w_current - w_average);
                }
                *it_weight += sign*delta;
                it_f++;
            }
        }
        it_w++;
    }
}

// Inits
void CA::initWeights(int n, float connectivity, bool print) {
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
            sign = counter < n_inhibitory ? -1.0f : 1.0f;
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

void CA::initFlags(int n) {
    /* Initialize firing flags randomly
     */
    flags.resize(n); 
    std::vector<bool>::iterator it;
    for (it = flags.begin(); it < flags.end(); it++) {
        *it = 0 == (rand() % 2);
    }
}

// Methods
float CA::dotP(std::vector<float>& weight, std::vector<bool>& flag) {
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
    
    float sum = 0.0f;
    int i = 0;
    std::vector<bool>::iterator it_f;
    std::vector<float>::iterator it_w;
    it_w = weight.begin();
    for (it_f = flags.begin(); it_f < flags.end(); it_f++) {
        if (*it_f) {
            sum += *it_w;
        }
        it_w++;
    }
    return sum;
}

int CA::num_fire(std::vector<bool>& firings) {
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

std::string CA::dateTimeStamp(const char* filename) {
    /* Create an @ugurc format timestamp
     * For example the date 09 May 1995 and time 02:48:05
     * is encoded like year-month-day-hour-minute-second
     * 950509024805
     *
     * Parameters:
     *      filename (const char*):
     *          filename to be concatenated with the timestamp
     *
     * Returns:
     *		nameSTAMP(std::string):
     *			name with a dateTimeStamp like ugurc950509024805
     */

    time_t t = time(0);   // get time now
    struct tm* now = localtime(&t);
    char buffer[80];
    strftime(buffer, 80, "%y%m%d%H%M%S", now);
    std::string dateTime = buffer;
    return filename + dateTime;
}

// PUBLIC MEMBERS
// Constructors - Destructors
CA::CA(int n, float threshold, float inh, float connectivity, float* C, bool print) {
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
    
    n_neuron = n;
    n_threshold = ceil(n * threshold);
    n_inhibitory = ceil(n * inh);
    n_excitatory = n - n_inhibitory;
    ID = ++nextID;
    ignition = (0 == (rand() % 2));

    // Constant Parameters
    theta = C[0]; // firing threshold
    c_decay = C[1]; // decay constant d
    f_recover = C[2]; // recovery constant F^R
    f_fatigue = C[3]; // fatigue constant F^C
    alpha = C[4]; // learning rate
    w_average = C[5]; // constant representing average total synaptic strength of the pre-synaptic neuron.
    w_current = C[6]; // current total synaptic strength

    // Neurons
    initWeights(n_neuron, connectivity, print);
    initFlags(n_neuron);
    energy.resize(n_neuron);
    fatigue.resize(n_neuron);
    pre_synaptic.resize(n_neuron);
    post_synaptic.resize(n_neuron);

    if (print) {
        std::cout << "\nCA constructed with ID: " << ID << ": " << std::endl
            << n_excitatory << " excitatory, " << std::endl
            << n_inhibitory << " inhibitory neurons; " << std::endl
            << n_threshold << " of neurons active threshold\n" << std::endl;

        std::cout << "Constant Parameters " << std::endl
            << "firing threshold : " << theta << std::endl
            << "decay constant : " << c_decay << std::endl
            << "recovery constant F^R : " << f_recover << std::endl
            << "fatigue constant F^C : " << f_fatigue << std::endl
            << "learning rate : " << alpha << std::endl
            << "average total synaptic strength : " << w_average << std::endl
            << "current total synaptic strength : " << w_current << std::endl;
    }
}

CA::~CA() {
    /* Destructor
     * Not in use for now
     */
    //std::cout << "CA destructed" << std::endl;
}

// Running
void CA::runFor(int timeStep) {
    /* Run the CA for defined timestep and record the activity
     * Implemented for raster plot drawing
     *
     * Parameters:
     *      timestep(int):
     *          number of steps to stop running
     */
    int prev = activityRecord.size();
    activityRecord.resize(activityRecord.size() + timeStep);

    record temp;
    /*temp.flags.resize(flags.size());
    temp.weights.resize(weights.size());
    std::vector<std::vector<float>>::iterator it_w;
    for (it_w = temp.weights.begin(); it_w < temp.weights.end(); it_w++) {
        (*it_w).resize(weights[0].size());
    }
    temp.energy.resize(energy.size());
    temp.fatigue.resize(fatigue.size());*/
    //

    std::vector<std::vector<bool>>::iterator it;
    std::vector<bool>::iterator it_f;
    std::vector<record>::iterator it_r;

    //

    //for (int i = 0; i < timeStep; i++) {
    //    temp.flags = flags;
    //    temp.weights = weights;
    //    temp.energy = energy;
    //    temp.fatigue = fatigue;
    //    activity.push_back(temp);
    //    update();
    //}

    for (it = activityRecord.begin()+prev; it < activityRecord.end(); it++) {
        for (it_f = flags.begin(); it_f < flags.end(); it_f++) {
            (*it).push_back(*it_f);
        }
        temp.flags = flags;
        temp.weights = weights;
        temp.energy = energy;
        temp.fatigue = fatigue;
        activity.push_back(temp);
        update();
    }

    
}

void CA::update() {
    /* Update the CA by updating neuron related data structures
     * ! pre_synaptic and post_synaptic not in use
     */
    updateE();
    updateF();
    updateWeights();
    updateFlags();
    ignition = num_fire(flags) > n_threshold;
    //std::cout << "Size\n" <<activity.size() << std::endl;
    //std::cout << activityRecord.size() << std::endl;
}

// Printing
std::string CA::toString(int timeStep) {
    /* Construct the string representing whole activity record 
     * in given timestep
     * 
     * Parameters:
     *      timeStep(int):
     *          number of steps to stop
     *
     * Returns:
     *      record(std::string):
     *          activity record
     */

    std::string temp = "\n";
    std::vector<std::vector<bool>>::iterator it;
    std::vector<bool>::iterator it_act;

    for (it = activityRecord.begin(); it < activityRecord.end(); it++) {
        for (it_act = (*it).begin(); it_act< (*it).end(); it_act++) {
            temp += to_string(*it_act) + "\t";
        }
        temp += " \n";
    }
    return temp;
}

std::string CA::getRaster(int timeStep) {
    /* Construct the string representing whole raster plot
     * for given time interval. IDs with * represents
     * inhibitory neurons.
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
     * Parameters:
     *      timeStep(int):
     *          number of steps to stop
     *
     * Returns:
     *      raster(std::string):
     *          raster plot
     */

    // Size Check
    if (timeStep > activityRecord.size()) {
        std::cout << "Not enough data to show " << timeStep << " steps!" << std::endl;
        timeStep = activityRecord.size();
    }
    if (timeStep < activityRecord.size()) {
        std::cout << "Last " << timeStep << " steps will be printed!" << std::endl;
    }

    std::string temp = "\n";
    std::vector<std::vector<bool>>::iterator it;
    std::vector<bool>::iterator it_act;

    // Header
    temp += "  \t";
    temp += " \n";
    temp += "N_ID \t||";
    temp += std::string(3 * (timeStep + 1), ' ');
    temp += "SPIKE ACTIVITY\n";
    temp += std::string(8 * (timeStep + 1) + 4, '-') + "\n";
    
    // Body
    for (int i = 0; i < n_neuron; i++) {
        temp += std::to_string(i);
        if (i < n_inhibitory) {
            temp += "*\t||\t";
        }
        else {
            temp += "\t||\t";
        }
        for (int j = (activityRecord.size() -timeStep); j < activityRecord.size(); j++) {
            if (activityRecord[j][i]) {
                temp += "|\t";
            }
            else {
                temp += " \t";
            }
        }
        temp += "\n";
    }

    temp += std::string(8 * (timeStep + 1) + 4, '-') + "\n";
    temp += "TIME \t||\t";
    for (int i = 0; i < timeStep; i++) {
        temp += std::to_string(i) + "\t";
    }
    temp += "\n";
    temp += std::string(8 * (timeStep + 1) + 4, '-') + "\n";
    temp += "FIRE \t||\t";
    for (int i = (activityRecord.size() - timeStep); i < activityRecord.size(); i++) {
        temp += std::to_string(num_fire(activityRecord[i])) + "\t";
    }
    temp += "\n";
    temp += std::string(8 * (timeStep + 1) + 4, '-') + "\n";
    temp += "IGNIT \t||\t";
    for (int i = (activityRecord.size() - timeStep); i < activityRecord.size(); i++) {
        temp += std::to_string(num_fire(activityRecord[i]) >= n_threshold) + "\t";
    }
    return temp;
}

void CA::saveRaster(char* filename, int timeStep) {
    /* Save the raster plot constructed by getRaster()
     * to a .txt file, into ./test/raster/ folder.
     * Parameters:
     *      filename(char*):
     *          filename to be stamped then used
     *      timeStep(int):
     *          number of steps to stop
     */
    std::string name = "./test/raster/" + dateTimeStamp(filename) + ".txt";
    std::ofstream raster(name, std::ofstream::out);

    if (raster.is_open()) {
        raster << getRaster(timeStep) << std::endl;
    }
}

std::string CA::getActivity(int timeStep) {
    std::string temp = "\n";
    std::vector<record>::iterator it_r;
    int count = 0;
    if (timeStep == 0) {
        timeStep = activity.size();
    }

    for (it_r = activity.begin(); it_r < activity.end(); it_r++) {
        temp += "timeStep " + std::to_string(count) + "\n";
 
        temp += "\nFlags \n";
        temp+= vectorToString<bool>((*it_r).flags);

        temp += "\n\nEnergy Levels \n";
        temp += vectorToString<float>((*it_r).energy);

        temp += "\n\nFatigue Levels \n";
        temp += vectorToString<float>((*it_r).fatigue);

        temp += "\n\nWeights \n";
        std::vector<std::vector<float>>::iterator it_w;
        for (it_w = (*it_r).weights.begin(); it_w < (*it_r).weights.end(); it_w++) {
            temp += "|" + vectorToString<float>(*it_w) + "|\n";
        }

        temp += "\n\n";
        count++;
    }

    return temp;
}

void CA::saveActivity(char* filename, int timeStep) {

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

int CA::getID() {
    /* ID getter
     *
     * Returns:
     *      ID(int):
     *          ID of the CA starting from 0
     */
    return ID;
}