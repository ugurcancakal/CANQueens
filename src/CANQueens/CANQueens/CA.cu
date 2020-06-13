/* Cell Assembly Class Source File
 *
 * 200516
 * author = @ugurc
 * ugurcan.cakal@gmail.com
 */

#include "CA.cuh"

int CA::nextID = 0;
int CA::d_n_neuron = 10;
float CA::d_inh = 0.3f;
float CA::d_threshold = 0.3f;
float CA::d_C[7] = { 1.0, 1.0, 1.0, 1.0, 0.2, 1.0, 1.0 };

//CA::CA() {
//    ignition = (0 == (rand() % 2));
//    ID = ++nextID;
//    //std::cout << "CA constructed with ID: " << ID << std::endl;
//}
//
//CA::CA(int n) {
//    n_neuron = n;
//    ignition = (0 == (rand() % 2));
//    ID = ++nextID;
//    //std::cout << "CA constructed with ID: " << ID << std::endl;
//}

CA::CA(int n, float threshold, float inh, float* C, bool print) {
    n_neuron = n;
    n_threshold = floor(n * threshold);
    n_inhibitory = floor(n * inh);
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
    initWeights(print);
    initFlags();
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
    //std::cout << "CA destructed" << std::endl;
}

void CA::initWeights(bool print) {
    // all connected for now
    float sign = -1.0f;
    weights.resize(n_neuron);
    for (int i = 0; i < n_neuron; i++) {
        weights[i].resize(n_neuron);
    }

    for (int i = 0; i < n_neuron; i++) {
        for (int j = 0; j < n_neuron; j++) {
            sign = j < n_inhibitory ? -1.0f : 1.0f;
            weights[i][j] = sign*static_cast <float> (rand()) / static_cast <float> (RAND_MAX);
            if (print) {
                std::cout << weights[i][j] << " " << std::endl;
            }
        }
    }
}

void CA::initFlags() {
    flags.resize(n_neuron);
    for (int i = 0; i < n_neuron; i++) { 
        flags[i] = 0 == (rand() % 2);
    }
}

std::string CA::toString(int timeStep) {
    std::string temp = "\n";
    for (int i = 0; i < timeStep; i++)
    {
        for (int j = 0; j < n_neuron; j++)
        {
            temp+= to_string(activityRecord[i][j]) + "\t";
        }
        temp += " \n";
    }
    return temp;
}

bool CA::getIgnition() {
	return ignition;
}

int CA::getID() {
    return ID;
}

void CA::runFor(int timeStep) {
    activityRecord.resize(timeStep);
    for (int i = 0; i < timeStep; i++) {
        for (int j = 0; j < n_neuron; j++) {
            activityRecord[i].push_back(flags[j]);
        }
        update();
    }
}

void CA::update(){
    updateWeights();
    updateEF();
    updateFlags();
    ignition = firingStatus(flags);
}

void CA::updateFlags() {
    //flags[i] = 0 == (rand() % 2);
    for (int i = 0; i < n_neuron; i++) {
        if (energy[i] - fatigue[i] > theta) {
            flags[i] = true;
        }
        else {
            flags[i] = false;
        }
    }
}

void CA::updateEF() {
    for (int i = 0; i < n_neuron; i++) {
        if (flags[i]) {
            energy[i] = dotP(weights[i], flags);
            fatigue[i] += f_fatigue;
        }
        else {
            energy[i] = (1.0f/c_decay)*energy[i] + dotP(weights[i], flags);
            if (fatigue[i] - f_recover > 0) {
                fatigue[i] = fatigue[i] - f_recover;
            }
            else {
                fatigue[i] = 0.0f;
            }
        }
        //std::cout << "\nNeuronID : " << i << std::endl
        //        << "Energy Level : " << energy[i] << std::endl
        //        << "Fatigue Level : " << fatigue[i] << std::endl;
    }
}

float CA::dotP(std::vector<float>& weight, std::vector<bool>& flag) {
    float sum = 0.0f;
    for (int i = 0; i < n_neuron; i++) {
        if (flag[i]) {
            sum += weight[i];
        }
        else {
            continue;
        }
    }
    return sum;
}

void CA::updateWeights() {
    // W-current W-average updates are necessary
    float delta = 0.0f;

    for (int i = 0; i < n_neuron; i++) {
        if (flags[i] == 1) {
            for (int j = 0; j < n_neuron; j++) {
                if (flags[i] == true) {
                    alpha* (1.0f - weights[i][j]) * exp(w_average - w_current);
                }
                else {
                    (-1.0f)*alpha* weights[i][j] * exp(w_current - w_average);
                }
            }
        }
        else {
            continue;
        }
    }
}

std::string CA::getRaster(int timeStep) {

    std::string temp = "\n";
    temp += "  \t";
    temp += " \n";
    temp += "N_ID \t||";
    temp += std::string(3 * (timeStep + 1), ' ');
    temp += "SPIKE ACTIVITY\n";
    temp += std::string(8 * (timeStep + 1) + 4, '-') + "\n";
    for (int i = 0; i < n_neuron; i++) {
        temp += std::to_string(i);
        if (i < n_inhibitory) {
            temp += "*\t||\t";
        }
        else {
            temp += "\t||\t";
        }
        for (int j = 0; j < timeStep; j++) {
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
    for (int i = 0; i < timeStep; i++) {
        temp += std::to_string(firingStatus(activityRecord[i])) + "\t";
    }
    temp += "\n";
    temp += std::string(8 * (timeStep + 1) + 4, '-') + "\n";
    temp += "IGNIT \t||\t";
    for (int i = 0; i < timeStep; i++) {
        temp += std::to_string(firingStatus(activityRecord[i]) >= n_threshold) + "\t";
    }
    return temp;
}

void CA::saveRaster(char* filename, int timeStep) {
    std::string name = "./test/raster/" + dateTimeStamp(filename) + ".txt";
    std::ofstream raster(name, std::ofstream::out);

    if (raster.is_open()) {
        raster << getRaster(timeStep) << std::endl;
    }
}

std::string CA::dateTimeStamp(const char* filename) {
    /* Create an @ugurc format timestamp
     * For example The date 09 May 1995 and time 02:48:05
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

//int CA::firingStatus(int timeStep) {
//    int fire = 0;
//    for (int i = 0; i < n_neuron; i++) {
//        if (activityRecord[timeStep][i]) {
//            fire++;
//        }
//    }
//    return fire;
//}

int CA::firingStatus(std::vector<bool>& firings) {
    int fire = 0;
    for (int i = 0; i < n_neuron; i++) {
        if (firings[i]) {
            fire++;
        }
    }
    return fire;
}