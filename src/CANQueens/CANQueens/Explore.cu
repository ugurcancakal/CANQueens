/* Explore Class Source File
 *
 * 200517
 * author = @ugurc
 * ugurcan.cakal@gmail.com
 */

#include "Explore.cuh"

int Explore::d_n_neuron = 20;
float Explore::d_activity = 0.5f;
int Explore::d_available = 0b1000;

Explore::Explore(int n, float activity) {
    std::cout << "Explore constructed" << std::endl;
    this->n_neuron = n;
    this->activity = activity;

    // Neurons
    initFlags(n, activity, this->flags);
}

Explore::~Explore() {
    std::cout << "Explore destructed" << std::endl;
}

void Explore::updateFlags(std::vector<bool>& flag_vec,
                          const float& activity) {

    std::vector<bool>::iterator it;
    for (it = flag_vec.begin(); it < flag_vec.end(); it++) {
        *it = 0 == (rand() % static_cast<int>(floor(1.0f / activity)));
    }
}


// Running
void Explore::runFor(int timeStep, int available) {

    for (int i = 0; i < timeStep; i++) {
        record.push_back(setRecord(available));
        update();
    }
}

void Explore::update(float act) {
    /* Update the CA by updating neuron related data structures
     * ! pre_synaptic and post_synaptic not in use
     */
    if (act >= 0.0f && act < 1.0f) {
        setActivity(act);
    }
    updateFlags(this -> flags, this -> activity);
}

// Set
void Explore::setActivity(float act) {
    this -> activity = act;
}

void Explore::POC() {
    int timeStep = 10;
    Explore* myEXP;
    myEXP = new Explore(3);

    myEXP -> runFor(timeStep);
    std::cout << myEXP -> getActivity() << std::endl;

    myEXP -> setActivity(0.1);

    myEXP -> runFor(timeStep);
    std::cout << myEXP -> getActivity() << std::endl;
}
