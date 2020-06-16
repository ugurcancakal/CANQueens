/* Explore Class Source File
 *
 * 200517
 * author = @ugurc
 * ugurcan.cakal@gmail.com
 */

#include "Explore.cuh"

int Explore::d_n_neuron = 20;
float Explore::d_act = 0.5f;

void Explore::initFlags(int n, int n_act) {
    /* Initialize firing flags randomly
     */
    flags.resize(n);
    std::vector<bool>::iterator it;
    for (it = flags.begin(); it < flags.end(); it++) {
        *it = 0 == (rand() % n_activation);
    }
}

Explore::Explore(int n, float act, bool print) {
    std::cout << "Explore constructed" << std::endl;
    n_neuron = n;
    updateA(act);

    // Neurons
    initFlags(n_neuron, n_activation);

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

Explore::~Explore() {
    std::cout << "Explore destructed" << std::endl;
}

std::string Explore::toString() {
    return "\nExplore activity: " + std::to_string(activation) 
        +"\n(" + std::to_string(num_fire(flags)) +
        "/" + std::to_string(n_neuron) + ")\n" +
        vectorToString<bool>(flags);
}

void Explore::updateFlags() {
    std::vector<bool>::iterator it;
    for (it = flags.begin(); it < flags.end(); it++) {
        *it = 0 == (rand() % n_activation);
    }
}

void Explore::updateA(float act) {
    activation = act;
    n_activation = floor(1.0f / act);
}

void Explore::update(float act) {
    /* Update the CA by updating neuron related data structures
     * ! pre_synaptic and post_synaptic not in use
     */
    updateA(act);
    updateFlags();
}

// Running
void Explore::runFor(int timeStep) {

    for (int i = 0; i < timeStep; i++) {
        activity.push_back(flags);
        update(activation);
    }
}

std::string Explore::getActivity(int timeStep) {
    std::string temp = "\n";
    std::vector<std::vector<bool>>::iterator it_a;
    int count = 0;
    if (timeStep == 0) {
        timeStep = activity.size();
    }

    for (it_a = activity.begin(); it_a < activity.end(); it_a++) {
        temp += "timeStep " + std::to_string(count) +
            "(" + std::to_string(num_fire(*it_a)) +
            "/" + std::to_string(n_neuron) + ")" + "\n";
        temp += vectorToString<bool>((*it_a));

        temp += "\n\n";
        count++;
    }

    return temp;
}

int Explore::num_fire(std::vector<bool>& firings) {
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