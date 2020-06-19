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
    //std::cout << "Memory constructed" << std::endl;

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