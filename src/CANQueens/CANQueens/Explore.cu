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
    //std::cout << "Explore constructed" << std::endl;
    this->n_neuron = n;
    this->activity = activity;

    // Neurons
    //initFlags(n, activity, this->flags);
    initFlags(n, activity, this->h_flags);
}

Explore::~Explore() {
    std::cout << "Explore destructed" << std::endl;
}

// Running
void Explore::runFor_CPU(int timeStep, int available) {

    for (int i = 0; i < timeStep; i++) {
        record.push_back(setRecord(available));
        update_CPU();
    }
}

void Explore::runFor_GPU(int timeStep, int available) {

    for (int i = 0; i < timeStep; i++) {
        cudaMemcpy(this->h_flags, this->d_flags, (this->n_neuron) * sizeof(bool), cudaMemcpyDeviceToHost);
        record.push_back(setRecord(available));
        update_GPU();
    }
}

void Explore::update_CPU(float act) {
    /* Update the CA by updating neuron related data structures
     * ! pre_synaptic and post_synaptic not in use
     */
    if (act >= 0.0f && act < 1.0f) {
        setActivity(act);
    }
    //updateFlags(this -> flags, this -> activity);
    updateFlags(this-> n_neuron, this->h_flags, this->activity);
}

void Explore::update_GPU(float act) {
    /* Update the CA by updating neuron related data structures
     * ! pre_synaptic and post_synaptic not in use
     */

    dim3 gridSize = 1;
    dim3 blockSize = this->n_neuron; // Limitted to 1024.
    
    if (act >= 0.0f && act < 1.0f) {
        setActivity(act);
    }
    updateFlags_kernel <<<gridSize, blockSize>>> (this->n_neuron, this->d_flags, this->activity);
    
}

void Explore::POC_CPU() {
    int timeStep = 10;
    std::cout << "EXPLORE CPU" << std::endl;
    Explore* myEXP;
    myEXP = new Explore(20);
    myEXP -> runFor_CPU(timeStep);
    myEXP->setActivity(0.1);
    myEXP -> runFor_CPU(timeStep);
    std::cout << myEXP -> getActivity() << std::endl;
}

void Explore::POC_GPU() {
    int timeStep = 10;
    std::cout << "EXPLORE GPU" << std::endl;

    Explore* myEXP;
    myEXP = new Explore(20);
    myEXP->initExploreGPU();

    myEXP->runFor_GPU(timeStep);
    myEXP->setActivity(0.1);

    myEXP->runFor_GPU(timeStep);
    std::cout << myEXP->getActivity() << std::endl;
}

void Explore::initExploreGPU() {
    this->initBoolDevice(this->n_neuron, this->d_flags, this->h_flags);
}
