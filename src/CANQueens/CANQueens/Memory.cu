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
    //initFlags(n, activity, this->flags);
    initFlags(n, activity, this->h_flags);


    //initWeights(n, n, connectivity, inhibitory, this->weights);
    initWeights(n, n, connectivity, inhibitory, this->h_weights);
}

Memory::~Memory() {
    std::cout << "Memory destructed" << std::endl;
}


// Running
void Memory::runFor_CPU(int timeStep, int available) {
    /* Run the CA for defined timestep and record the activity
     * Implemented for raster plot drawing
     *
     * Parameters:
     *      timestep(int):
     *          number of steps to stop running
     */
    for (int i = 0; i < timeStep; i++) {
        CSCToDense(this->weights, this->h_weights);
        record.push_back(setRecord(available));
        update_CPU();
    }

}

// Running
void Memory::runFor_GPU(int timeStep, int available) {
    /* Run the CA for defined timestep and record the activity
     * Implemented for raster plot drawing
     *
     * Parameters:
     *      timestep(int):
     *          number of steps to stop running
     */
    for (int i = 0; i < timeStep; i++) {
        getDeviceToHostCSC(this->h_weights, this->d_weights);
        cudaMemcpy(this->h_flags, this->d_flags, (this->n_neuron) * sizeof(bool), cudaMemcpyDeviceToHost);
        CSCToDense(this->weights, this->h_weights);

        record.push_back(setRecord(available));
        update_GPU();
    }

}

void Memory::update_CPU(float act) {
    /* Update the CA by updating neuron related data structures
     * ! pre_synaptic and post_synaptic not in use
     */
    if (act >= 0.0f && act < 1.0f) {
        setActivity(act);
    }
    //updateWeights(this->weights, this->flags, this->flags, this->alpha, this->w_average, this->w_current);
    updateWeights(this->h_weights, this->n_neuron, this->h_flags, this->n_neuron, this->h_flags, this->alpha, this->w_average, this->w_current);

    //updateFlags(this->flags, this->activity);
    updateFlags(this->n_neuron, this->h_flags, this->activity);

}


void Memory::update_GPU(float act) {
    /* Update the CA by updating neuron related data structures
     * ! pre_synaptic and post_synaptic not in use
     */
    dim3 gridSize = 1;
    dim3 blockSize = this->n_neuron; // Limitted to 1024

    if (act >= 0.0f && act < 1.0f) {
        setActivity(act);
    }
    updateWeights_kernel << <gridSize, blockSize >> > (this->n_neuron, 
        this->d_flags, 
        this->n_neuron, 
        this->d_flags, 
        this->alpha,
        this->w_average,
        this->w_current,
        this->d_weights->CO,
        this->d_weights->RI,
        this->d_weights->data);
    updateFlags_kernel << <gridSize, blockSize >> > (this->n_neuron, this->d_flags, this->activity);
}

void Memory::POC_CPU() {
    int timeStep = 10;
    std::cout << "MEMORY CPU" << std::endl;

    Memory* myMEM;
    myMEM = new Memory(10);

    myMEM->runFor_CPU(timeStep);
    myMEM->setActivity(0.1);
    myMEM->runFor_CPU(timeStep);

    std::cout << myMEM->getActivity() << std::endl;
}

void Memory::POC_GPU() {
    int timeStep = 10;
    std::cout << "MEMORY GPU" << std::endl;

    Memory* myMEM;
    myMEM = new Memory(10);

    myMEM->initMemoryGPU();

    myMEM->runFor_GPU(timeStep);
    myMEM->setActivity(0.1);

    myMEM->runFor_GPU(timeStep);
    std::cout << myMEM->getActivity() << std::endl;
}

void Memory::initMemoryGPU() {
    this->initBoolDevice(this->n_neuron, this->d_flags, this->h_flags);
    this->initCSCDevice(this->d_weights, this->h_weights);
}