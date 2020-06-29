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

__global__ void updatePhi_kernel(int n, bool* d_flags, float* d_energy, float* d_fatigue, float theta) {
    unsigned int index = threadIdx.x + blockDim.x * blockIdx.x;
    unsigned int stride = blockDim.x * gridDim.x;
    while (index < n) {
        d_flags[index] = (d_energy[index] - d_fatigue[index]) > theta ? true : false;
        index += stride;
    }
}

__global__ void dotP_kernel(float* product, int start, int stop, int* RI, float* data, bool* d_flags) {
    /*float sum = 0.0f;
    for (int i = start; i < stop; i++) {
        if (flags[RI[i]]) {
            sum += data[i];
        }
    }*/

    unsigned int index = threadIdx.x + blockDim.x * blockIdx.x;
    unsigned int stride = blockDim.x * gridDim.x;

    extern __shared__ float cache[]; // to use the thread-block shared memory

    float temp = 0.0f;
    float sum = 0.0f;

    while (index < stop - start) {
        if (d_flags[RI[index+start]]) {
            sum += data[index + start];
        }
        index += stride;
    }
    cache[threadIdx.x] = temp;
    __syncthreads();

    //Reduction
    unsigned int i = blockDim.x / 2;
    while (i != 0) {
        if (threadIdx.x < i) {
            cache[threadIdx.x] += cache[threadIdx.x + i];
        }
        __syncthreads();
        i /= 2;
    }

    if (threadIdx.x == 0) {
        atomicAdd(product, cache[0]);
    }
}

__global__ void updateE_kernel(int n, float* d_energy, int c_decay, float* product, int* CO, int* RI, float*data, bool* d_flags) {
    unsigned int index = threadIdx.x + blockDim.x * blockIdx.x;
    unsigned int stride = blockDim.x * gridDim.x;
    while (index < n) {    
        dotP_kernel << <1, CO[index + 1]- CO[index], n >> > (product, CO[index], CO[index + 1], RI, data, d_flags);
        
        if (d_flags[index]) {  
            d_energy[index] = *product;
        }
        else {
            d_energy[index] = ((1.0f / (1.0f*c_decay)) * (d_energy[index])) + *product;
        }
        index += stride;
    }
}

__global__ void updateF_kernel(int n, float* d_fatigue, bool* const d_flags, float f_fatigue, float f_recover){
    unsigned int index = threadIdx.x + blockDim.x * blockIdx.x;
    unsigned int stride = blockDim.x * gridDim.x;
    while (index < n) {
        if (d_flags[index]) {
            d_fatigue[index] += f_fatigue;
        }
        else {
            if (d_fatigue[index] - f_recover > 0.0f) {
                d_fatigue[index] = d_fatigue[index] - f_recover;
            }
            else {
                d_fatigue[index] = 0.0f;
            }
        }
        index += stride;
    }
}


// Default values for CA initiation
int CA::d_n_neuron = 10;
float CA::d_activity = 0.5f;
float CA::d_connectivity = 0.5f;
float CA::d_inhibitory = 0.2f;
float CA::d_threshold = 0.3f;
float CA::d_C[7] = { 5.0, 0.5, 0.5, 1.0, 0.0, 1.0, 1.0 };

/*C[7](float*) :
 *
 *C[0]:theta; // firing threshold
 *C[1]:c_decay; // decay constant d
 *C[2]:f_recover; // recovery constant F^R
 *C[3]:f_fatigue; // fatigue constant F^C
 *C[4]:alpha; // learning rate
 *C[5]:w_average; // constant representing average total synaptic strength of the pre-synaptic neuron.
 *C[6]:w_current; // current total synaptic strength
 */

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

void CA::updateFlags(int n, bool*& h_flags, float*& const h_energy, float*& const h_fatigue, const float& theta) {
    for (int i = 0; i < n; i++) {
        if (h_energy[i] - h_fatigue[i] > theta) {
            h_flags[i] = true;
        }
        else {
            h_flags[i] = false;
        }
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

void CA::updateE(int n, float*& h_energy, CSC*& const h_weights, bool*& const h_preFlags, const int& c_decay) {
   for (int i = 0; i < n; i++) {
        if (h_preFlags[i]) {
            h_energy[i] = dotP(h_weights->CO[i], 
                h_weights->CO[i + 1],
                h_weights->RI, 
                h_weights->data, 
                h_preFlags);
        }
        else {
            h_energy[i] = ((1.0f / c_decay) * (h_energy[i])) + dotP(h_weights->CO[i],
                h_weights->CO[i + 1],
                h_weights->RI,
                h_weights->data,
                h_preFlags);
        }
        
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

void CA::updateF(int n, float*& h_fatigue, bool*& const h_flags, float& const f_fatigue, float& const f_recover) {
    for (int i = 0; i < n; i++) {
        if (h_flags[i]) {
            h_fatigue[i] += f_fatigue;
        }
        else {
            if (h_fatigue[i] - f_recover > 0.0f) {
                h_fatigue[i] = h_fatigue[i] - f_recover;
            }
            else {
                h_fatigue[i] = 0.0f;
            }
        }
    }
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

float CA::dotP(const int& start, const int& stop, int*& const RI, float*& const data, bool*& const flags) {
    float sum = 0.0f;
    for (int i = start; i < stop; i++) {
        if (flags[RI[i]]) {
            sum += data[i];
        }
    }
    return sum;
}

void CA::initCADevice() {
    this->initBoolDevice(this->n_neuron, this->d_flags, this->h_flags);
    this->initBoolDevice(this->preSize, this->d_preFlags, this->h_preFlags);
    this->initBoolDevice(this->postSize, this->d_postFlags, this->h_postFlags);
    this->initFloatDevice(this->n_neuron, this->d_energy, this->h_energy);
    this->initFloatDevice(this->n_neuron, this->d_fatigue, this->h_fatigue);
    this->initCSCDevice(this->d_weights, this->h_weights);
}

void CA::freeCADevice() {
    this->freeBoolDevice(this->d_flags);
    this->freeBoolDevice(this->d_preFlags);
    this->freeBoolDevice(this->d_postFlags);
    this->freeFloatDevice(this->d_energy);
    this->freeFloatDevice(this->d_fatigue);
    this->freeCSCDevice(this->d_weights);
}

cudaError_t CA::getDeviceToHostEF(const int& n, float*& h_EF, float*& const d_EF) {
    cudaError_t cudaStatus;
    // Copy output vector from GPU buffer to host memory.
    cudaStatus = cudaMemcpy(h_EF, d_EF, n * sizeof(float), cudaMemcpyDeviceToHost);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "EF cudaMemcpy failed!");
        return cudaStatus;
    }
    return cudaStatus;
}

cudaError_t CA::getDeviceToHostFlags(const int& n, bool*& h_flags, bool*& const d_flags) {
    cudaError_t cudaStatus;
    // Copy output vector from GPU buffer to host memory.
    cudaStatus = cudaMemcpy(h_flags, d_flags, n * sizeof(bool), cudaMemcpyDeviceToHost);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "Flags cudaMemcpy failed!");
        return cudaStatus;
    }
    return cudaStatus;
}

cudaError_t CA::errorCheckCUDA(bool synchronize) {
    // Check for any errors launching the kernel
    cudaError_t cudaStatus;
    cudaStatus = cudaGetLastError();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "kernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
        return cudaStatus;
    }

    // cudaDeviceSynchronize waits for the kernel to finish, and returns
    // any errors encountered during the launch.
    if (synchronize) {
        cudaStatus = cudaDeviceSynchronize();
        if (cudaStatus != cudaSuccess) {
            fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching kernel!\n", cudaStatus);
            return cudaStatus;
        }
    }
    return cudaStatus;
    
}

cudaError_t CA::updatePreGPU() {
    cudaError_t cudaStatus;
    updatePre(this->h_preFlags, this->preSize, this->incomingList);

    //std::cout << "UP PRE SIZE :" << this->preSize << std::endl;
    //std::cout << vectorToString<bool>(std::vector<bool>(this->h_preFlags, this->h_preFlags+this->preSize)) << std::endl;
    cudaStatus = freeBoolDevice(this->d_preFlags);
    cudaStatus = initBoolDevice(this->preSize, this->d_preFlags, this->h_preFlags);
    return cudaStatus;
    
}

cudaError_t CA::updatePostGPU() {
    cudaError_t cudaStatus;
    updatePost(this->h_postFlags, this->postSize, this->outgoingList);
    /*std::cout << "UP POST SIZE :" << this->postSize << std::endl;
    std::cout << vectorToString<bool>(std::vector<bool>(this->h_postFlags, this->h_postFlags + this->postSize)) << std::endl;*/
    
    cudaStatus = freeBoolDevice(this->d_postFlags);
    cudaStatus = initBoolDevice(this->postSize, this->d_postFlags, this->h_postFlags);
    return cudaStatus;
}

cudaError_t CA::updateE_GPU(dim3 gridSize, dim3 blockSize, bool synchronize, bool memCopy) {
    cudaError_t cudaStatus;

    updateE_kernel << <gridSize, blockSize >> > (this->n_neuron,
        this->d_energy,
        this->c_decay,
        this->product,
        this->d_weights->CO,
        this->d_weights->RI,
        this->d_weights->data,
        this->d_preFlags);
    cudaStatus = errorCheckCUDA(synchronize);

    if (memCopy) {
        cudaStatus = getDeviceToHostEF(this->n_neuron, this->h_energy, this->d_energy);
    }

    return cudaStatus;
}

cudaError_t CA::updateF_GPU(dim3 gridSize, dim3 blockSize, bool synchronize, bool memCopy) {
    cudaError_t cudaStatus;

    updateF_kernel << <gridSize, blockSize >> > (this->n_neuron,
        this->d_fatigue,
        this->d_flags,
        this->f_fatigue,
        this->f_recover);

    cudaStatus = errorCheckCUDA(synchronize);

    if (memCopy) {
        cudaStatus = getDeviceToHostEF(this->n_neuron, this->h_fatigue, this->d_fatigue);
    }

    return cudaStatus;
}

cudaError_t CA::updateWeights_GPU(dim3 gridSize, dim3 blockSize, bool synchronize, bool memCopy) {
    cudaError_t cudaStatus;
    updateWeights_kernel << <gridSize, blockSize >> > (this->preSize,
        this->d_preFlags,
        this->postSize,
        this->d_postFlags,
        this->alpha,
        this->w_average,
        this->w_current,
        this->d_weights->CO,
        this->d_weights->RI,
        this->d_weights->data);

    cudaStatus = errorCheckCUDA(synchronize);

    if (memCopy) {
        cudaStatus = getDeviceToHostCSC(this->h_weights, this->d_weights);
    }

    return cudaStatus;
}

cudaError_t CA::updatePhi_GPU(dim3 gridSize, dim3 blockSize, bool synchronize, bool memCopy) {
    
    cudaError_t cudaStatus;
    updatePhi_kernel << <gridSize, blockSize >> > (this->n_neuron,
        this->d_flags,
        this->d_energy,
        this->d_fatigue,
        this->theta);

    cudaStatus = errorCheckCUDA(synchronize);

    if (memCopy) {
        cudaStatus = getDeviceToHostFlags(this->n_neuron, this->h_flags, this->d_flags);
    }

    return cudaStatus;
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
    n_threshold = static_cast<int>(ceilf(n * threshold));
    n_inhibitory = static_cast<int>(ceilf(n * inhibitory));
    n_excitatory = n - static_cast<int>(ceilf(n * inhibitory));
    n_activation = static_cast<int>(floorf(1.0f / activity));
    ignition = (0 == (rand() % static_cast<int>(floorf(1.0f / activity))));

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

    // Neurons -- Host
    preSize = n;
    postSize = n;
    initFlags(n, activity, this->h_flags);
    initFlags(n, activity, this->h_preFlags);
    initFlags(n, activity, this->h_postFlags);
    initWeights(n, n, connectivity, inhibitory, this->h_weights);
    initEF(n, C[0] + C[3], 0.0f, this -> h_energy);
    initEF(n, C[3], 0.0f, this->h_fatigue);

    cudaMalloc((void**)&product, sizeof(float));

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
void CA::runFor_CPU(int timeStep, int available) {
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



void CA::runFor_GPU(int timeStep, int available) {
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
        update_GPU();
    }

}

void CA::update_CPU() {
    /* Update the CA by updating neuron related data structures
     * ! pre_synaptic and post_synaptic not in use
     */
    //updatePre(this->pre_flags, this->incomingList);
    //updatePost(this->post_flags, this->outgoingList);
    //updateE(this->energy, this->weights, this->pre_flags, this->c_decay);
    //updateF(this->fatigue, this->flags, this->f_fatigue, this->f_recover);
    //// UPDATE W_AVERAGE and W_CURRENT
    //updateWeights(this->weights, this->pre_flags, this->post_flags, this->alpha, this->w_average, this->w_current);
    //updateFlags(this->flags, this->energy, this->fatigue, this->theta);
    //this->ignition = num_fire(this->flags) > (this->n_threshold);

    updatePre(this->h_preFlags, this->preSize, this->incomingList);
    updatePost(this->h_postFlags, this->postSize, this->outgoingList);
    updateE(this->n_neuron, this->h_energy, this->h_weights, this->h_preFlags, this->c_decay);   
    updateF(this->n_neuron, this->h_fatigue, this->h_flags, this->f_fatigue, this->f_recover);
    //std::cout << "PRE SIZE: " << this->preSize << std::endl;
    //std::cout << "POST SIZE: " << this->postSize << std::endl;
    updateWeights(this->h_weights, this->preSize, this->h_preFlags, this->postSize, this->h_postFlags, this->alpha, this->w_average, this->w_current);
    updateFlags(this->n_neuron, this->h_flags, this->h_energy, this->h_fatigue, this->theta);
    this->ignition = num_fire(this->n_neuron, this->h_flags) > (this->n_threshold);


    //std::cout << "Size\n" <<activity.size() << std::endl;
    //std::cout << activityRecord.size() << std::endl;
}

void CA::update_GPU() {
    /* Update the CA by updating neuron related data structures
     * ! pre_synaptic and post_synaptic not in use
     */
    //updatePre(this->pre_flags, this->incomingList);
    //updatePost(this->post_flags, this->outgoingList);
    //updateE(this->energy, this->weights, this->pre_flags, this->c_decay);
    //updateF(this->fatigue, this->flags, this->f_fatigue, this->f_recover);
    //// UPDATE W_AVERAGE and W_CURRENT
    //updateWeights(this->weights, this->pre_flags, this->post_flags, this->alpha, this->w_average, this->w_current);
    //updateFlags(this->flags, this->energy, this->fatigue, this->theta);
    //this->ignition = num_fire(this->flags) > (this->n_threshold);

    dim3 gridSize = 1;
    dim3 blockSize = this->n_neuron; // Limitted to 1024.
    cudaError_t cudaStatus;
    
    cudaStatus = updatePreGPU();
    cudaStatus = updatePostGPU();
    cudaStatus = updateE_GPU(this->postSize, this->preSize);
    cudaStatus = updateF_GPU(gridSize, blockSize);
    cudaStatus = updateWeights_GPU(this->postSize, this->preSize);
    cudaStatus = updatePhi_GPU(gridSize, blockSize);
    this->ignition = num_fire(this->n_neuron, this->h_flags) > (this->n_threshold);
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

void CA::POC_CPU() {
    CA* myCA1;
    CA* myCA2;
    CA* myCA3;

    myCA1 = new CA(10);
    myCA2 = new CA(4);
    myCA3 = new CA(5);

    myCA1->runFor_CPU(10);
    myCA2->runFor_CPU(1);
    myCA3->runFor_CPU(1);

    CA::connect(myCA1, 0.2, 0.0, myCA2, 0.2, 0.0);
    CA::connect(myCA3, 0.2, 0.0, myCA1, 0.2, 0.0);

    myCA1->runFor_CPU(1);
    myCA2->runFor_CPU(1);
    myCA3->runFor_CPU(1);

    std::cout << myCA1->getActivity() << std::endl;
    std::cout << myCA2->getActivity() << std::endl;
    std::cout << myCA3->getActivity() << std::endl;

    //myCA1->saveCSV("");
}

void CA::POC_GPU() {
    // CONNECT DURUMUNDA GPU YA GONDERMEK GEREK POST PRE VE WEIGHTS
    CA* myCA1;
    CA* myCA2;
    CA* myCA3;

    myCA1 = new CA(5);
    myCA1->initCADevice();
    
    myCA2 = new CA(4);
    myCA2->initCADevice();

    myCA3 = new CA(5);
    myCA3->initCADevice();

    myCA1->runFor_GPU(10);
    myCA2->runFor_GPU(1);
    myCA3->runFor_GPU(1);

    CA::connect_GPU(myCA1, 0.2f, 0.0, myCA2, 0.2f, 0.0);
    CA::connect_GPU(myCA2, 0.2f, 0.0, myCA1, 0.2f, 0.0);
       
    myCA1->runFor_GPU(1);
    myCA2->runFor_GPU(1);
    myCA3->runFor_GPU(1);

    std::cout << myCA1->getActivity() << std::endl;
    //std::cout << myCA2->getActivity() << std::endl;
    //std::cout << myCA3->getActivity() << std::endl;

    //myCA1->saveCSV("");
}
