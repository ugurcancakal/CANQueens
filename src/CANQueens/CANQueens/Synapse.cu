/* Synapse Class Source File
 *
 * 200619
 * author = @ugurc
 * ugurcan.cakal@gmail.com
 */

#include "Synapse.cuh"

__global__ void updateWeights_kernel(const int pre_size,
    bool* const d_preFlags,
    const int post_size,
    bool* const d_postFlags,
    const float alpha,
    const float w_average,
    const float w_current,
    int* const CO,
    int* const RI,
    float* data) {

    // NO NEW CONNECTIONS ALLOWED FOR NOW

    unsigned int index = threadIdx.x + blockDim.x * blockIdx.x;
    unsigned int stride = blockDim.x * gridDim.x;
    unsigned int pre_index;
    unsigned int post_index;
    float delta;
    float tempData;
    float sign;
    int c;

    while (index < pre_size * post_size) {
        sign = 1.0f;
        // Indexing
        //pre_index = index / post_size;
        //post_index = index - (pre_index * post_size);   

        post_index = index / pre_size;
        pre_index = index - (post_index * pre_size);

        if (d_preFlags[pre_index]) {
            delta = 0.0f;
            // GET CSC DATA 
            tempData = 0.0f;
            if (CO[post_index] == CO[post_index + 1]) {
                tempData = 0.0f;
            }
            else {
                for (c = CO[post_index]; c < CO[post_index + 1]; c++) {
                    if (RI[c] == pre_index) {
                        tempData = data[c];
                        break;
                    }
                }
            }
            if (tempData != 0.0f) {
                sign = tempData / abs(tempData);
                if (d_postFlags[post_index]) {
                    delta = alpha * (1.0f - abs(tempData)) * expf(w_average - w_current);
                }
                else {
                    delta = (-1.0f) * alpha * abs(tempData) * expf(w_current - w_average);
                }
                data[c] += sign * delta;
                //data[index] = index;
            }
        }
        index += stride;
    }
}

Synapse::Synapse() {
    //std::cout << "Synapse constructed" << std::endl;
    connectivity = 0.0f;
    inhibitory = 0.0f;
    alpha = 0.0f; // learning rate
    w_average = 0.0f; // constant representing average total synaptic strength of the pre-synaptic neuron.
    w_current = 0.0f; // current total synaptic strength
    //n_neuron = 6;
    //initWeights(n_neuron, n_neuron, 0.2f, 0.0f, this->h_weights);
}

Synapse::~Synapse() {
    //std::cout << "Synapse destructed" << std::endl;
}

void Synapse::initWeights(int in, int out, float connectivity, float inhibitory, std::vector<std::vector<float>>& weight_vec) {
    /* Initialize neuron weights randomly
     * Sign of the weigth is determined by the inhibitory neuron rate
     *
     * An example connection map: (10x10, 0.2 inhibitory, 1.0 connectivity)
     * -------------------
     * | - - + + + + + + | <-- incoming line
     * | - - + + + + + + |
     * | - - + + + + + + |
     * | - - + + + + + + |
     * | - - + + + + + + |
     * | - - + + + + + + |
     * | - - + + + + + + |
     * | - - + + + + + + |
     * -------------------
     * 0<w<1
     *
     * Parameters:
     *      in(int):
     *          incoming connections. in = 10 creates 10 rows
     *      out(int):
     *          outgoing connections. out = 10 creates 10 columns
     *      connectivity(float):
     *          connectivity ratio inside network.
     *          1.0 means fully connected.
     *      inhibitory(float):
     *          inhibitory neuron rate inside network.
     *          1.0 full inhibitory and 0.0 means full excitatory.
     *      weight_vec(std::vector<std::vector<float>>&):
     *          reference to weight vector to be filled.
     */
    int n_inh;
    if (inhibitory > 0) {
        n_inh = static_cast<int>(floorf(1.0f / inhibitory));
    }
    else {
        n_inh = -1;
    }
    float sign = -1.0f;
    weight_vec.resize(in);
    std::vector<std::vector<float>>::iterator it;
    for (it = weight_vec.begin(); it < weight_vec.end(); it++) {
        (*it).resize(out);
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

    for (it_w = weight_vec.begin(); it_w < weight_vec.end(); it_w++) {
        for (it_weight = (*it_w).begin(); it_weight < (*it_w).end(); it_weight++) {
            if (n_inh > 0) {
                sign = (rand() % n_inh) == 0 ? -1.0f : 1.0f;
            }
            else {
                sign = 1.0f;
            }
            if (static_cast <float> (rand()) / static_cast <float> (RAND_MAX) < connectivity) {
                *it_weight = sign * static_cast <float> (rand()) / static_cast <float> (RAND_MAX);
            }
            else {
                *it_weight = 0.0f;
            }
        }
    }
}

void Synapse::initWeights(int in, int out, float connectivity, float inhibitory, CSC*& h_weights) {
    /* 
     * CO :
     * 0 1 1 2 3 4 4
     * RI :
     * 1 2 2 3
     * Data :
     * 0.634938 0.427015 0.15772 0.113651
     *
     * DENSE :
     * |0.000000 0.000000 0.000000 0.000000 0.000000 0.000000 |
     * |0.634938 0.000000 0.000000 0.000000 0.000000 0.000000 |
     * |0.000000 0.000000 0.427015 0.157720 0.000000 0.000000 |
     * |0.000000 0.000000 0.000000 0.000000 0.113651 0.000000 |
     * (2,3): 0.15772
     */

    int n_inh;
    float temp;
    float sign = -1.0f;
    
    std::vector<COO> tempWeights;
    //std::cout << "INIT WEIGHTS" << std::endl;

    if (inhibitory > 0) {
        n_inh = static_cast<int>(floorf(1.0f / inhibitory));
    }
    else {
        n_inh = -1;
    }
    // Connectivity range check
    if (connectivity < 0.0f) {
        connectivity = 0.0f;
    }
    else if (connectivity > 1.0f) {
        connectivity = 1.0f;
    }

    for (int j = 0; j < out; j++) {
        for (int i = 0; i < in; i++) {
            if (n_inh > 0) {
                sign = (rand() % n_inh) == 0 ? -1.0f : 1.0f;
            }
            else {
                sign = 1.0f;
            }
            if (static_cast <float> (rand()) / static_cast <float> (RAND_MAX) < connectivity) {
                temp = sign * static_cast <float> (rand()) / static_cast <float> (RAND_MAX);
                tempWeights.push_back(COO{ i, j, temp });
            }
        }
    }
    COOToCSC(h_weights, tempWeights, in, out);
}

void Synapse::COOToCSC(CSC*& target, const std::vector<COO>& source,int row, int col) {
    //target = new CSC();
    //target->rowSize = row;
    //target->columnSize = col;
    //target->nonzeros = source.size();
    //target->CO = new int[col + 1];
    //target->RI = new int[source.size()];
    //target->data = new float[source.size()];
    target = initCSC(row, col, source.size());

    int k = 0;
    int prevInd = 0;
    int counter = 1;

    target->CO[0] = 0;

    for (k = 0; k < *(target->nonzeros); k++) {
        while (source[k].j != prevInd) {
            target->CO[counter] = k;
            prevInd++;
            counter++;
        }
        target->RI[k] = source[k].i; 
        target->data[k] = source[k].data;
    }
    if (prevInd < col) {
        while (prevInd != col) {
            target->CO[counter] = k;
            prevInd++;
            counter++;
        }
    }
}

void Synapse::CSCToDense(std::vector<std::vector<float>>& target, CSC*& const source) {
    int indice = 0;
    int counter = 0;

    target.resize(*(source->rowSize));
    std::vector<std::vector<float>>::iterator it;
    for (it = target.begin(); it < target.end(); it++) {
        (*it).resize(*(source->columnSize));
    }

    for (int j = 0; j < *(source->columnSize); j++) {
        for (int i = 0; i < *(source->rowSize); i++) {  
            target[i][j] = getDataCSC(source, i, j);
        }
    }
}

float Synapse::getDataCSC(CSC*& target, int i, int j)
{
    if (target->CO[j] == target->CO[j + 1]){
        return 0.0f;
    }
    int colOff = target->CO[j];
    int colMax = target->CO[j + 1];
    for (int c = colOff; c < colMax; c++) {
        if (target->RI[c] == i) {
            return target->data[c];
        }
    }
    return 0.0f;
}

void Synapse::setDataCSC(CSC*& target, int i, int j, const float& data) {
    if (target->CO[j] == target->CO[j + 1]) {
        std::cout << "Empty cell cannot be set by this method" << std::endl;
        return;
    }
    int colOff = target->CO[j];
    int colMax = target->CO[j + 1];
    for (int c = colOff; c < colMax; c++) {
        if (target->RI[c] == i) {
            target->data[c] = data;
            return;
        }
    }
    std::cout << "Empty cell cannot be set by this method" << std::endl;
}

CSC* Synapse::initCSC(int rowSize, int columnSize, int nonzeros)
{
    CSC* target = new CSC();
    target->rowSize = new int(rowSize);
    target->columnSize = new int(columnSize);
    target->nonzeros = new int(nonzeros);
    target->CO = new int[columnSize + 1];
    target->RI = new int[nonzeros];
    target->data = new float[nonzeros];
    return target;
}

void Synapse::deleteCSC(CSC*& target) {
    delete target->rowSize;
    delete target->columnSize;
    delete target->nonzeros;
    delete[] target->CO;
    delete[] target->RI;
    delete[] target->data;
    delete target;
}

cudaError_t Synapse::initCSCDevice(CSC*& d_CSC, CSC*& const h_CSC, bool allocHost, bool alloc) {
    //cudaMalloc((void**)&d_CSC, sizeof(CSC));
    cudaError_t cudaStatus;
    if (allocHost) {
        d_CSC = new CSC(); // this will store 6 device pointers on host.
    }

    if (alloc) {
        cudaStatus = cudaMalloc((void**)&(d_CSC->rowSize), sizeof(int));
        if (cudaStatus != cudaSuccess) {
            fprintf(stderr, "CSC rowSize cudaMalloc failed!");
            return cudaStatus;
        }
    }
    cudaStatus = cudaMemcpy((d_CSC->rowSize), (h_CSC->rowSize), sizeof(int), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "CSC rowsize memcopy h2d failed!");
        return cudaStatus;
    }
    
    

    if (alloc) {
        cudaStatus = cudaMalloc((void**)&(d_CSC->columnSize), sizeof(int));
        if (cudaStatus != cudaSuccess) {
            fprintf(stderr, "CSC columnSize cudaMalloc failed!");
            return cudaStatus;
        }
    }
    cudaStatus = cudaMemcpy((d_CSC->columnSize), (h_CSC->columnSize), sizeof(int), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "CSC columnsize memcopy h2d failed!");
        return cudaStatus;
    }



    if (alloc) {
        cudaStatus = cudaMalloc((void**)&(d_CSC->nonzeros), sizeof(int));
        if (cudaStatus != cudaSuccess) {
            fprintf(stderr, "CSC nonzeros cudaMalloc failed!");
            return cudaStatus;
        }
    } 
    cudaStatus = cudaMemcpy((d_CSC->nonzeros), (h_CSC->nonzeros), sizeof(int), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "CSC columnsize memcopy h2d failed!");
        return cudaStatus;
    }


    if (alloc) {
        cudaStatus = cudaMalloc((void**)&(d_CSC->CO), (*(h_CSC->columnSize) + 1) * sizeof(int));
        if (cudaStatus != cudaSuccess) {
            fprintf(stderr, "CSC CO cudaMalloc failed!");
            return cudaStatus;
        }
    }
    
    cudaStatus = cudaMemcpy((d_CSC->CO), (h_CSC->CO), (*(h_CSC->columnSize) + 1) * sizeof(int), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "CSC CO memcopy h2d failed!");
        return cudaStatus;
    }



    if (alloc) {
        cudaStatus = cudaMalloc((void**)&(d_CSC->RI), (*(h_CSC->nonzeros)) * sizeof(int));
        if (cudaStatus != cudaSuccess) {
            fprintf(stderr, "CSC RI cudaMalloc failed!");
            return cudaStatus;
        }
    }
    
    cudaStatus = cudaMemcpy((d_CSC->RI), (h_CSC->RI), (*(h_CSC->nonzeros)) * sizeof(int), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "CSC RI memcopy h2d failed!");
        return cudaStatus;
    }

    if (alloc) {
        cudaStatus = cudaMalloc((void**)&(d_CSC->data), (*(h_CSC->nonzeros)) * sizeof(float));
        if (cudaStatus != cudaSuccess) {
            fprintf(stderr, "CSC data cudaMalloc failed!");
            return cudaStatus;
        }
    }
    
    cudaStatus = cudaMemcpy((d_CSC->data), (h_CSC->data), (*(h_CSC->nonzeros)) * sizeof(float), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "CSC data memcopy h2d failed!");
        return cudaStatus;
    }
    cudaStatus = cudaDeviceSynchronize();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceSynchronize returned error code %d after initCSCDevice!\n", cudaStatus);
        return cudaStatus;
    }
    return cudaStatus;
}

cudaError_t Synapse::freeCSCDevice(CSC*& d_CSC) {
    cudaError_t cudaStatus;

    cudaStatus = cudaFree(d_CSC->rowSize);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "CSC rowSize cudaFree failed!");
        return cudaStatus;
    }
    cudaStatus = cudaFree(d_CSC->columnSize);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "CSC rowSize cudaFree failed!");
        return cudaStatus;
    }
    cudaStatus = cudaFree(d_CSC->nonzeros);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "CSC rowSize cudaFree failed!");
        return cudaStatus;
    }
    cudaStatus = cudaFree(d_CSC->CO);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "CSC rowSize cudaFree failed!");
        return cudaStatus;
    }
    cudaStatus = cudaFree(d_CSC->RI);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "CSC rowSize cudaFree failed!");
        return cudaStatus;
    }
    cudaStatus = cudaFree(d_CSC->data);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "CSC rowSize cudaFree failed!");
        return cudaStatus;
    }
    delete d_CSC;

    return cudaStatus;
}

cudaError_t Synapse::getDeviceToHostCSC(CSC*& h_CSC, CSC*& const d_CSC) {
    cudaError_t cudaStatus;
    // Copy output vector from GPU buffer to host memory.
    cudaStatus = cudaMemcpy(h_CSC->rowSize, d_CSC->rowSize, sizeof(int), cudaMemcpyDeviceToHost);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "CSC rowSize cudaMemcpy failed!");
        return cudaStatus;
    }

    cudaStatus = cudaMemcpy(h_CSC->columnSize, d_CSC->columnSize, sizeof(int), cudaMemcpyDeviceToHost);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "CSC columnSize cudaMemcpy failed!");
        return cudaStatus;
    }

    cudaStatus = cudaMemcpy(h_CSC->nonzeros, d_CSC->nonzeros, sizeof(int), cudaMemcpyDeviceToHost);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "CSC nonzeros cudaMemcpy failed!");
        return cudaStatus;
    }

    cudaStatus = cudaMemcpy(h_CSC->CO, d_CSC->CO, (*(h_CSC->columnSize) + 1) * sizeof(int), cudaMemcpyDeviceToHost);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "CSC CO cudaMemcpy failed!");
        return cudaStatus;
    }

    cudaStatus = cudaMemcpy(h_CSC->RI, d_CSC->RI, (*(h_CSC->nonzeros)) * sizeof(int), cudaMemcpyDeviceToHost);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "CSC RI cudaMemcpy failed!");
        return cudaStatus;
    }

    cudaStatus = cudaMemcpy(h_CSC->data, d_CSC->data, (*(h_CSC->nonzeros)) * sizeof(float), cudaMemcpyDeviceToHost);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "CSC data cudaMemcpy failed!");
        return cudaStatus;
    }

    return cudaStatus;
}

void Synapse::updateWeights(std::vector<std::vector<float>>& weight_vec,
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
                if ((*it_weight) != 0.0f) {
                    sign = (*it_weight) / abs(*it_weight);
                    if (*it_post) {
                        delta = alpha * (1.0f - abs(*it_weight)) * exp(w_average - w_current);
                    }
                    else {
                        delta = (-1.0f) * alpha * abs(*it_weight) * exp(w_current - w_average);
                    }
                    *it_weight += sign * delta;
                } 
            }
            it_pre++;
        }
        it_post++;
    }
}

void Synapse::updateWeights(CSC*& h_weights,
                            const int& preSize,
                            bool*& const h_preFlags,
                            const int& postSize,
                            bool*& const h_postFlags,
                            const float& alpha,
                            const float& w_average,
                            const float& w_current) {
    std::vector<COO> tempWeights;
    float delta = 0.0f;
    float sign = 1.0f;
    float temp;
    float tempData;

    if (*(h_weights->columnSize) != postSize) {
        std::cout << "Weight matrix width(" << *(h_weights->columnSize) 
            <<") is different than post synaptic vector size"<<(postSize)
            <<"!" << std::endl;
        //return;
    }
    if (*(h_weights->rowSize) != preSize) {
        std::cout << "Weight matrix height(" << *(h_weights->rowSize) 
            <<") is different than pre synaptic vector size(" << preSize 
            <<")!" << std::endl;
        //return;
    }
    //for (int j = 0; j < preSize; j++) {
    //    for (int i = 0; i < postSize; i++) {
    //        tempData = getDataCSC(h_weights, i, j);
    //        if (h_preFlags[j]) {
    //            if (tempData > 0.0f) {
    //                sign = tempData / abs(tempData);
    //            }
    //            if (h_postFlags[i]) {  
    //                delta = alpha * (1.0f - abs(tempData)) * exp(w_average - w_current);
    //            }
    //            else {
    //                delta = (-1.0f) * alpha * abs(tempData) * exp(w_current - w_average);
    //            }
    //            temp = tempData + (sign * delta);
    //            if (temp != 0.0f) {
    //                tempWeights.push_back(COO{ i, j, temp });
    //            }      
    //        }
    //        else {
    //            if (tempData != 0.0f) {
    //                tempWeights.push_back(COO{ i, j, tempData });
    //            }
    //        }
    //    }
    //    
    //}
    //
    //int in = *(h_weights->rowSize);
    //int out = *(h_weights->columnSize);
    //deleteCSC(h_weights);
    //COOToCSC(h_weights, tempWeights, in, out);

    for (int i = 0; i < preSize; i++) {
        for (int j = 0; j < postSize; j++) {
            tempData = getDataCSC(h_weights, i, j);
            if (h_preFlags[i]) {
                if (tempData != 0.0f) {
                    sign = tempData / abs(tempData);
                    if (h_postFlags[i]) {
                        delta = alpha * (1.0f - abs(tempData)) * exp(w_average - w_current);
                    }
                    else {
                        delta = (-1.0f) * alpha * abs(tempData) * exp(w_current - w_average);
                    }
                    setDataCSC(h_weights, i, j, tempData + (sign * delta));
                }
            }
        }
    }
}

void Synapse::updatePre(std::vector<bool>& pre_synaptic_flags,
    const std::vector<FLIF*>& incoming)
{
    std::vector<FLIF*>::const_iterator it;
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

void Synapse::updatePost(std::vector<bool>& post_synaptic_flags,
    const std::vector<FLIF*>& outgoing) {

    std::vector<FLIF*>::const_iterator it;
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

void Synapse::updatePre(bool*& h_preFlags, int& preSize, const std::vector<FLIF*>& incoming) {
    std::vector<FLIF*>::const_iterator it;
    std::vector<bool>::iterator it_f;
    std::vector<bool> pre_synaptic_flags;
    int i = 0;
    delete[] h_preFlags;
    //std::cout << "\nINCOMING SIZE: " << incoming.size() << std::endl;
    for (it = incoming.begin(); it < incoming.end(); it++) {
        //std::cout << "PRE SIZE: " << pre_synaptic_flags.size() << std::endl;
        pre_synaptic_flags.insert(pre_synaptic_flags.end(),
            (*it)->h_flags,
            (*it)->h_flags + (*it)->n_neuron);
    }

    preSize = pre_synaptic_flags.size();
    //std::cout << "PRE SIZE: " << preSize << std::endl;
    h_preFlags = new bool[preSize];

    for (it_f = pre_synaptic_flags.begin(); it_f < pre_synaptic_flags.end(); it_f++) {
        h_preFlags[i] = *it_f;
        i++;
    }

}

void Synapse::updatePost(bool*& const h_postFlags, int& postSize, const std::vector<FLIF*>& outgoing) {
    std::vector<FLIF*>::const_iterator it;
    std::vector<bool>::iterator it_f;
    std::vector<bool> post_synaptic_flags;
    int i = 0;
    delete[] h_postFlags;
    //std::cout << "\nOUTGOING SIZE: " << outgoing.size() << std::endl;
    for (it = outgoing.begin(); it < outgoing.end(); it++) {
        //std::cout << "POST SIZE: " << post_synaptic_flags.size() << std::endl;
        post_synaptic_flags.insert(post_synaptic_flags.end(),
            (*it)->h_flags,
            (*it)->h_flags + (*it)->n_neuron);
    }

    postSize = post_synaptic_flags.size();
    
    h_postFlags = new bool[postSize];

    for (it_f = post_synaptic_flags.begin(); it_f < post_synaptic_flags.end(); it_f++) {
        h_postFlags[i] = *it_f;
        i++;
    }
}

// Connect
void Synapse::addIncomingWeights(std::vector<std::vector<float>>& resting,
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

void Synapse::addOutgoingWeights(std::vector<std::vector<float>>& resting,
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


void Synapse::addIncomingWeights(CSC*& resting, CSC*& const in) {
    // Be sure that incoming column size is equal or smaller
    int row_OFF = *(resting->rowSize);
    int row = *(resting->rowSize) + *(in->rowSize);
    int col = *(resting->columnSize);
    int nonzeros = *(resting->nonzeros) + *(in->nonzeros);

    int counter = 0;

    CSC* target = initCSC(row, col, nonzeros);

    int j = 0;
    int k = 0;
    for (int i = 0; i < col + 1; i++) {
        target->CO[i] = counter;
        for (j = resting->CO[i]; j < resting->CO[i + 1]; j++) {
            target->RI[j + k] = resting->RI[j];
            target->data[j + k] = resting->data[j];
            counter++;
        }
        if (i < *(in->columnSize)) { // check if in is smaller
            for (k = in->CO[i]; k < in->CO[i + 1]; k++) {
                target->RI[j + k] = row_OFF + in->RI[k];
                target->data[j + k] = in->data[k];
                counter++;
            }
        }
    }

    deleteCSC(resting);
    resting = target;
    
}

void Synapse::addOutgoingWeights(CSC*& resting, CSC*& const out) {
    // Be sure that outgoing row size is equal or smaller

    int CO_OFF = resting->CO[*(resting->columnSize)];

    int row = *(resting->rowSize);
    int col = *(resting->columnSize) + *(out->columnSize);
    int nonzeros = *(resting->nonzeros) + *(out->nonzeros);

    int counter = 0;

    CSC* target = initCSC(row, col, nonzeros);

    // CO
    for (int i = 0; i < *(resting->columnSize) + 1; i++) {
        target->CO[i] = resting->CO[i];
    }

    for (int i = *(resting->columnSize); i < col + 1; i++) {
        target->CO[i] = CO_OFF + out->CO[counter];
        counter++;
    }

    counter = 0;
    //RI
    for (int i = 0; i < *(resting->nonzeros); i++) {
        target->RI[i] = resting->RI[i];
    }

    for (int i = *(resting->nonzeros); i < nonzeros; i++) {
        target->RI[i] = out->RI[counter];
        counter++;
    }

    counter = 0;
    //DATA
    for (int i = 0; i < *(resting->nonzeros); i++) {
        target->data[i] = resting->data[i];
    }

    for (int i = *(resting->nonzeros); i < nonzeros; i++) {
        target->data[i] = out->data[counter];
        counter++;
    }
    deleteCSC(resting);
    resting = target;
}

void Synapse::connectIn(FLIF* incoming,
    float strength,
    float inhibitory) {

    /*(this->incomingList).push_back(incoming);
    std::vector<std::vector<float>> inWeights;
    initWeights(getN(), incoming->getN(), strength, inhibitory, inWeights);
    addIncomingWeights(this->weights, inWeights);*/

    (this->incomingList).push_back(incoming);
    CSC* inWeights;
    initWeights(incoming->getN(), this->getN(), strength, inhibitory, inWeights);
    addIncomingWeights(this->h_weights, inWeights);
}

void Synapse::connectOut(FLIF* outgoing,
    float strength,
    float inhibitory) {

    /*(this->outgoingList).push_back(outgoing);
    std::vector<std::vector<float>> outWeights;
    initWeights(outgoing->getN(), getN(), strength, inhibitory, outWeights);
    addOutgoingWeights(this->weights, outWeights);*/

    (this->outgoingList).push_back(outgoing);
    CSC* outWeights;
    initWeights(this->getN(), outgoing->getN(), strength, inhibitory, outWeights);
    addOutgoingWeights(this->h_weights, outWeights);
}

void Synapse::connect(Synapse* pre_synaptic, float pre_strength, float pre_inhibitory,
    Synapse* post_synaptic, float post_strength, float post_inhibitory) {
    post_synaptic->connectIn(pre_synaptic, pre_strength, pre_inhibitory);
    pre_synaptic->connectOut(post_synaptic, post_strength, post_inhibitory);
}

void Synapse::connect_GPU(Synapse* pre_synaptic, float pre_strength, float pre_inhibitory,
    Synapse* post_synaptic, float post_strength, float post_inhibitory) {
    post_synaptic->connectIn(pre_synaptic, pre_strength, pre_inhibitory);
    post_synaptic->connectRestore_GPU();

    pre_synaptic->connectOut(post_synaptic, post_strength, post_inhibitory);
    pre_synaptic->connectRestore_GPU();
    
}

void Synapse::connectRestore_GPU() {
    this->freeCSCDevice(this->d_weights);
    this->initCSCDevice(this->d_weights, this->h_weights);
}

void Synapse::POC() {
    Synapse* syn = new Synapse();
    Synapse* syn2 = new Synapse();
    std::cout << "Nonzeros: " <<*(syn->h_weights->nonzeros) << std::endl;
    std::cout << "CO :" << std::endl;
    for (int i = 0; i <= *(syn->h_weights->columnSize); i++) {
        std::cout << syn->h_weights->CO[i] << " " ;
    }
    std::cout << "\nRI :" << std::endl;
    for (int i = 0; i < *(syn->h_weights->nonzeros); i++) {
        std::cout << syn->h_weights->RI[i] << " ";
    }
    std::cout << "\nData :" << std::endl;
    for (int i = 0; i < *(syn->h_weights->nonzeros); i++) {
        std::cout << syn->h_weights->data[i] << " ";
    }
    std::vector<std::vector<float>> target;
    syn->CSCToDense(target, syn->h_weights);
    std::vector<std::vector<float>>::iterator it_w;

    std::cout << std::endl << "DENSE :" << std::endl;
    for (it_w = target.begin(); it_w < target.end(); it_w++) {
        std::cout <<  "|" << syn->vectorToString<float>(*it_w) << "|\n";
    }
    std::cout << "(2,3): " <<syn->getDataCSC(syn->h_weights, 2,3) << std::endl;
    bool* pre;
    pre = new bool[6];
    for (int i = 0; i < 6; i++) {
        pre[i] = true;
    }
    bool* post;
    post = new bool[4];
    for (int i = 0; i < 4; i++) {
        post[i] = true;
    }
    //syn->updateWeights(syn->h_weights, 6, pre, 4, post, 0.2f, 0.0f, 0.0f);

    for (int i = 0; i <= *(syn2->h_weights->columnSize); i++) {
        std::cout << syn2->h_weights->CO[i] << " ";
    }
    std::cout << "\nRI :" << std::endl;
    for (int i = 0; i < *(syn2->h_weights->nonzeros); i++) {
        std::cout << syn2->h_weights->RI[i] << " ";
    }
    std::cout << "\nData :" << std::endl;
    for (int i = 0; i < *(syn2->h_weights->nonzeros); i++) {
        std::cout << syn2->h_weights->data[i] << " ";
    }

    syn->CSCToDense(target, syn2->h_weights);


    std::cout << std::endl << "DENSE :" << std::endl;
    for (it_w = target.begin(); it_w < target.end(); it_w++) {
        std::cout << "|" << syn2->vectorToString<float>(*it_w) << "|\n";
    }
    std::cout << "(2,3): " << syn2->getDataCSC(syn2->h_weights, 2, 3) << std::endl;



    //syn->addOutgoingWeights(syn->h_weights, syn2->h_weights);
    Synapse::connect(syn, 0.4, 0.1, syn2, 0.6, 0.2);
    std::cout << "Nonzeros: " << *(syn->h_weights->nonzeros) << std::endl;
    std::cout << "CO :" << std::endl;
    for (int i = 0; i <= *(syn->h_weights->columnSize); i++) {
        std::cout << syn->h_weights->CO[i] << " ";
    }
    std::cout << "\nRI :" << std::endl;
    for (int i = 0; i < *(syn->h_weights->nonzeros); i++) {
        std::cout << syn->h_weights->RI[i] << " ";
    }
    std::cout << "\nData :" << std::endl;
    for (int i = 0; i < *(syn->h_weights->nonzeros); i++) {
        std::cout << syn->h_weights->data[i] << " ";
    }

    syn->CSCToDense(target, syn->h_weights);


    std::cout << std::endl << "DENSE :" << std::endl;
    for (it_w = target.begin(); it_w < target.end(); it_w++) {
        std::cout << "|" << syn->vectorToString<float>(*it_w) << "|\n";
    }
    std::cout << "(2,3): " << syn->getDataCSC(syn->h_weights, 2, 3) << std::endl;


    syn->deleteCSC(syn->h_weights);



   /* syn->initWeights(8, 8, 0.2f, 0.0f, syn->h_weights);


    std::cout << "Nonzeros: " << syn->h_weights->nonzeros << std::endl;
    std::cout << "CO :" << std::endl;
    for (int i = 0; i <= syn->h_weights->columnSize; i++) {
        std::cout << syn->h_weights->CO[i] << " ";
    }
    std::cout << "\nRI :" << std::endl;
    for (int i = 0; i < syn->h_weights->nonzeros; i++) {
        std::cout << syn->h_weights->RI[i] << " ";
    }
    std::cout << "\nData :" << std::endl;
    for (int i = 0; i < syn->h_weights->nonzeros; i++) {
        std::cout << syn->h_weights->data[i] << " ";
    }

    syn->CSCToDense(target, syn->h_weights);

    std::cout << std::endl << "DENSE :" << std::endl;
    for (it_w = target.begin(); it_w < target.end(); it_w++) {
        std::cout << "|" << syn->vectorToString<float>(*it_w) << "|\n";
    }
    std::cout << std::endl;
    syn->deleteCSC(syn->h_weights);*/

    //std::cout << syn->vectorToString<int>(vec);
}