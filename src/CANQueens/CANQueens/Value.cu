/* Value Class Source File
 *
 * 200517
 * author = @ugurc
 * ugurcan.cakal@gmail.com
 */

#include "Value.cuh"

__global__ void fitness_kernel(int* chromosome, int* collision) {
    /*unsigned int index = threadIdx.x + blockDim.x * blockIdx.x;
    unsigned int stride = blockDim.x * gridDim.x;*/
    unsigned int tid = threadIdx.x;
    unsigned int bid = blockIdx.x;
    int temp = chromosome[bid];
    int d = 0;
    extern __shared__ int cache[]; // to use the thread-block shared memory
    cache[tid] = 0;
    if (tid < bid) {
        d = abs(temp - chromosome[tid]);
        if ((d == 0) || (d == (bid - tid))) {
            cache[tid] = 1;
        }
        else {
            cache[tid] = 0;
        }
    }
    
    __syncthreads();

    //Reduction
    unsigned int i = blockDim.x / 2;
    while (i >0) {
        if (tid < i) {
            cache[tid] += cache[tid + i];
        }
        __syncthreads();
        i /= 2;
    }

    if (tid == 0) {
        atomicAdd(collision, cache[0]);
    }

    /*while (index < n) {
        temp = chromosome[index];
        index += stride;
    }*/
}

Value::Value(int n) {
    row = n;
    h_collision = new int[n];
    h_collision = new int(0);
    initIntDevice(n, this->d_chromosome, h_collision);
    initIntDevice(1, this->d_collision, h_collision);
    //std::cout << "Value constructed" << std::endl;
}

Value::~Value() {
    std::cout << "Value destructed" << std::endl;
}

std::string Value::toString() {
    return "Value";
}

void Value::update(int* chromosome) {
    std::cout << fitness(chromosome) << std::endl;
}

float Value::activity(int n, int* chromosome) {
    return (fitness(chromosome) *1.0f) / (maxCollision(n)*1.0f);
}

int Value::maxCollision(int n) {
    return ((n * (n - 1)) / 2);
}

std::string Value::getInfo() {
    return "HI\n";
}

cudaError_t Value::initIntDevice(int n, int*& d_vec, int*& const h_vec, bool alloc){
    cudaError_t cudaStatus;
    if (alloc) {
        cudaStatus = cudaMalloc((void**)&d_vec, n * sizeof(int));
        if (cudaStatus != cudaSuccess) {
            fprintf(stderr, "int cudaMalloc failed!");
            return cudaStatus;
        }
    }
    cudaStatus = cudaMemcpy(d_vec, h_vec, n * sizeof(int), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "int h2d cudaMemcpy failed!");
        return cudaStatus;
    }
    return cudaStatus;
}

int Value::fitness(int* chromosome) {
    /*int collision = 0;
    int d;
    for (int i = row; i >= 0; i--) {
        for (int k = i - 1; k >= 0; k--) {
            d = abs(chromosome[i] - chromosome[k]);
            if ((d == 0) || (d == i - k)) {
                collision++;
            }
        }
    }
    return collision;*/
    int collision = 0;
    int d;
    for (int i = 0; i < row; i++) {
        //std::cout << "i: " << i << std::endl;
        if (chromosome[i] >= row) {
            collision += 666;
        }
        for (int k = i+1; k < row; k++) {
            //std::cout << "k: " << k << std::endl;
            d = abs(chromosome[k] - chromosome[i]);
            if ((d == 0) || (d == k-i)) {
                collision++;
            }
        }
    }
    return collision;
}

cudaError_t Value::errorCheckCUDA(bool synchronize) {
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

cudaError_t Value::getDeviceToHostCh(const int n, int*& h_chromosome, int*& const d_chromosome) {
    cudaError_t cudaStatus;
    // Copy output vector from GPU buffer to host memory.
    cudaStatus = cudaMemcpy(h_chromosome, d_chromosome, n * sizeof(int), cudaMemcpyDeviceToHost);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "Chromosome cudaMemcpy failed!");
        return cudaStatus;
    }
    return cudaStatus;
}

//int Value::fitness(int* chromosome, dim3 gridSize, dim3 blockSize, size_t shared, bool synchronize, bool memCopy) {
//    cudaError_t cudaStatus;
//
//    initIntDevice(row, this->d_chromosome, chromosome, false);
//    fitness_kernel << <row,row,row>> > (d_chromosome, d_collision);
//    cudaStatus = errorCheckCUDA(synchronize);
//
//    if (memCopy) {
//        cudaStatus = getDeviceToHostCh(1, this->h_collision, this->d_collision);
//    }
//    return *h_collision;
//}
