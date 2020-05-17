/* Fatigue Leaky Integrate and Fire Neuron Class Header File
 *
 * 200517
 * author = @ugurc
 * ugurcan.cakal@gmail.com
 */

#ifdef __CUDACC__
#define CUDA_CALLABLE_MEMBER __host__ __device__
#else
#define CUDA_CALLABLE_MEMBER
#endif 

#ifndef FLIF_H
#define FLIF_H

#include <string>
#include <iostream>

class FLIF {
public:
    FLIF();
    ~FLIF();
    std::string toString();
    //CUDA_CALLABLE_MEMBER std::string toString();
};

#endif // FLIF_H