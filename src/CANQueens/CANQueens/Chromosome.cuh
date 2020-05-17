/* Chromosome Class Header File
 *
 * 200516
 * author = @ugurc
 * ugurcan.cakal@gmail.com
 */

#ifdef __CUDACC__
#define CUDA_CALLABLE_MEMBER __host__ __device__
#else
#define CUDA_CALLABLE_MEMBER
#endif 

#ifndef CHROMOSOME_H
#define CHROMOSOME_H

#include <string>
#include <iostream>

class Chromosome {
public:
    Chromosome();
    ~Chromosome();
    std::string toString();
    //CUDA_CALLABLE_MEMBER std::string toString();
};

#endif // CHROMOSOME_H