/* Explore Class Header File
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

#ifndef EXPLORE_H
#define EXPLORE_H

#include <string>
#include <iostream>

class Explore {
public:
    Explore();
    ~Explore();
    std::string toString();
    //CUDA_CALLABLE_MEMBER std::string toString();
};

#endif // EXPLORE_H