/* Memory Class Header File
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

#ifndef MEMORY_H
#define MEMORY_H

#include <string>
#include <iostream>

class Memory {
public:
    Memory();
    ~Memory();
    std::string toString();
    //CUDA_CALLABLE_MEMBER std::string toString();
};

#endif // MEMORY_H