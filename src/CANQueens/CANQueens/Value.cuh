/* Value Class Header File
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

#ifndef VALUE_H
#define VALUE_H

#include <string>
#include <iostream>

class Value {
private:
    int row;
    
public:
    int fitness(int* chromosome);
    Value(int n = 8);
    ~Value();
    std::string toString();
    void update(int* chromosome);
    
    //CUDA_CALLABLE_MEMBER std::string toString();
};

#endif // VALUE_H