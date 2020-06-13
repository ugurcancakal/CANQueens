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
#include "CA.cuh"

//class Explore : public CA{
class Explore {
private:
    CA* explore;

protected:
    static int n_neuron;
    static float inh;
    static float threshold;
    static float C[7];
public:
    Explore();
    ~Explore();
    std::string toString();
    void update();
    //CUDA_CALLABLE_MEMBER std::string toString();
};

#endif // EXPLORE_H