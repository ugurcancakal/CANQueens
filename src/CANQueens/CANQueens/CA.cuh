/* Cell Assembly Class Header File
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

#ifndef CA_H
#define CA_H

#include <string>
#include <iostream>

class CA {

private:
    bool ignition;
    int ID;

protected:
    static int nextID;

public:
    CA();
    ~CA();
    std::string toString();
    //CUDA_CALLABLE_MEMBER std::string toString();
    bool getIgnition();
    int getID();

};

#endif // CA_H