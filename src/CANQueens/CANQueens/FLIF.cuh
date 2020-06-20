/* Fatigue Leaky Integrate and Fire Neuron Class Header File
 * Parent class for Explore Memory and CA
 *
 * 200616
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
#include <vector>
#include <fstream>
#include <time.h>
#include <windows.h>

struct REC {
    // 0000 represent none 1111 all 0101 energy and weights
    int available; 
    std::vector<bool> flags;
    std::vector<float> energy;
    std::vector<float> fatigue;
    std::vector<std::vector<float>> weights;
};

struct REC_SIZE {
    int start;
    int stop;
    bool check;
};

class FLIF {

protected:
    static int nextID;
    int ID;

    int n_neuron;
    float activity;
    float connectivity;
    float inhibitory;

    std::vector<REC> record;

    // Neurons
    std::vector<bool> flags; // firing flags phi
    std::vector<float> energy; // energy levels
    std::vector<float> fatigue; // fatigue levels
    std::vector<std::vector<float>> weights;

    // Inits
    void initFlags(int n, float activity,
        std::vector<bool>& flag_vec);
    void initEF(int n, float upper, float lower,
        std::vector<float>& EF_vec);

    // Updates
    void updateFlags(std::vector<bool>& flag_vec,
                     const float& activity);

    //Methods
    std::string dateTimeStamp(const char* filename);
    int num_fire(std::vector<bool>& firings);
    REC_SIZE sizeCheckRecord(int stop, int start);
    REC setRecord(int available);

    template <typename T>
    std::string vectorToString(const std::vector<T>& vec);    
    template <typename T>
    void vectorToCSV(std::ostream& file, const std::vector<T>& entry);

    void getWeightCSV(char* filename, int stop, int start);
   

    friend class Synapse;
public:
    //Constructors
    FLIF();
    ~FLIF();

    // Printing
    std::string getRecord(int timeStep);
    std::string getActivity(int stop = -1, int start = 0);
    std::string getRaster(float threshold = 0.0f, int stop = -1, int start = 0);
    //void getRasterCSV(char* filename, float threshold = 0.0f, int stop = -1, int start = 0);
    void getCSV(char* filename, int type, float threshold = 0.0f, int stop = -1, int start = 0);
    void saveRecord(char* filename, float threshold = 0.0f, int stop = -1, int start = 0);
    void saveCSV(char* filename, float threshold = 0.0f, int stop = -1, int start = 0);

    // GET
    int getID();
    int getN();
    std::string getInfo();

    // Set
    void setActivity(float act);
    

    //CUDA_CALLABLE_MEMBER std::string toString();
};

#endif // FLIF_H


