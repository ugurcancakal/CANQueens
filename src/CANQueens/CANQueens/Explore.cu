/* Explore Class Source File
 *
 * 200517
 * author = @ugurc
 * ugurcan.cakal@gmail.com
 */

#include "Explore.cuh"

int Explore::n_neuron = 400;
float Explore::inh = 0.3f;
float Explore::threshold = 0.3f;
float Explore::C[7] = { 1.0, 1.0, 1.0, 1.0, 0.2, 1.0, 1.0 };

Explore::Explore() {
    std::cout << "Explore constructed" << std::endl;
    explore = new CA(n_neuron, threshold, inh, C);
}

Explore::~Explore() {
    std::cout << "Explore destructed" << std::endl;
}

std::string Explore::toString() {
    return "Explore";
}