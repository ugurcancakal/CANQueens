/* Fatigue Leaky Integrate and Fire Neuron Class Source File
 *
 * 200517
 * author = @ugurc
 * ugurcan.cakal@gmail.com
 */

#include "FLIF.cuh"

FLIF::FLIF() {
    std::cout << "FLIF constructed" << std::endl;
}

FLIF::~FLIF() {
    std::cout << "FLIF destructed" << std::endl;
}

std::string FLIF::toString() {
    return "FLIF";
}