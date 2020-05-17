/* Chromosome Class Source File
 *
 * 200517
 * author = @ugurc
 * ugurcan.cakal@gmail.com
 */

#include "Chromosome.cuh"

Chromosome::Chromosome() {
    std::cout << "Chromosome constructed" << std::endl;
}

Chromosome::~Chromosome() {
    std::cout << "Chromosome destructed" << std::endl;
}

std::string Chromosome::toString() {
    return "Chromosome";
}