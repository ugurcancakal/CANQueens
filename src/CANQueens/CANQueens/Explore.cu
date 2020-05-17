/* Explore Class Source File
 *
 * 200517
 * author = @ugurc
 * ugurcan.cakal@gmail.com
 */

#include "Explore.cuh"

Explore::Explore() {
    std::cout << "Explore constructed" << std::endl;
}

Explore::~Explore() {
    std::cout << "Explore destructed" << std::endl;
}

std::string Explore::toString() {
    return "Explore";
}