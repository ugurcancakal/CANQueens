/* Cell Assembly Class Source File
 *
 * 200516
 * author = @ugurc
 * ugurcan.cakal@gmail.com
 */

#include "CA.cuh"

CA::CA() {
    std::cout << "CA constructed" << std::endl;
}

CA::~CA() {
    std::cout << "CA destructed" << std::endl;
}

std::string CA::toString() {
    return "CA";
}

bool CA::getIgnition() {
	return ignition;
}

int CA::getID() {
    return ID;
}
