/* Cell Assembly Class Source File
 *
 * 200516
 * author = @ugurc
 * ugurcan.cakal@gmail.com
 */

#include "CA.cuh"

int CA::nextID = 0;

CA::CA() {
    ignition = (0 == (rand() % 2));
    ID = ++nextID;
    //std::cout << "CA constructed with ID: " << ID << std::endl;
}

CA::~CA() {
    //std::cout << "CA destructed" << std::endl;
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
