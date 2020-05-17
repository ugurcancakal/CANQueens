/* Memory Class Source File
 *
 * 200517
 * author = @ugurc
 * ugurcan.cakal@gmail.com
 */

#include "Memory.cuh"

Memory::Memory() {
    std::cout << "Memory constructed" << std::endl;
}

Memory::~Memory() {
    std::cout << "Memory destructed" << std::endl;
}

std::string Memory::toString() {
    return "Memory";
}