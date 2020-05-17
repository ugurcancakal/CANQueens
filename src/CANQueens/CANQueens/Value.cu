/* Value Class Source File
 *
 * 200517
 * author = @ugurc
 * ugurcan.cakal@gmail.com
 */

#include "Value.cuh"

Value::Value() {
    std::cout << "Value constructed" << std::endl;
}

Value::~Value() {
    std::cout << "Value destructed" << std::endl;
}

std::string Value::toString() {
    return "Value";
}