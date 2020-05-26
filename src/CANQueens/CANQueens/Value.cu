/* Value Class Source File
 *
 * 200517
 * author = @ugurc
 * ugurcan.cakal@gmail.com
 */

#include "Value.cuh"

Value::Value(int n) {
    row = n;
    std::cout << "Value constructed" << std::endl;
}

Value::~Value() {
    std::cout << "Value destructed" << std::endl;
}

std::string Value::toString() {
    return "Value";
}

void Value::update(int* chromosome) {
    std::cout << fitness(chromosome) << std::endl;
}

int Value::fitness(int* chromosome) {
    // KONTROL ET
    int collision = 0;
    int d;
    for (int i = row; i >= 0; i--) {
        for (int k = i - 1; k >= 0; k--) {
            d = abs(chromosome[i] - chromosome[k]);
            if ((d == 0) || (d == i - k)) {
                collision++;
            }
        }
    }
    return collision;
}
