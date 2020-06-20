/* Fatigue Leaky Integrate and Fire Neuron Class Source File
 * Parent class for Explore Memory and CA
 *
 * 200616
 * author = @ugurc
 * ugurcan.cakal@gmail.com
 */

#include "FLIF.cuh"


int FLIF::nextID = 0;

 // Inits

void FLIF::initFlags(int n, float activity, std::vector<bool>& flag_vec) {
    /* Initialize firing flags randomly
     *
     * Parameters:
     *      n(int):
     *          number of neurons
     *      activity(float):
     *          activity rate of neurons. 1.0 result in always fire.
     *      inhibitory(float):
     *          inhibitory neuron rate inside network.
     *          1.0 full inhibitory and 0.0 means full excitatory.
     *      flag_vec(std::vector<bool>&):
     *          reference to flag vector to be filled.
     */
    std::vector<bool>::iterator it;
    flag_vec.resize(n);
    
    for (it = flag_vec.begin(); it < flag_vec.end(); it++) {
        *it = (0 == (rand() % static_cast<int>(floorf(1.0f / activity))));
    }
}


void FLIF::initEF(int n, float upper, float lower, std::vector<float>& EF_vec) {
    /* Initialize energy/fatigueness levels randomly
     *
     * Parameters:
     *      n(int):
     *          number of neurons
     *      upper(float):
     *          upper bound for the level
     *      lower(float):
     *          lower bound for the level
     *      EF_vec(std::vector<float>&):
     *          reference to energy/fatigueness vector to be filled.
     */
    float temp;
    std::vector<float>::iterator it;
    EF_vec.resize(n);

    for (it = EF_vec.begin(); it < EF_vec.end(); it++) {
        temp = static_cast <float> (rand()) / static_cast <float> (RAND_MAX);
        temp *= (upper - lower);
        (*it) = temp - lower;
    }
}

// Updates
void FLIF::updateFlags(std::vector<bool>& flag_vec,
    const float& activity) {

    std::vector<bool>::iterator it;
    for (it = flag_vec.begin(); it < flag_vec.end(); it++) {
        *it = 0 == (rand() % static_cast<int>(floorf(1.0f / activity)));
    }
}


// Methods

std::string FLIF::dateTimeStamp(const char* filename) {
    /* Create an @ugurc format timestamp
     * For example the date 09 May 1995 and time 02:48:05
     * is encoded like year-month-day-hour-minute-second
     * 950509024805
     *
     * Parameters:
     *      filename (const char*):
     *          filename to be concatenated with the timestamp
     *
     * Returns:
     *		nameSTAMP(std::string):
     *			name with a dateTimeStamp like ugurc950509024805
     */

    time_t t = time(0);   // get time now
    struct tm* now = localtime(&t);
    char buffer[80];
    strftime(buffer, 80, "%y%m%d%H%M%S", now);
    std::string dateTime = buffer;
    return filename + dateTime;
}

int FLIF::num_fire(std::vector<bool>& firings) {
    /* Number of neurons fired in a given flag vector.
     *
     * Parameters:
     *      firings(std::vector<bool>&):
     *          firing flag vector consisting of booleans
     *
     * Returns:
     *      num(int):
     *          total number of fire
     */
    int fire = 0;
    std::vector<bool>::iterator it;
    for (it = firings.begin(); it < firings.end(); it++) {
        if (*it) {
            fire++;
        }
    }
    return fire;
}

REC_SIZE FLIF::sizeCheckRecord(int stop, int start) {
    REC_SIZE temp;
    temp.check = true;
    //std::cout << "SIZE CHECK" << std::endl;
    //std::cout << record[0].energy.size() << std::endl;
    int range = stop - start;
    if (start < 0) {
        std::cout << "Starting point cannot be less than 0!" << std::endl;
        start = 0;
        range = stop - start;
        std::cout << "Starting point has fixed to " << start << std::endl;
    }
    if (stop < start) {
        std::cout << "End point cannot be less than start " << start << "!" << std::endl;
        stop = start + abs(range);
        range = stop - start;
        std::cout << "End point has fixed to " << stop << std::endl;
    }

    if (stop > record.size()) {
        std::cout << "Not enough record to print " << stop << "(st/nd/rd/th) step!" << std::endl;
        stop = record.size();
        std::cout << "End point has fixed to " << stop << std::endl;
        if (stop - range > 0) {
            start = stop - range;
        }
        else {
            start = 0;
        }
        range = stop - start;
        std::cout << "Starting point has fixed to " << start << std::endl;
    }

    if (range <= 0) {
        std::cout << "Range is " << range << "!" << std::endl;
        temp.check = false;
    }

    temp.start = start;
    temp.stop = stop;
    return temp;
}

REC FLIF::setRecord(int available) {
    REC temp;
    temp.available = available;
    if ((available | 0b0111) == 0b1111) {
        //std::cout << (this->flags).size() << std::endl;
        temp.flags = this->flags;
    }

    if ((available | 0b1011) == 0b1111) {
        //std::cout << "rec2" << std::endl;
        temp.energy = this->energy;
    }

    if ((available | 0b1101) == 0b1111) {
        //std::cout << "rec3" << std::endl;
        temp.fatigue = this->fatigue;
    }

    if ((available | 0b1110) == 0b1111) {
        //std::cout << "rec4" << std::endl;
        temp.weights = this->weights;
    }
    
    return temp;
}

template<typename T>
std::string FLIF::vectorToString(const std::vector<T>& vec) {
    /* Convert a vector of U type to an std::string using
     * " " as delimiter.
     * 
     * Parameters:
     *      vec(std::vector<U>&):
     *          vector to be printed
     *
     * Returns:
     *      vec_string(std::string):
     *          string form of the vector
     */
    std::string temp = "";
    typename std::vector<T>::const_iterator it;
    for (it = vec.begin(); it < vec.end(); it++) {
        temp += std::to_string(*it) + " ";
    }
    return temp;
}

template<typename T>
void FLIF::vectorToCSV(std::ostream& file, const std::vector<T>& entry) {
    typename std::vector<T>::const_iterator it;
    bool start = true;
    for (it = entry.begin(); it < entry.end(); it++) {
        if (start) {
            file << *it;
            start = false;
        }
        else {
            file << " , " << *it;
        }
    }
    file << std::endl;
}

// Constructors

FLIF::FLIF() {
    ID = ++nextID;
    n_neuron = 0;
    activity = 0.0f;
    connectivity = 0.0f;
    inhibitory = 0.0f;
}

FLIF::~FLIF() {
}

std::string FLIF::getRecord(int timeStep) {
    std::string temp = "\n";

    temp += "timeStep " + std::to_string(timeStep) + "\n";

    if ((record[timeStep].available | 0b0111) == 0b1111) {
        temp += "\nFlags ["+ std::to_string(activity) +"] : ";
        temp += "(" + std::to_string(num_fire(record[timeStep].flags)) +
            "/" + std::to_string(n_neuron) + ")\n";
        temp += vectorToString<bool>(record[timeStep].flags);
    }
    
    if ((record[timeStep].available | 0b1011) == 0b1111) {
        temp += "\n\nEnergy Levels \n";
        temp += vectorToString<float>(record[timeStep].energy);
    }
    
    if ((record[timeStep].available | 0b1101) == 0b1111) {
        temp += "\n\nFatigue Levels \n";
        temp += vectorToString<float>(record[timeStep].fatigue);
    }

    if ((record[timeStep].available | 0b1110) == 0b1111) {
        temp += "\n\nWeights \n";
        std::vector<std::vector<float>>::iterator it_w;
        for (it_w = record[timeStep].weights.begin(); it_w < record[timeStep].weights.end(); it_w++) {
            temp += "|" + vectorToString<float>(*it_w) + "|\n";
        }
    }
    temp += "\n";
    return temp;
}

std::string FLIF::getActivity(int stop, int start) {

    if (stop == -1) {
        stop = record.size();
    }

    REC_SIZE rec = sizeCheckRecord(stop, start);
    start = rec.start;
    stop = rec.stop;
    // Size Check
    if (!rec.check) {
        std::cout << "Activity cannot be shown!" << std::endl;
        return "NA";
    }

    std::string temp = "\n";
    temp += "CA ID: " + std::to_string(getID()) + "\n";

    for (int i = start; i < stop; i++) {
        temp+=  getRecord(i) + "\n";
    }

    return temp;

}

std::string FLIF::getRaster(float threshold, int stop, int start) {
    /* Construct the string representing whole raster plot
     * for given time interval.
     *
     *  N_ID    ||         SPIKE ACTIVITY
     *  --------------------------------------------
     *  0       ||      |
     *  1       ||      |               |
     *  2       ||      |                       |
     *  3       ||
     *  --------------------------------------------
     *  TIME    ||      0       1       2       3
     *  --------------------------------------------
     *  FIRE    ||      2       0       1       1
     *  --------------------------------------------
     *  IGNIT   ||      1       0       1       1
     *
     * Parameters:
     *      start(int):
     *          starting timestep
     *      stop(int):
     *          ending timestep
     *      threshold(float):
     *          minimum rate of firing to ignit (0 by default)
     *          to show the ignit line, it must be greater than 0.
     *
     * Returns:
     *      raster(std::string):
     *          raster plot
     */
    if (stop == -1) {
        stop = record.size();
    }

    int range = stop - start;
    std::string temp = "\n";
    int n_threshold = threshold * n_neuron;
    
    REC_SIZE rec = sizeCheckRecord(stop, start);
    start = rec.start;
    stop = rec.stop;

    // Size Check
    if (!rec.check) {
        std::cout << "Raster cannot be plotted!" << std::endl;
        return "NA";
    }

    if ((record[0].available | 0b0111) != 0b1111) {
        std::cout << "No firing record available!" << std::endl
            << "Raster cannot be plotted!" << std::endl;
        return "NA";
    }

    // Header
    temp += "  \t";
    temp += " \n";
    temp += "N_ID \t||";
    temp += std::string(3 * (range), ' ');
    temp += "SPIKE ACTIVITY\n";
    temp += std::string(8 * (range + 1) + 4, '-') + "\n";

    // Body
    for (int i = 0; i < n_neuron; i++) {
        temp += std::to_string(i);
        temp += "\t||\t";
        for (int j = start; j < stop; j++) {
            if (record[j].flags[i]) {
                temp += "|\t";
            }
            else {
                temp += " \t";
            }
        }
        temp += "\n";
    }

    temp += std::string(8 * (range + 1) + 4, '-') + "\n";
    temp += "TIME \t||\t";
    for (int i = start; i < stop; i++) {
        temp += std::to_string(i) + "\t";
    }
    temp += "\n";
    temp += std::string(8 * (range + 1) + 4, '-') + "\n";
    temp += "FIRE \t||\t";
    for (int i = start; i < stop; i++) {
        temp += std::to_string(num_fire(record[i].flags)) + "\t";
    }
    temp += "\n";
    temp += std::string(8 * (range + 1) + 4, '-') + "\n";
    if (threshold > 0.0f) {
        temp += "IGNIT \t||\t";
        for (int i = start; i < stop; i++) {
            temp += std::to_string(num_fire(record[i].flags) >= n_threshold) + "\t";
        }
    }
    
    return temp;
}

//void FLIF::getRasterCSV(char* filename, float threshold, int stop, int start) {
//    /* Construct the string representing whole raster plot
//     * for given time interval.
//     *
//     *  N_ID    ||         SPIKE ACTIVITY
//     *  --------------------------------------------
//     *  0       ||      |
//     *  1       ||      |               |
//     *  2       ||      |                       |
//     *  3       ||
//     *  --------------------------------------------
//     *  TIME    ||      0       1       2       3
//     *  --------------------------------------------
//     *  FIRE    ||      2       0       1       1
//     *  --------------------------------------------
//     *  IGNIT   ||      1       0       1       1
//     *
//     * Parameters:
//     *      start(int):
//     *          starting timestep
//     *      stop(int):
//     *          ending timestep
//     *      threshold(float):
//     *          minimum rate of firing to ignit (0 by default)
//     *          to show the ignit line, it must be greater than 0.
//     *
//     * Returns:
//     *      raster(std::string):
//     *          raster plot
//     */
//    if (stop == -1) {
//        stop = record.size();
//    }
//
//    int range = stop - start;
//    int n_threshold = threshold * n_neuron;
//
//    REC_SIZE rec = sizeCheckRecord(stop, start);
//    start = rec.start;
//    stop = rec.stop;
//
//    // Size Check
//    if (!rec.check) {
//        std::cout << "Raster cannot be plotted!" << std::endl;
//        return ;
//    }
//
//    if ((record[0].available | 0b0111) != 0b1111) {
//        std::cout << "No firing record available!" << std::endl
//            << "Raster cannot be plotted!" << std::endl;
//        return ;
//    }
//
//    // File
//    std::string name = std::string(filename) + "_rasterID" +std::to_string(getID()) +".csv";
//    std::ofstream file(name, std::ofstream::out);
//
//    if (file.is_open()) {
//        // Header
//
//        file << "N_ID" << " , " << std::endl;
//
//        // Body
//        for (int i = 0; i < n_neuron; i++) {
//            file << i << " , ";
//            for (int j = start; j < stop; j++) {
//                if (record[j].flags[i]) {
//                    file << "x" << " , ";
//                }
//                else {
//                    file << " " << " , ";
//                }
//            }
//            file << " " << std::endl;
//        }
//        file << "TIME" << " , ";
//        for (int i = start; i < stop; i++) {
//            file << i << " , ";
//        }
//        file << std::endl;
//        file << "FIRE" << " , ";
//        for (int i = start; i < stop; i++) {
//            file << num_fire(record[i].flags) << " , ";
//        }
//        file << std::endl;
//        if (threshold > 0.0f) {
//            file << "IGNIT" << " , ";
//            for (int i = start; i < stop; i++) {
//                file << std::to_string(num_fire(record[i].flags) >= n_threshold) << " , ";
//            }
//        }
//    }
//    else {
//        std::cout << "Raster CSV file cannot open!" << std::endl;
//    }
//    file.close();
//
//}

void FLIF::getCSV(char* filename, int type, float threshold, int stop, int start) {

    if (stop == -1) {
        stop = record.size();
    }

    int range = stop - start;
    int n_threshold = threshold * n_neuron;
    REC_SIZE rec = sizeCheckRecord(stop, start);
    start = rec.start;
    stop = rec.stop;

    // Size Check
    if (!rec.check) {
        std::cout << "EF cannot be plotted!" << std::endl;
        return;
    }

    // File
    std::string specifier = "specifier";
    if (type == 0) {
        specifier = "_raster";
    }
    if (type == 1) {
        specifier = "_energy";
    }
    if (type == 2) {
        specifier = "_fatigue";
    }
    std::string name = std::string(filename) + specifier + ".csv";
    std::ofstream file(name, std::ofstream::out);

    if (file.is_open()) {
        // Header

        file << "N_ID" << " , " << std::endl;

        // Body
        for (int i = 0; i < n_neuron; i++) {
            file << i << " , ";
            for (int j = start; j < stop; j++) {
                if (type == 0) {
                    if (record[j].flags[i]) {
                        file << "x" << " , ";
                    }
                    else {
                        file << " " << " , ";
                    }
                }
                if (type == 1) {
                    file << record[j].energy[i] << " , ";
                }
                else if (type == 2) {
                    file << record[j].fatigue[i] << " , ";
                }
            }
            file << " " << std::endl;
        }
        file << "TIME" << " , ";
        for (int i = start; i < stop; i++) {
            file << i << " , ";
        }
        file << std::endl;
        if (type == 0) {
            file << "FIRE" << " , ";
            for (int i = start; i < stop; i++) {
                file << num_fire(record[i].flags) << " , ";
            }
            file << std::endl;
            if (threshold > 0.0f) {
                file << "IGNIT" << " , ";
                for (int i = start; i < stop; i++) {
                    file << std::to_string(num_fire(record[i].flags) >= n_threshold) << " , ";
                }
            }
        }
        
    }
    else {
        std::cout << "Type " << type << " file cannot open!" << std::endl;
    }
    file.close();

}

void FLIF::saveRecord(char* filename, float threshold, int stop, int start) {
    /* Save the record and the raster plot constructed by 
     * getRaster() and getRecord() recursively
     * to a .txt file, into ./test/ folder.
     *
     * Parameters:
     *      filename(char*):
     *          filename to be stamped then used
     *      start(int):
     *          starting timestep
     *      stop(int):
     *          ending timestep
     *      threshold(float):
     *          minimum rate of firing to ignit (0 by default)
     *          to show the ignit line, it must be greater than 0.
     */
    
    std::string raster_name = "./test/raster_" + dateTimeStamp(filename) + ".txt";
    std::ofstream raster_file(raster_name, std::ofstream::out);

    std::string record_name = "./test/record_" + dateTimeStamp(filename) + ".txt";
    std::ofstream record_file(record_name, std::ofstream::out);

    if (stop == -1) {
        stop = record.size();
    }

    REC_SIZE rec = sizeCheckRecord(stop, start);
    start = rec.start;
    stop = rec.stop;
    // Size Check
    if (!rec.check) {
        std::cout << "Record cannot be saved!" << std::endl;
        return;
    }

    if ((record[0].available | 0b0111) != 0b1111) {
        std::cout << "No firing record available!" << std::endl
            << "Raster cannot be plotted!" << std::endl;
    }
    else {
        if (raster_file.is_open()) {
            raster_file << getRaster(threshold, stop, start) << std::endl;
        }
        else {
            std::cout << "Raster file cannot open!" << std::endl;
        }
        raster_file.close();
    }

    if (record_file.is_open()) {
        for (int i = start; i < stop; i++) {
            record_file << getRecord(i) << std::endl;
        }
        record_file.close();
    }
    else {
        std::cout << "Record file cannot open!" << std::endl;
    }
}

void FLIF::getWeightCSV(char* filename, int stop, int start) {
    std::string name = std::string(filename) + "_weight.csv";
    std::vector <int> num;
    std::vector<int>::iterator it_num;
    int counter = 0;

    std::ofstream file(name, std::ofstream::out);
    std::vector<std::vector<float>>::iterator it;
    int outgoing;
    if (file.is_open()) {
        
        for (int i = start; i < stop; i++) {
            file << "t = " << i << std::endl;
            file << "N_ID / N_ID" << " , ";
            counter = 0;
            num.resize(record[i].weights[0].size());
            for (it_num = num.begin(); it_num < num.end(); it_num++) {
                *it_num = counter;
                counter++;
            }

            vectorToCSV<int>(file, num);
            outgoing = 0;
            for (it = record[i].weights.begin(); it < record[i].weights.end(); it++) {
                file << outgoing << " , ";
                vectorToCSV<float>(file, *it);
                outgoing++;
            }
            file << std::endl;
        }
    }
    else {
        std::cout << "Weights file cannot open!" << std::endl;
    }
    file.close();
}

void FLIF::saveCSV(char* filename, float threshold, int stop, int start) {

    if (stop == -1) {
        stop = record.size();
    }

    if ((record[0].available | 0b0111) == 0b1111) { // FLAGS 
        getCSV(filename, 0, threshold, stop, start);
    }

    if ((record[0].available | 0b1011) == 0b1111) { // ENERGY
        getCSV(filename, 1, threshold, stop, start);
    }

    if ((record[0].available | 0b1101) == 0b1111) { // FATIGUE
        getCSV(filename, 2, threshold, stop, start);
    }

    if ((record[0].available | 0b1110) == 0b1111) { // WEIGHT
        getWeightCSV(filename, stop, start);
    }
}

int FLIF::getID(){
    /* ID getter
     *
     * Returns:
     *      ID(int):
     *          ID of the network starting from 1
     */
    return ID;
}

int FLIF::getN(){
    /* n_neuron getter
     *
     * Returns:
     *      n_neuron(int):
     *          Number of neurons inside network
     */
    return n_neuron;
}

std::string FLIF::getInfo() {
    return "HI\n";
}

// Set
void FLIF::setActivity(float act) {
    this->activity = act;
}

