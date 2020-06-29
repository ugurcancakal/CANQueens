/* Controller Class Source File
 *
 * 200613
 * author = @ugurc
 * ugurcan.cakal@gmail.com
 */

#include "Controller.cuh"

void Controller::runFor_CPU(int stepSize) {
    for (int i = 0; i < stepSize; i++) {
        step_CPU();
    }
}

void Controller::runFor_GPU(int stepSize) {
    for (int i = 0; i < stepSize; i++) {
        step_GPU();
    }
}

Controller::Controller(int n) {
    //std::cout << "Controller constructed" << std::endl
    //          << "Welcome to CANQueens Project" << std::endl;

    srand(time(NULL));

    n_neuron = n;
    board = new Board(n);
    value = new Value(n);
    chromosome = board -> getChromosome();
    explore = new Explore();
    memory = new Memory();

    
    //value.connect(explore);
    //memory.connect(board);
    
    //std::string temp = "\n";
    //for (int i = 0; i < n; i++) {
    //    temp += std::to_string(i) + "| " + std::to_string(chromosome[i]) + " |\n";
    //}
    //std::cout << temp << std::endl;

    
    //std::cout << myValue.toString() << std::endl;

    //std::cout << myBoard.toString(Board::PrintType::chrom) << std::endl;
    //std::cout << myBoard.toString(Board::PrintType::comp) << std::endl;
    //std::cout << myBoard.toString(Board::PrintType::full) << std::endl;
}

void Controller::step_CPU() {
    VAL temp;
    // Show Board
    //std::cout << board->toString(Board::PrintType::full) << std::endl;
    //std::cout << board->toString(Board::PrintType::chrom) << std::endl;
    std::vector<int> chVec(board->getChromosome(), board->getChromosome() + n_neuron);
    choromosomeRec.push_back(chVec);
    
    // Evaluate Board
    temp.fitness = value->fitness(chromosome);
    //std::cout << temp.fitness << std::endl;

    // Update Explore
    temp.activity = value->activity(n_neuron, chromosome);

    explore->setActivity(temp.activity);
    explore->runFor_CPU(1);
    //std::cout << explore->getActivity() << std::endl;

    // Update Memory
    memory->runFor_CPU(1);

    // Update Board
    board->runFor_CPU(1);
    //std::cout << board->getActivity() << std::endl;
    valueRec.push_back(temp);
}

void Controller::step_GPU() {
    VAL temp;
    // Show Board
    //std::cout << board->toString(Board::PrintType::full) << std::endl;
    //std::cout << board->toString(Board::PrintType::chrom) << std::endl;
    std::vector<int> chVec(board->getChromosome(), board->getChromosome() + n_neuron);
    choromosomeRec.push_back(chVec);

    // Evaluate Board
    temp.fitness = value->fitness(chromosome);
    //std::cout << temp.fitness << std::endl;

    // Update Explore
    temp.activity = value->activity(n_neuron, chromosome);

    explore->setActivity(temp.activity);
    explore->runFor_GPU(1);
    //std::cout << explore->getActivity() << std::endl;

    // Update Memory
    memory->runFor_GPU(1);

    // Update Board
    board->runFor_GPU(1);
    //std::cout << board->getActivity() << std::endl;
    valueRec.push_back(temp);
}



Controller::~Controller() {
    //std::cout << "Controller destructed" << std::endl;
}

std::string Controller::toString() {
    return "Controller";
}

std::string Controller::dateTimeStamp(const char* filename) {
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

void Controller::saveChCSV(char* filename, int stop, int start) {
    std::string name = std::string(filename) + "_raster.csv";
    std::ofstream file(name, std::ofstream::out);

    if (stop == -1) {
        stop = choromosomeRec.size();
    }
    if (file.is_open()) {
        // Header
        file << "ROW" << " , " << std::endl;
        std::cout << choromosomeRec.size() << std::endl;
        // Body
        for (int i = 0; i < n_neuron; i++) {
            file << i << " , ";
            for (int j = start; j < stop; j++) {
                file << choromosomeRec[j][i] << " , ";
            }
            file << " " << std::endl;
        }
        file << "TIME" << " , ";
        for (int i = start; i < stop; i++) {
            file << i << " , ";
        }
        file << std::endl << "Fitness" << " , ";
        for (int i = start; i < stop; i++) {
            file << valueRec[i].fitness << " , ";
        }

        file << std::endl << "Explore" << " , ";
        for (int i = start; i < stop; i++) {
            file << valueRec[i].activity << " , ";
        }
        file << std::endl;
    }
    else {
        std::cout << "Chromosome file cannot open!" << std::endl;
    }
    file.close();

}

void Controller::saveValueCSV(char* filename, int stop, int start) {
    if (stop == -1) {
        stop = choromosomeRec.size();
    }

    std::string name = std::string(filename) + "_fitness.csv";
    std::ofstream file(name, std::ofstream::out);
    if (file.is_open()) {
        file << "Time" << " , ";
        for (int i = start; i < stop; i++) {
            file << i << " , ";

        }
        file << std::endl << "Fitness" << " , ";
        for (int i = start; i < stop; i++) {
            file << valueRec[i].fitness << " , ";
        }

        file << std::endl << "Activity" << " , ";
        for (int i = start; i < stop; i++) {
            file << valueRec[i].activity << " , ";
        }
        file << std::endl;
    }
    else {
        std::cout << "Value file cannot open!" << std::endl;
    }
    file.close();
}

void Controller::saveInfo(char* filename) {
    std::string name = std::string(filename) + "_ugurc.txt";
    std::ofstream file(name, std::ofstream::out);

    if (file.is_open()) {
        file << board->getInfo();
        file << explore->getInfo();
        file << memory->getInfo();
        file << value->getInfo();
    }
    else {
        std::cout << "Info file cannot open!" << std::endl;
    }
    file.close();
}

void Controller::saveLog() {
    std::string base = "./log";
    std::string logPath = base+"/" + dateTimeStamp("log");
    std::string boardPath = logPath+"/board";
    std::string explorePath = logPath +"/explore";
    std::string memoryPath = logPath +"/memory";
    std::string valuePath = logPath + "/value";
    std::string temp;

    if (CreateDirectory(base.c_str(), NULL) ||
        ERROR_ALREADY_EXISTS == GetLastError()) {

        if (CreateDirectory(logPath.c_str(), NULL) ||
            ERROR_ALREADY_EXISTS == GetLastError()) {
            temp = logPath + "/info";
            saveInfo(strdup(temp.c_str()));

            if (CreateDirectory(boardPath.c_str(), NULL) ||
                ERROR_ALREADY_EXISTS == GetLastError()) {
                board->saveCSV(strdup(boardPath.c_str()));
                temp = boardPath + "/chromosome";
                saveChCSV(strdup(temp.c_str()));
            }
            else {
                std::cout << boardPath << " directory could not created!" << std::endl;
            }

            if (CreateDirectory(explorePath.c_str(), NULL) ||
                ERROR_ALREADY_EXISTS == GetLastError()) {
                temp = explorePath + "/explore_ID_" + std::to_string(explore->getID());
                explore->saveCSV(strdup(temp.c_str()));
            }
            else {
                std::cout << explorePath << " directory could not created!" << std::endl;
            }

            if (CreateDirectory(memoryPath.c_str(), NULL) ||
                ERROR_ALREADY_EXISTS == GetLastError()) {
                temp = memoryPath + "/memory_ID_" + std::to_string(memory->getID());
                memory->saveCSV(strdup(temp.c_str()));

            }
            else {
                std::cout << memoryPath << " directory could not created!" << std::endl;
            }

            if (CreateDirectory(valuePath.c_str(), NULL) ||
                ERROR_ALREADY_EXISTS == GetLastError()) {
                temp = valuePath + "/value";
                saveValueCSV(strdup(temp.c_str()));
            }
            else {
                std::cout << memoryPath << " directory could not created!" << std::endl;
            }

        }
        else {
            std::cout << logPath << " directory could not created!" << std::endl;
        }

    }
    else {
        std::cout << base << " directory could not created!" << std::endl;
    }
}


void Controller::POC_CPU() {
    int n = 8;
    Controller CANQueen = getControllerCPU(n);
    CANQueen.runFor_CPU(10);
    //CANQueen.saveLog();

}

void Controller::POC_GPU() {
    // INIT
    
    int n = 8;
    Controller CANQueen = getControllerGPU(n);
    CANQueen.runFor_GPU(10);
    CANQueen.saveLog();

}

Controller Controller::getControllerCPU(int n){
    Controller CANQueen(n);
    CANQueen.board->connect_CPU(CANQueen.memory);
    CANQueen.board->setMemory_CPU(CANQueen.memory);
    CANQueen.board->connect_CPU(CANQueen.explore);

    return CANQueen;
}

Controller Controller::getControllerGPU(int n) {
    Controller CANQueen(n);
    CANQueen.explore->initExploreGPU();
    CANQueen.memory->initMemoryGPU();
    CANQueen.board->initBoardGPU();
    CANQueen.board->connect_GPU(CANQueen.memory);
    CANQueen.board->setMemory_GPU(CANQueen.memory);
    CANQueen.board->connect_GPU(CANQueen.explore);
    return CANQueen;
}



