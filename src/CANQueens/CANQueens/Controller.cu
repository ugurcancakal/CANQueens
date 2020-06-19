/* Controller Class Source File
 *
 * 200613
 * author = @ugurc
 * ugurcan.cakal@gmail.com
 */

#include "Controller.cuh"

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

    //SADECE CONNECTIONLAR KALDI
    board->connect(memory);
    board->connect(explore);
    
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

void Controller::step() {
    // Show Board
    std::cout << board->toString(Board::PrintType::full) << std::endl;
    std::cout << board->toString(Board::PrintType::chrom) << std::endl;
    
    // Evaluate Board
    std::cout << value->fitness(chromosome) << std::endl;

    // Update Explore
    float act = value->activity(n_neuron, chromosome);
    explore->update(act);
    std::cout << explore->getActivity() << std::endl;

    // Update Memory
    memory->update();

    // Update Board
    board->update();
}

void Controller::runFor(int stepSize) {
    for (int i = 0; i < stepSize; i++) {
        step();
    }
}

Controller::~Controller() {
    //std::cout << "Controller destructed" << std::endl;
}

std::string Controller::toString() {
    return "Controller";
}