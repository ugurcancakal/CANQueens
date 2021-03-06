/* Board Class Source File
 * A CA based chess board and its associated chromosome
 * An example of 8-queen binary CA board is as follows:
 * ---------
 * |       |
 * |     * |
 * |   *   |
 * |   * * |
 * | *     |
 * | *   * |
 * | * *   |
 * | * * * |
 * ---------
 * On a full board it looks like:
 * -------------------
 * | *               |
 * |   *             |
 * |     *           |
 * |       *         |
 * |         *       |
 * |           *     |
 * |             *   |
 * |               * |
 * -------------------
 * And the chromosome representation of it is like:
 * ---
 * |0|
 * |1|
 * |2|
 * |3|
 * |4|
 * |5|
 * |6|
 * |7|
 * ---
 * 200517
 * author = @ugurc
 * ugurcan.cakal@gmail.com
 */

#include "Board.cuh"

int Board::history = 0;
// PRIVATE MEMBERS
// Initiation - Destruction

CA*** Board::initiateBoard(int row, int col) {
    /* Initiate a CA board using dynamic memory.
     * !!!Requires memory REALLOCATION
     *
     * Parameters:
     *      row(int):
     *          number of rows in the CA board of interest
     *      col(int):
     *          number of columns in the CA board of interest
     *
     * Returns:
     *      board(CA**):
     *          A CA board in a 2D dynamic array structure
     */
    CA*** board;
    board = new CA** [row];
    for (int i = 0; i < row; ++i) {
        board[i] = new CA*[col];
        for (int j = 0; j < col; j++) {
            board[i][j] = new CA();
        }
    }
    return board;
}

void Board::deleteBoard(CA*** board, int row) {
    /* Reallocate the memory used for CA board 
     *
     * Parameters:
     *      board(CA**):
     *          The CA board of interest
     *      row(int):
     *          number of rows in the CA board of interest
     */
    for (int i = 0; i < row; ++i) {
        for (int j = 0; j < col; j++) {
            delete board[i][j];
        }
        delete[] board[i];
    }
    delete[] board;
}

int* Board::initiateCh(int row) {
    /* Initiate a chromosome using dynamic memory.
     * !!!Requires memory REALLOCATION
     *
     * Parameters:
     *      row(int):
     *          number of rows in the CA board of interest
     */
    int* chromosome = new int[row];
    return chromosome;
}

void Board::deleteCh(int* chromosome) {
    /* Reallocate the memory used the chromosome
     *
     * Parameters:
     *      chromosome(int*):
     *          The chromosome of interest
     */
    delete[] chromosome;
}

// Printing

std::string Board::fullBoard() {
    /* Prints the board in full size
     * Requires up to date chromosome
     * ! be sure that boardToCh() is called before 
     *
     * -------------------
     * | *               |
     * |   *             |
     * |     *           |
     * |       *         |
     * |         *       |
     * |           *     |
     * |             *   |
     * |               * |
     * -------------------
     *
     * Returns:
     *      board(std::string):
     *          Full size board in string format
     */
    std::string temp = "\n";
    temp += "  \t";
    for (int i = 0; i < row; i++) {
        temp += std::to_string(i) + "\t";
    }
    temp += " \n";
    temp += " " + std::string(8 * (row + 1), '-') + "\n";
    for (int i = 0; i < row; i++) {
        temp += std::to_string(i) + "|\t";
        for (int j = 0; j < row; j++) {
            if (chromosome[i] == j)
                temp += "*\t";
            else
                temp += "\t";
        }
        temp += "|\n";
        temp += " " + std::string(8 * (row + 1), '-') + "\n";
    }
    return temp;
}

std::string Board::compressedBoard() {
    /* Prints the board in compressed binary representation
     *
     * ---------
     * |       |
     * |     * |
     * |   *   |
     * |   * * |
     * | *     |
     * | *   * |
     * | * *   |
     * | * * * |
     * ---------
     *
     * Returns:
     *      board(std::string):
     *          Compressed board in string format
     */
    std::string temp = "\n";
    temp += "  \t";
    for (int i = 0; i < col; i++) {
        temp += std::to_string(i) + "\t";
    }
    temp += " \n";
    temp += " " + std::string(8 * (col + 1), '-') + "\n";
    for (int i = 0; i < row; i++) {
        temp += std::to_string(i) + "|\t";
        for (int j = 0; j < col; j++) {
            if (board[i][j]->getIgnition())
                temp += "*\t";
            else
                temp += "\t";
        }
        temp += "|\n";
        temp += " " + std::string(8 * (col + 1), '-') + "\n";
    }
    return temp;
}

std::string Board::chromosomeDecimal() {
    /* Binary representation of the compressed board
     * 
     * ---
     * |0|
     * |1|
     * |2|
     * |3|
     * |4|
     * |5|
     * |6|
     * |7|
     * ---
     *
     * Returns:
     *      chromosome(std::string):
     *          Chromosome in string format
     */
    std::string temp = "\n";
    for (int i = 0; i < row; i++) {
        temp += std::to_string(i) + "| " + std::to_string(chromosome[i]) + " |\n";
    }
    return temp;
}

// Representation
void Board::boardToCh() {
    /* Store the queen placement information on the board 
     * in a decimal chromosome
     */
    int temp = 0;
    for (int i = 0; i < row; i++) {
        for (int j = 0; j < col; j++) {
            if (board[i][j]->getIgnition())
                temp += static_cast<int>(powf(2, (col - j - 1)));
        }
        chromosome[i] = temp;
        temp = 0;
    }
}



// PUBLIC MEMBERS
// Constructors - Destructors

Board::Board(int n) {
    /* Constructor
     * Create a binary represented chess board having n rows and log2n columns
     * A classic chess board is represented as follows
     *
     * ---------
     * |       |
     * |     * |
     * |   *   |
     * |   * * |
     * | *     |
     * | *   * |
     * | * *   |
     * | * * * |
     * ---------
     *
     * Parameters:
     *      n(int): 8 by default
     *          number of rows in the CA board of interest
     */

    row = n; 
    col = n>1? static_cast<int>(ceilf(log2(n))) : 1;
    board = initiateBoard(n, col);
    chromosome = initiateCh(n);
    boardToCh();
}

Board::~Board() {
    /* Destructor
     * Does the required memory reallocations
     * for CA board and the chromosome
     */
    deleteBoard(board, row);
    deleteCh(chromosome);
    std::cout << "Board destructed" << std::endl;
}


// Printing
std::string Board::toString(PrintType type) {
    /* Prints the board in desired format
     * full - compressed - decimal chromosome
     *
     * full
     * -------------------
     * | *               |
     * |   *             |
     * |     *           |
     * |       *         |
     * |         *       |
     * |           *     |
     * |             *   |
     * |               * |
     * -------------------
     *
     * comp
     * ---------
     * |       |
     * |     * |
     * |   *   |
     * |   * * |
     * | *     |
     * | *   * |
     * | * *   |
     * | * * * |
     * ---------
     
     * chrom
     * ---
     * |0|
     * |1|
     * |2|
     * |3|
     * |4|
     * |5|
     * |6|
     * |7|
     * ---
     *
     * Returns:
     *      representation(std::string):
     *          Board or chromosome in string format
     */
    switch (type)
    {
    case PrintType::full:
        return fullBoard();
        break;
    case PrintType::comp:
        return compressedBoard();
        break;
    case PrintType::chrom:
        return chromosomeDecimal();
        break;
    default:
        return compressedBoard();
        break;
    }
}

void Board::connect_CPU(FLIF* pre_synaptic, float inhibitory, float strength, float rate) {
    for (int i = 0; i < row; i++) {
        for (int j = 0; j < col; j++) {
            if (0 == (rand() % static_cast<int>(floorf(1.0f / rate)))) {
                board[i][j]->connectIn(pre_synaptic, strength, inhibitory);
            }         
        }
    }
}

void Board::setMemory_CPU(Synapse* memory, float inhibitory, float strength) {
    for (int i = 0; i < row; i++) {
        for (int j = 0; j < col; j++) {
            memory->connectOut(board[i][j], strength, inhibitory);
        }
    }
}




void Board::connect_GPU(FLIF* pre_synaptic, float inhibitory, float strength, float rate) {
    for (int i = 0; i < row; i++) {
        for (int j = 0; j < col; j++) {
            if (0 == (rand() % static_cast<int>(floorf(1.0f / rate)))) {
                board[i][j]->connectIn(pre_synaptic, strength, inhibitory);
                board[i][j]->connectRestore_GPU();
            }
        }
    }
}

void Board::setMemory_GPU(Synapse* memory, float inhibitory, float strength) {
    for (int i = 0; i < row; i++) {
        for (int j = 0; j < col; j++) {
            memory->connectOut(board[i][j], strength, inhibitory);
            memory->connectRestore_GPU();
        }
    }
}

void Board::update_CPU() {
    /* Update each CA in the board.
     * Update policy is depended on each CA seperately.
     * Board is interested in resulting ignition flag only.
     */
    for (int i = 0; i < row; i++) {
        for (int j = 0; j < col; j++) {
            board[i][j]->update_CPU();  
        }
    }
    boardToCh(); 
}

void Board::runFor_CPU(int timeStep) {
    for (int t = 0; t < timeStep; t++) {
        for (int i = 0; i < row; i++) {
            for (int j = 0; j < col; j++) {
                board[i][j]->runFor_CPU(1);
            }
        }
        history++;
    }
    boardToCh();
    
}

void Board::update_GPU() {
    /* Update each CA in the board.
     * Update policy is depended on each CA seperately.
     * Board is interested in resulting ignition flag only.
     */
    for (int i = 0; i < row; i++) {
        for (int j = 0; j < col; j++) {
            board[i][j]->update_GPU();
        }
    }
    boardToCh();
}

void Board::runFor_GPU(int timeStep) {
    for (int t = 0; t < timeStep; t++) {
        for (int i = 0; i < row; i++) {
            for (int j = 0; j < col; j++) {
                board[i][j]->runFor_GPU(1);
            }
        }
        history++;
    }
    boardToCh();

}

// GET
int* Board::getChromosome() {
    /* Chromosome getter
     * 
     * Returns:
     *      board(int*):
     *          chromosome in 1D dynamic array structure
     */
    return chromosome;
}

std::string Board::getInfo() {
    return "HI\n";
}

std::string Board::getActivity(int stop, int start) {

    std::string temp = "\n";
    if (stop == -1) {
        stop = history;
    }
    for (int i = 0; i < row; i++) {
        for (int j = 0; j < col; j++) {
            temp+= board[i][j]->getActivity(stop, start);
        }
    }
    
    return temp;

}

void Board::saveCSV(char* filename, float threshold, int stop, int start) {
    std::string ca = std::string(filename)+"/CA";
    std::string temp;
    if (CreateDirectory(ca.c_str(), NULL) ||
        ERROR_ALREADY_EXISTS == GetLastError()) {
        for (int i = 0; i < row; i++) {
            for (int j = 0; j < col; j++) {
                temp = ca + "/CA" + std::to_string(board[i][j]->getID());
                if (CreateDirectory(temp.c_str(), NULL) ||
                    ERROR_ALREADY_EXISTS == GetLastError()) {
                    temp+= "/CA_ID_" + std::to_string(board[i][j]->getID());
                    board[i][j]->saveCSV(strdup(temp.c_str()), threshold, stop, start);
                }
                else {
                    std::cout << temp << " directory could not created!" << std::endl;
                }
            }
        }
    }
    else {
        std::cout << ca << " directory could not created!" << std::endl;
    }
}


void Board::POC_CPU() {
    int timeStep = 10;
    Board* board;
    board = new Board(8);

    board->runFor_CPU(timeStep);
    std::cout << board->getActivity() << std::endl;
}

void Board::POC_GPU() {
    int timeStep = 10;
    Board* board;
    board = new Board(8);
    board->initBoardGPU();
    board->runFor_GPU(timeStep);
    std::cout << board->getActivity() << std::endl;
}

void Board::initBoardGPU() {
    for (int i = 0; i < row; i++) {
        for (int j = 0; j < col; j++) {
            this->board[i][j]->initCADevice();
        }
    }
}
