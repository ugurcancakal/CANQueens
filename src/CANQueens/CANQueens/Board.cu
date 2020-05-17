/* Board Class Source File
 *
 * 200517
 * author = @ugurc
 * ugurcan.cakal@gmail.com
 */

#include "Board.cuh"

void Board::initiateBoard() {
    if (!initiatedB) {
        board = new CA * [row];
        for (int i = 0; i < row; ++i) {
            board[i] = new CA[col];
        }
        initiatedB = true;
    }
    else {
        std::cout << "Board has already been initiated!" << std::endl;
    }
}

void Board::deleteBoard() {
    if (initiatedB) {
        for (int i = 0; i < row; ++i) {
            delete[] board[row];
        }
        delete[] board;
        initiatedB = false;
    }
    else {
        std::cout << "Board has not been created yet!" << std::endl;
    }
}

void Board::initiateCh() {
    if (!initiatedCh) {
        chromosome = new int[row];
        initiatedCh = true;
    }
    else {
        std::cout << "Chromosome has already been initiated!" << std::endl;
    }
}

void Board::deleteCh() {
    if (initiatedCh) {
        delete[] chromosome;
        initiatedCh = false;
    }
    else {
        std::cout << "Chromosome has not been created yet!" << std::endl;
    }
}

void Board::boardToCh() {
    int temp = 0;
    if (initiatedB && initiatedCh) {
        for (int i = 0; i < row; i++) {
            for (int j = 0; j < col; j++) {
                board[row][col];
                // Burade tempe kaydetme islemi
            }
            chromosome[row] = temp;
            temp = 0;
        }
    }
    else if (!initiatedB){
        std::cout << "Board has not been created yet!" << std::endl;
    }
    else if (!initiatedCh) {
        std::cout << "Chromosome has not been created yet!" << std::endl;
    }
    
}

int* Board::getChromosome() {
    return chromosome;
}

Board::Board() {
    std::cout << "Board constructed" << std::endl;
}

Board::~Board() {
    std::cout << "Board destructed" << std::endl;
}

std::string Board::toString() {
    return "Board";
    if (initiatedB) {
        for (int i = 0; i < row; i++) {
            for (int j = 0; j < col; j++) {
                board[row][col];
            }
        }
    }
    else {
        std::cout << "Board has not been created yet!" << std::endl;
    }
}