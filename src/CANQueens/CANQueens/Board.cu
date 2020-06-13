/* Board Class Source File
 *
 * 200517
 * author = @ugurc
 * ugurcan.cakal@gmail.com
 */

#include "Board.cuh"

Board::Board(int n) {
    row = n; // n
    col = n>1? ceil(log2(n)) : 1; // log2n
    board = initiateBoard(n, col);
    chromosome = initiateCh(n);
    boardToCh();
    //std::cout << "row: " << row << " col: " << col << std::endl;
}

Board::~Board() {
    deleteBoard(board, row, col);
    std::cout << "Board destructed" << std::endl;
}

CA** Board::initiateBoard(int row, int col) {
    CA** board;
    board = new CA * [row];
    for (int i = 0; i < row; ++i) {
        board[i] = new CA[col];
    }
    return board;
}

void Board::update() {
    for (int i = 0; i < row; i++) {
        for (int j = 0; j < col; j++) {
            board[i][j].update();
        }
    }
    boardToCh();
}

void Board::deleteBoard(CA** board, int row, int col) {
    for (int i = 0; i < row; ++i) {
        delete[] board[i];
    }
    delete[] board;
}

int* Board::initiateCh(int row) {
    int* chromosome = new int[row];
    return chromosome;
}

void Board::deleteCh(int* chromosome) {
    delete[] chromosome;
}

void Board::boardToCh() {
    int temp = 0;
    for (int i = 0; i < row; i++) {
        for (int j = 0; j < col; j++) {
            if (board[i][j].getIgnition())
                temp+=  pow(2, (col - j -1));
        }
        chromosome[i] = temp;
        //std::cout << temp << std::endl;
        temp = 0;
    }
}

int* Board::getChromosome() {
    return chromosome;
}

std::string Board::toString(PrintType type) {

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

std::string Board::compressedBoard() {
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
            if (board[i][j].getIgnition())
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
    std::string temp = "\n";
    for (int i = 0; i < row; i++) {
        temp += std::to_string(i) + "| " + std::to_string(chromosome[i]) +" |\n";
    }
    return temp;
}

std::string Board::fullBoard() {

    // call board to ch first
    std::string temp = "\n";
    temp += "  \t";
    for (int i = 0; i < row; i++) {
        temp += std::to_string(i) + "\t";
    }
    temp += " \n";
    temp += " " + std::string(8*(row+1), '-') + "\n";
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