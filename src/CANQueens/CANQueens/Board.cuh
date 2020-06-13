/* Board Class Header File
 *
 * 200516
 * author = @ugurc
 * ugurcan.cakal@gmail.com
 */

#ifdef __CUDACC__
#define CUDA_CALLABLE_MEMBER __host__ __device__
#else
#define CUDA_CALLABLE_MEMBER
#endif 

#ifndef BOARD_H
#define BOARD_H

#include <string>
#include <iostream>
#include <math.h>

#include "CA.cuh"

class Board {
private:
    CA** board;
    int* chromosome;

    int row; // n
    int col; // log2n
    
    CA** initiateBoard(int row, int col);
    void deleteBoard(CA** board, int row, int col);

    int* initiateCh(int row);
    void deleteCh(int* chromosome);

    std::string fullBoard();
    std::string compressedBoard();
    std::string chromosomeDecimal();

    void boardToCh();

public:
    enum class PrintType { full, comp, chrom };
    Board(int n = 8);
    ~Board();
    std::string toString(PrintType type);
    int* getChromosome();
    void update();
    //CUDA_CALLABLE_MEMBER std::string toString();
};

#endif // BOARD_H