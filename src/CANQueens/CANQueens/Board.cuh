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

    void boardToCh();

public:
    Board(int n);
    ~Board();
    std::string toString();
    std::string toStringCh();
    std::string toStringEx();
    int* getChromosome();
    //CUDA_CALLABLE_MEMBER std::string toString();
};

#endif // BOARD_H