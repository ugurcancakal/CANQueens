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

#include "CA.cuh"

class Board {
private:
    CA** board;
    bool initiatedB;
    
    int* chromosome;
    bool initiatedCh;

    int row; // n
    int col; // log2n
    
    void initiateBoard();
    void deleteBoard();

    void initiateCh();
    void deleteCh();

    void boardToCh();

public:
    Board();
    ~Board();
    std::string toString();
    int* getChromosome();
    //CUDA_CALLABLE_MEMBER std::string toString();
};

#endif // BOARD_H