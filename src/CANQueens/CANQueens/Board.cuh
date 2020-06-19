/* Board Class Header File
 * Check the source file for detailed explanations
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
    // Data
    int row; // n
    int col; // log2n
    CA** board;
    int* chromosome;

    // Initiation - Destruction
    CA** initiateBoard(int row, int col);
    void deleteBoard(CA** board, int row);
    int* initiateCh(int row);
    void deleteCh(int* chromosome);

    // Printing
    std::string fullBoard();
    std::string compressedBoard();
    std::string chromosomeDecimal();

    // Representation
    void boardToCh();

public:
    // Constructors - Destructors
    Board(int n = 8);
    ~Board();

    // Printing
    enum class PrintType { full, comp, chrom };
    std::string toString(PrintType type);
    void connect(FLIF* pre_synaptic, float inhibitory = 0.0f, float strength = 1.0f, float rate = 1.0f);

    // Update
    void update();

    // GET
    int* getChromosome();
};

#endif // BOARD_H