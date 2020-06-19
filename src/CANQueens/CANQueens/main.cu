#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <iostream>
#include <ctime>
#include <cstdlib>

#include "CA.cuh"
#include "Controller.cuh"
#include "Board.cuh"
#include "Memory.cuh"
#include "Explore.cuh"
#include "Value.cuh"



int main(int argc, char** argv) {
	/* Functions here are implemented to save the space in the main function.
	 * In order to play with the parameters, please go to the related funciton definition.
	 */
	srand(time(0));

	std::cout << "Welcome to CANQueens Project" << std::endl;

	//for (int i = 0; i < 50; i++) {
	//	CA myCA;
	//	std::cout << myCA.getID() << std::endl;
	//	std::cout << myCA.getIgnition() << std::endl;
	//	std::cout << myCA.toString() << std::endl;
	//}


	//FLIF myFLIF;
	//std::cout << myFLIF.toString() << std::endl;

	//int n = 8;
	//int* chromosome;
	//Board myBoard(n);
	//CA** board = myBoard.getBoard();
	//std::cout << board[2][3].getID() << std::endl;
	//std::cout << myBoard.toString(Board::PrintType::chrom) << std::endl;
	//std::cout << myBoard.toString(Board::PrintType::comp) << std::endl;
	//std::cout << myBoard.toString(Board::PrintType::full) << std::endl;
	
	// 202605 ONEMLI
	// 3 bit olursa ama n = 8 olmazsa yasakli bitleri belirlemek gerek
	// Cok buyuk ceza verilebilir
	// Modulo falan olabilir
	//chromosome = myBoard.getChromosome();
	//std::string temp = "\n";
	//for (int i = 0; i < n; i++) {
	//	temp += std::to_string(i) + "| " + std::to_string(chromosome[i]) + " |\n";
	//}
	//std::cout << temp << std::endl;

	//Memory myMemory;
	//std::cout << myMemory.toString() << std::endl;
	//int k = 5;
	//Value myValue(k);
	//int* chromosome = new int[k];
	//chromosome[0] = 3;
	//chromosome[1] = 1;
	//chromosome[2] = 4;
	//chromosome[3] = 0;
	//chromosome[4] = 3;
	//Value myValue(n);
	//std::cout << myValue.toString() << std::endl;
	//myValue.update(chromosome);

	//Explore myExplore;
	//std::cout << myExplore.toString() << std::endl;

	//Chromosome myChromosome;
	//std::cout << myChromosome.toString() << std::endl;

	//200612

	/*for (int i = 0; i < 50; i++) {
		CA myCA;
		std::cout << myCA.getID() << std::endl;
		std::cout << myCA.getIgnition() << std::endl;
		std::cout << myCA.toString() << std::endl;
	}*/


	//int timeStep = 10;
	//CA myCA(10);

	//myCA.runFor(timeStep);
	//std::cout << myCA.getRaster() << std::endl;

	//myCA.runFor(5);
	//std::cout << myCA.getActivity() << std::endl;

	//myCA.saveRecord("rast");

	//int n = 8;

	//Controller myController(n);
	//myController.runFor(10);

	//int max = maxCollision(n);

	//std::cout << max << std::endl;

	/*int timeStep = 10;
	Explore myEXP;

	myEXP.runFor(timeStep);
	std::cout << myEXP.getActivity() << std::endl;

	myEXP.updateA(0.1);

	myEXP.runFor(timeStep);
	std::cout << myEXP.getActivity() << std::endl;*/

	/*int timeStep = 10;
	Memory myMEM;
	myMEM.runFor(timeStep);
	std::cout << myMEM.getActivity() << std::endl;*/

	//myCA1.connect(myCA2);

	// 200619
	//CA::POC();
	//Explore::POC();
	Memory::POC();

	return 0;
}



//cudaError_t addWithCuda(int *c, const int *a, const int *b, unsigned int size);
//
//__global__ void addKernel(int *c, const int *a, const int *b)
//{
//    int i = threadIdx.x;
//    c[i] = a[i] + b[i];
//}
//
//int main()
//{
//    const int arraySize = 5;
//    const int a[arraySize] = { 1, 2, 3, 4, 5 };
//    const int b[arraySize] = { 10, 20, 30, 40, 50 };
//    int c[arraySize] = { 0 };
//
//    // Add vectors in parallel.
//    cudaError_t cudaStatus = addWithCuda(c, a, b, arraySize);
//    if (cudaStatus != cudaSuccess) {
//        fprintf(stderr, "addWithCuda failed!");
//        return 1;
//    }
//
//    printf("{1,2,3,4,5} + {10,20,30,40,50} = {%d,%d,%d,%d,%d}\n",
//        c[0], c[1], c[2], c[3], c[4]);
//
//    // cudaDeviceReset must be called before exiting in order for profiling and
//    // tracing tools such as Nsight and Visual Profiler to show complete traces.
//    cudaStatus = cudaDeviceReset();
//    if (cudaStatus != cudaSuccess) {
//        fprintf(stderr, "cudaDeviceReset failed!");
//        return 1;
//    }
//
//    return 0;
//}
//
//// Helper function for using CUDA to add vectors in parallel.
//cudaError_t addWithCuda(int *c, const int *a, const int *b, unsigned int size)
//{
//    int *dev_a = 0;
//    int *dev_b = 0;
//    int *dev_c = 0;
//    cudaError_t cudaStatus;
//
//    // Choose which GPU to run on, change this on a multi-GPU system.
//    cudaStatus = cudaSetDevice(0);
//    if (cudaStatus != cudaSuccess) {
//        fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
//        goto Error;
//    }
//
//    // Allocate GPU buffers for three vectors (two input, one output)    .
//    cudaStatus = cudaMalloc((void**)&dev_c, size * sizeof(int));
//    if (cudaStatus != cudaSuccess) {
//        fprintf(stderr, "cudaMalloc failed!");
//        goto Error;
//    }
//
//    cudaStatus = cudaMalloc((void**)&dev_a, size * sizeof(int));
//    if (cudaStatus != cudaSuccess) {
//        fprintf(stderr, "cudaMalloc failed!");
//        goto Error;
//    }
//
//    cudaStatus = cudaMalloc((void**)&dev_b, size * sizeof(int));
//    if (cudaStatus != cudaSuccess) {
//        fprintf(stderr, "cudaMalloc failed!");
//        goto Error;
//    }
//
//    // Copy input vectors from host memory to GPU buffers.
//    cudaStatus = cudaMemcpy(dev_a, a, size * sizeof(int), cudaMemcpyHostToDevice);
//    if (cudaStatus != cudaSuccess) {
//        fprintf(stderr, "cudaMemcpy failed!");
//        goto Error;
//    }
//
//    cudaStatus = cudaMemcpy(dev_b, b, size * sizeof(int), cudaMemcpyHostToDevice);
//    if (cudaStatus != cudaSuccess) {
//        fprintf(stderr, "cudaMemcpy failed!");
//        goto Error;
//    }
//
//    // Launch a kernel on the GPU with one thread for each element.
//    addKernel<<<1, size>>>(dev_c, dev_a, dev_b);
//
//    // Check for any errors launching the kernel
//    cudaStatus = cudaGetLastError();
//    if (cudaStatus != cudaSuccess) {
//        fprintf(stderr, "addKernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
//        goto Error;
//    }
//    
//    // cudaDeviceSynchronize waits for the kernel to finish, and returns
//    // any errors encountered during the launch.
//    cudaStatus = cudaDeviceSynchronize();
//    if (cudaStatus != cudaSuccess) {
//        fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching addKernel!\n", cudaStatus);
//        goto Error;
//    }
//
//    // Copy output vector from GPU buffer to host memory.
//    cudaStatus = cudaMemcpy(c, dev_c, size * sizeof(int), cudaMemcpyDeviceToHost);
//    if (cudaStatus != cudaSuccess) {
//        fprintf(stderr, "cudaMemcpy failed!");
//        goto Error;
//    }
//
//Error:
//    cudaFree(dev_c);
//    cudaFree(dev_a);
//    cudaFree(dev_b);
//    
//    return cudaStatus;
//}
