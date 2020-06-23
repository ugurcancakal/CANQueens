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

void startCounter(double& PCFreq, __int64& CounterStart);
double getCounter(double& PCFreq, __int64& CounterStart);
double experimentCPU(int repeat);
float experimentGPU(int repeat);

int main(int argc, char** argv) {
	/* Functions here are implemented to save the space in the main function.
	 * In order to play with the parameters, please go to the related funciton definition.
	 */
	srand(time(0));

	std::cout << "Welcome to CANQueens Project" << std::endl;
	
	// 202605 ONEMLI

	// 200619
	//Synapse::POC();
	// CA::POC_CPU();
	// CA::POC_GPU();
	// Explore::POC_CPU();
	// Explore::POC_GPU();
	// Memory::POC_CPU();
	// Memory::POC_GPU();
	// Board::POC_CPU();
	 //Board::POC_GPU();
	 //Controller::POC_CPU();
	 //Controller::POC_GPU();

	//experimentCPU(10);
	experimentGPU(10);
	
	// 201621 GPU
	// cudaDeviceReset must be called before exiting in order for profiling and
    // tracing tools such as Nsight and Visual Profiler to show complete traces.
	cudaError_t cudaStatus = cudaDeviceReset();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceReset failed!");
        return 1;
    }
}

void startCounter(double& PCFreq, __int64& CounterStart) {
	/* A function to measure CPU time during any operation.
	 * Call before the operation starts
	 *
	 * Parameters:
	 *      PCFreq(double &):
	 *          Referance to a double
	 *		CounterStart(__int64):
	 *          Referance to a __int64
	 */
	LARGE_INTEGER li;
	if (!QueryPerformanceFrequency(&li))
		std::cout << "QueryPerformanceFrequency failed!\n";

	PCFreq = double(li.QuadPart) / 1000.0;

	QueryPerformanceCounter(&li);
	CounterStart = li.QuadPart;
}
double getCounter(double& PCFreq, __int64& CounterStart) {
	/* A function to measure CPU time during any operation.
	 * Call after the operation starts
	 *
	 * Parameters:
	 *      PCFreq(double &):
	 *          Referance to a double
	 *		CounterStart(__int64):
	 *          Referance to a __int64
	 *
	 * Returns:
	 *		time_passed(double):
	 *			time passed from the time counter has been initiated by
	 *			startCounter() funciton call until getCounter() has been called.
	 */
	LARGE_INTEGER li;
	QueryPerformanceCounter(&li);
	return double(li.QuadPart - CounterStart) / PCFreq;
}

double experimentCPU(int repeat) {
	/* A black-box function to handle CPU timing experiment over "repeat" times
	 *
	 * - Load image
	 * - Start timer
	 * - Find minimum and maximum
	 * - Subract minimum pixel value from all pixels.
	 * - Normalize the image by multiplying all pixels by a factor
	 * - Stop timer
	 * - Accumulate the result
	 * - Go back to Start timer step if the operation not repeated "repeat" times
	 * - Write a new image
	 *
	 * Parameters:
	 *		repeat(int):
	 *			operation repetition amount
	 *      sFileName(char*):
	 *          Source image path
	 *		dFileName(char*):
	 *			destination image path
	 * Returns
	 *		average_time(double):
	 *			average time passed over all repetitions
	 *
	 */
	double PCFreq = 0.0;
	__int64 CounterStart = 0;
	double cpu_elapsed_time = 0.0;
	double average_time = 0.0;
	int exp = 0;
	int n = 8;
	int printCount = (int)repeat / 10.0;

	printf("\nTiming Experiment CPU");
	Controller CANQueen = Controller::getControllerCPU(n);
	while (exp <= repeat) {
		if (exp % printCount == 0) {
			printf("\nREPEAT: %d", exp);
		}

		startCounter(PCFreq, CounterStart);

		// Do the operations
		CANQueen.runFor_CPU(100);
		cpu_elapsed_time = getCounter(PCFreq, CounterStart);
		average_time += cpu_elapsed_time;
		exp++;
	}

	average_time /= exp;
	printf("\nAverage Time for 1 step: %f", average_time);
	// Cleaning
	return average_time;
}

float experimentGPU(int repeat) {
	/* A black-box function to handle GPU timing experiment over "repeat" times
	 *
	 * - Load image
	 * - Check the num_threads and blockWidth parameters
	 * - Set device pointers and do required memory allocation
	 * - Copy image from host to device
	 * - Start timer
	 * - Find minimum and maximum
	 * - Copy minimum and maximum back to host
	 * - Find optimum scaling parameters
	 * - Subract minimum pixel value from all pixels.
	 * - Normalize the image by multiplying all pixels by a factor
	 * - Stop timer
	 * - Copy image from device to host
	 * - Write a new image
	 *
	 * Parameters:
	 *		repeat(int):
	 *			operation repetition amount
	 *		num_threads(unsigned int):
	 *			Limited to 1024
	 *		blockWidth(unsigned int):
	 *			Limited to image width
	 *      sFileName(char*):
	 *          Source image path
	 *		dFileName(char*):
	 *			destination image path
	 * Returns
	 *		average_time(float):
	 *			average time passed over all repetitions
	 *
	 */

	float average_time = 0.0f;
	int exp = 0;
	int printCount = (int)repeat / 10.0;
	int n = 8;

	Controller CANQueen = Controller::getControllerGPU(n);
	printf("\nTiming Experiment GPU");
	printf("\nTotal Repetition: \t%d\n\n", repeat);

	// Do the operations
	//dim3 blockSize; // Limitted to 1024
	//dim3 gridSize;
	//dim3 gridSizeMinMax;
	//size_t s_memSize = num_threads * sizeof(Npp8u);
	//blockSize.x = blockWidth;
	//blockSize.y = num_threads / blockSize.x;
	//gridSize.x = width / blockSize.x;
	//gridSize.y = height / blockSize.y;
	//gridSizeMinMax.x = width / blockSize.x;
	//gridSizeMinMax.y = height / (blockSize.y * 2);

	//printf("\nthreads/block \t: %d", num_threads);
	//printf("\nblockSize.x \t: %d", blockSize.x);
	//printf("\nblockSize.y \t: %d", blockSize.y);
	//printf("\ngridSize.x \t: %d", gridSize.x);
	//printf("\ngridSize.y \t: %d\n", gridSize.y);

	// Timing variables
	float gpu_elapsed_time = 0.0;
	cudaEvent_t gpu_start, gpu_stop;

	while (exp <= repeat) {
		if (exp % printCount == 0) {
			printf("\nREPEAT: %d", exp);
		}
		cudaEventCreate(&gpu_start);
		cudaEventCreate(&gpu_stop);
		cudaEventRecord(gpu_start, 0);

		// Do the operations
		CANQueen.runFor_GPU(100);

		cudaThreadSynchronize();

		// Time record and copy data to host
		cudaEventRecord(gpu_stop, 0);
		cudaEventSynchronize(gpu_stop);
		cudaEventElapsedTime(&gpu_elapsed_time, gpu_start, gpu_stop);
		average_time += gpu_elapsed_time;
		cudaEventDestroy(gpu_start);
		cudaEventDestroy(gpu_stop);
		exp++;
	}

	average_time /= exp;
	printf("\nAverage Time for 1 step: %f", average_time);
	//Clean up
	//delete[] h_dst;

	// Device parameters
	//cudaFree(d_src);

	return average_time;
}

