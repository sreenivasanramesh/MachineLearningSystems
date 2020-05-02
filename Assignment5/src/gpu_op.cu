#include "./c_runtime_api.h"
#include <cassert>
#include <cstdio>
#include <cublas_v2.h>
#include <cuda_runtime.h>

/* TODO: Your code here */
/* all your GPU kernel code, e.g. matrix_softmax_cross_entropy_kernel */

// y = inputs[0], y_ = inputs[1]
// np.mean(-np.sum(y_ * np.log(softmax(y)), axis=1), keepdims=True)
__global__ void matrix_softmax_cross_entropy_kernel(int nrow, int ncol,const float *input_a, const float *input_b, float *output) {
	// Dynamic shared memory, size provided at kernel launch.
	extern __shared__ float loss_per_row[];
	// Two dimensional thread blocks.
	int y = blockIdx.x * blockDim.x * blockDim.y + threadIdx.y * blockDim.x +
					threadIdx.x;
	if (y >= nrow) {
		return;
	}
	input_a += y * ncol;
	input_b += y * ncol;
	float maxval = *input_a;
	// Find max for a row.
	for (int x = 1; x < ncol; ++x) {
		maxval = max(maxval, input_a[x]);
	}
	// Deduct by max for a row, and raise to exp.
	float sum = 0;
	for (int x = 0; x < ncol; ++x) {
		sum += exp(input_a[x] - maxval);
	}
	// Compute per-row loss.
	float loss = 0;
	for (int x = 0; x < ncol; ++x) {
		loss -= input_b[x] * log(exp(input_a[x] - maxval) / sum);
	}
	loss_per_row[y] = loss;
	__syncthreads();
	// Compute reduce_mean across rows.
	float mean_loss = 0;
	// Use a single thread to reduce mean across rows.
	if ((threadIdx.x == 0) && (threadIdx.y == 0)) {
		for (int i = 0; i < nrow; ++i) {
			mean_loss += loss_per_row[i];
		}
		mean_loss /= nrow;
		output[0] = mean_loss;
	}
}

__global__ void ArraySetKernel(int len, float *arr, float value){
	int index = blockIdx.x * blockDim.x + threadIdx.x;
	if (index < len){
		arr[index] = value;
	}
}

int DLGpuArraySet(DLArrayHandle arr, float value) { 
	/* TODO: Your code here */
	int threads_per_block = 1024;
	int number_of_threads = 1;
	for (int i=0; i < arr->ndim; i++){
		number_of_threads = number_of_threads * arr->shape[i];
	}
	float *data = (float *)arr->data;
	dim3 threads, blocks;
	if (number_of_threads <= threads_per_block){
		threads.x = number_of_threads;
		blocks.x = 1;
	}
	else{
		threads.x = threads_per_block;
		blocks.x = (number_of_threads + threads_per_block -1)/threads_per_block;
	}
	ArraySetKernel<<<blocks, threads>>>(number_of_threads, data, value);
	return 0;
}

__global__ void BroadcastToKernel(int in_threads, int out_threads, const float *input, float *output){
	int index = blockIdx.x * blockDim.x + threadIdx.x;
	if(index < out_threads) {
		output[index] = input[index % in_threads];
	}
		
}

int DLGpuBroadcastTo(const DLArrayHandle input, DLArrayHandle output) {
	int in_threads = 1; 
	int out_threads = 1;
	int threads_per_block = 1024;
	for(int i = 0; i < input->ndim; i++){
		in_threads = in_threads * input->shape[i];
	}
	for(int i = 0; i < output->ndim; i++){
		out_threads = out_threads * output->shape[i];
	}
	const float *input_data = (const float*)input->data; 
	float *output_data = (float*)output->data;
	dim3 threads, blocks;
	if (in_threads <= threads_per_block){
		threads.x = in_threads;
		blocks.x = 1;
	}
	else{
		threads.x = threads_per_block;
		blocks.x = (out_threads + threads_per_block -1)/threads_per_block;
	}
	BroadcastToKernel<<<blocks, threads>>>(in_threads, out_threads, input_data, output_data);
	return 0;
}

__global__ void ReduceSumAxisZero(const float *input_data, float *output_data, int rows, int input){
	int index = blockIdx.x * blockDim.x + threadIdx.x;
	if (index < input){
		output_data[index] = 0;
		for (int i=0; i < rows; i++){
				output_data[index] += input_data[i * input + index];
			}
	}
}

int DLGpuReduceSumAxisZero(const DLArrayHandle input, DLArrayHandle output) {
	/* TODO: Your code here */
	int number_of_threads = 1;
	int threads_per_block = 1024;
	for (int i = 1; i < input->ndim; i++){
		number_of_threads = number_of_threads * input->shape[i];
	}
	dim3 threads, blocks;
	float *output_data = (float *)output->data;
	const float *input_data = (const float *)input->data;
	if (number_of_threads <= threads_per_block){
		threads.x = number_of_threads;
		blocks.x = 1;
	}
	else{
		threads.x = threads_per_block;
		blocks.x = (number_of_threads + threads_per_block - 1)/threads_per_block;
	}
	ReduceSumAxisZero<<<blocks, threads>>>(input_data, output_data, input->shape[0], number_of_threads);
	return 0;
}

__global__ void MatrixElementwiseAdd(int input, const float *input_a, const float *input_b, float *output_data){
	int index = blockIdx.x * blockDim.x + threadIdx.x;
	if (index < input){
		output_data[index] = input_a[index] + input_b[index];
	}
}

int DLGpuMatrixElementwiseAdd(const DLArrayHandle matA, const DLArrayHandle matB, DLArrayHandle output) {
	/* TODO: Your code here */
	int number_of_threads = 1;
	int threads_per_block = 1024;
	dim3 threads, blocks;
	for (int i = 0; i < matA->ndim; i++){
		number_of_threads = number_of_threads * matA->shape[i];
	}
	const float *input_a = (const float *)matA->data;
	const float *input_b = (const float *)matB->data;
	float *output_data = (float *)output->data;
	if (number_of_threads <= threads_per_block){
		threads.x = number_of_threads;
		blocks.x = 1;
	}
	else{
		threads.x = threads_per_block;
		blocks.x = (number_of_threads + threads_per_block - 1)/threads_per_block;
	}
	MatrixElementwiseAdd<<<blocks, threads>>>(number_of_threads, input_a, input_b, output_data);
	return 0;

}

__global__ void MatrixElementwiseAddByConst(int input, const float *input_a, float val, float *output_data){
	int index = blockIdx.x * blockDim.x + threadIdx.x;
	if (index < input){
		output_data[index] = input_a[index] + val;
	}
}

int DLGpuMatrixElementwiseAddByConst(const DLArrayHandle matA, float val, DLArrayHandle output) {
	/* TODO: Your code here */
	int number_of_threads = 1;
	int threads_per_block = 1024;
	dim3 threads, blocks;
	for (int i = 0; i < matA->ndim; i++){
		number_of_threads = number_of_threads * matA->shape[i];
	}
	const float *input_a = (const float *)matA->data;
	float *output_data = (float *)output->data;
	if (number_of_threads <= threads_per_block){
		threads.x = number_of_threads;
		blocks.x = 1;
	}
	else{
		threads.x = threads_per_block;
		blocks.x = (number_of_threads + threads_per_block - 1)/threads_per_block;
	}
	MatrixElementwiseAddByConst<<<blocks, threads>>>(number_of_threads, input_a, val, output_data);
	return 0;
}

__global__ void MatrixElementwiseMultiply(int input, const float *input_a, const float *input_b, float *output_data){
	int index = blockIdx.x * blockDim.x + threadIdx.x;
	if (index < input){
		output_data[index] = input_a[index] * input_b[index];
	}
}

int DLGpuMatrixElementwiseMultiply(const DLArrayHandle matA, const DLArrayHandle matB, DLArrayHandle output) {
	/* TODO: Your code here */
	int number_of_threads = 1;
	int threads_per_block = 1024;
	dim3 threads, blocks;
	for (int i = 0; i < matA->ndim; i++){
		number_of_threads = number_of_threads * matA->shape[i];
	}
	const float *input_a = (const float *)matA->data;
	const float *input_b = (const float *)matB->data;
	float *output_data = (float *)output->data;
	if (number_of_threads <= threads_per_block){
		threads.x = number_of_threads;
		blocks.x = 1;
	}
	else{
		threads.x = threads_per_block;
		blocks.x = (number_of_threads + threads_per_block - 1)/threads_per_block;
	}
	MatrixElementwiseMultiply<<<blocks, threads>>>(number_of_threads, input_a, input_b, output_data);
	return 0;
}

__global__ void MatrixElementwiseMultiplyConst(int input, const float *input_a, float val, float *output_data){
	int index = blockIdx.x * blockDim.x + threadIdx.x;
	if (index < input){
		output_data[index] = input_a[index] * val;
	}
}

int DLGpuMatrixMultiplyByConst(const DLArrayHandle matA, float val, DLArrayHandle matB) {
	/* TODO: Your code here */
	int number_of_threads = 1;
	int threads_per_block = 1024;
	dim3 threads, blocks;
	for (int i = 0; i < matA->ndim; i++){
		number_of_threads = number_of_threads * matA->shape[i];
	}
	const float *input_a = (const float *)matA->data;
	float *output_data = (float *)matB->data;
	if (number_of_threads <= threads_per_block){
		threads.x = number_of_threads;
		blocks.x = 1;
	}
	else{
		threads.x = threads_per_block;
		blocks.x = (number_of_threads + threads_per_block - 1)/threads_per_block;
	}
	MatrixElementwiseMultiplyConst<<<blocks, threads>>>(number_of_threads, input_a, val, output_data);
	return 0;
}

int DLGpuMatrixMultiply(const DLArrayHandle matA, bool transposeA, const DLArrayHandle matB, bool transposeB, DLArrayHandle matC) {
	/* TODO: Your code here */
	// Hint: use cublas
	// cublas assume matrix is column major
	cublasHandle_t handle;
	if (cublasCreate(&handle) != CUBLAS_STATUS_SUCCESS){
		return 0;
	}
	const float *matrix_A_data = (const float *)matA->data;
	const float *matrix_B_data = (const float *)matB->data;
	float *matrix_C_data = (float *)matC->data;
	int i = matC->shape[1];
	int j = matC->shape[0];
	int k = transposeA ? matA->shape[0] : matA->shape[1];
	float alpha = 1.0, beta = 0.0;
	cublasSgemm(handle, transposeB ? CUBLAS_OP_T : CUBLAS_OP_N,transposeA ? CUBLAS_OP_T : CUBLAS_OP_N, i, j, k, &alpha, matrix_B_data, transposeB ? k : i,matrix_A_data, transposeA ? j : k, &beta, matrix_C_data, i);
  return 0;
}

__global__ void Relu(int len, const float *input_data, float *output_data){
	int index = blockIdx.x * blockDim.x + threadIdx.x;
	if (index < len){
		if (input_data[index] > 0){
			output_data[index] = input_data[index];
		}
		else{
			output_data[index] = 0;
		}
	}
}

int DLGpuRelu(const DLArrayHandle input, DLArrayHandle output) {
	/* TODO: Your code here */
	int number_of_threads = 1;
	int threads_per_block = 1024;
	for (int i = 0; i < input->ndim; i++){
		number_of_threads = number_of_threads * input->shape[i];
	}
	const float *input_data = (const float *)input->data;
	float *output_data = (float *)output->data;
	dim3 blocks, threads;
	if (number_of_threads <= threads_per_block){
		threads.x = number_of_threads;
		blocks.x = 1;
	}
	else{
		threads.x = threads_per_block;
		blocks.x = (number_of_threads + threads_per_block - 1)/threads_per_block;
	}
	Relu<<<blocks, threads>>>(number_of_threads, input_data, output_data);
	return 0;
}

__global__ void ReluGradient(int len, const float *input_data, const float *input_gradient, float *output_data){
	int index = blockIdx.x * blockDim.x + threadIdx.x;
	if (index < len){
		if (input_data[index] > 0){
			output_data[index] = input_gradient[index];
		}
		else{
			output_data[index] = 0;
		}
	}
}

int DLGpuReluGradient(const DLArrayHandle input, const DLArrayHandle in_grad, DLArrayHandle output) {
	/* TODO: Your code here */
	
	int number_of_threads = 1;
	int threads_per_block = 1024;
	for (int i = 0; i < input->ndim; i++){
		number_of_threads = number_of_threads * input->shape[i];
	}
	const float *input_data = (const float *)input->data;
	const float *input_gradient = (const float *)in_grad->data;
	float *output_data = (float *)output->data;
	dim3 blocks, threads;
	if (number_of_threads <= threads_per_block){
		threads.x = number_of_threads;
		blocks.x = 1;
	}
	else{
		threads.x = threads_per_block;
		blocks.x = (number_of_threads + threads_per_block - 1)/threads_per_block;
	}
	ReluGradient<<<blocks, threads>>>(number_of_threads, input_data, input_gradient ,output_data);
	return 0;
}

__global__  void Softmax(const float *input, float *output, int r, int c) {
	int index = blockDim.x * blockDim.y * blockIdx.x + threadIdx.y * blockDim.x + threadIdx.x;

	if (index < r){
		input += index * c;
		output += index * c;
		float maximum = *input; 

		for (int i = 1; i < c; i++) {
			maximum = max(maximum, input[i]);
		}

		float sum = 0.0;
		for (int i = 0; i < c; i++) {
			sum += exp(input[i] - maximum);
		}

		for (int i = 0; i < c; i++) {
			output[i] = exp(input[i] - maximum) / sum;
		}
	}

	
}

int DLGpuSoftmax(const DLArrayHandle input, DLArrayHandle output) {
	/* TODO: Your code here */
	int r, c;
	dim3 threads;
	int threads_per_block = 1024;
	r = input->shape[0];
	c = input->shape[1];
	
	const float *input_data = (const float *)input->data;
	float *output_data = (float *)output->data;
	
	if (r <= threads_per_block) {
		threads.x = r;
	} 
	else{
		threads.x = threads_per_block;
		threads.y = (r + threads_per_block - 1) / threads_per_block;
	}

	Softmax<<<1, threads, r * sizeof(float)>>>(input_data, output_data, r, c);
	return 0;
}

int DLGpuSoftmaxCrossEntropy(const DLArrayHandle input_a, const DLArrayHandle input_b, DLArrayHandle output) {
	assert(input_a->ndim == 2);
	assert(input_b->ndim == 2);
	assert(output->ndim == 1);
	assert(input_a->shape[0] == input_b->shape[0] &&
				 input_a->shape[1] == input_b->shape[1]);
	int nrow = input_a->shape[0];
	// Maximum x- or y-dimension of a block = 1024
	// But we need 'nrow' shared memory, and max shared memory is 48KB.
	// Conservatively allow max 16KB shared memory.
	assert(nrow <= 1024 * 4);
	int ncol = input_a->shape[1];
	const float *input_data_a = (const float *)input_a->data;
	const float *input_data_b = (const float *)input_b->data;
	float *output_data = (float *)output->data;
	dim3 threads;
	if (nrow <= 1024) {
		threads.x = nrow;
	} else {
		threads.x = 1024;
		threads.y = (nrow + 1023) / 1024;
	}
	// 1 block, each block with 'threads' number of threads with 'nrow' shared
	// memory size
	matrix_softmax_cross_entropy_kernel<<<1, threads, nrow * sizeof(float)>>>(
			nrow, ncol, input_data_a, input_data_b, output_data);
	return 0;
}
