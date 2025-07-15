#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>
#include <vector>
#include <stdexcept>

// Include shared device functions
#include "rinhash_device.cuh"
#include "argon2d_device.cuh"
#include "sha3-256.cu"
#include "blake3_device.cuh"


// External references to our CUDA implementations
extern "C" void blake3_hash(const uint8_t* input, size_t input_len, uint8_t* output);
extern "C" void argon2d_hash_rinhash(uint8_t* output, const uint8_t* input, size_t input_len);
extern "C" void sha3_256_hash(const uint8_t* input, size_t input_len, uint8_t* output);

// Modified kernel to use device functions
extern "C" __global__ void rinhash_cuda_kernel(
    const uint8_t* input, 
    size_t input_len, 
    uint8_t* output
) {
    // Intermediate results in shared memory
    __shared__ uint8_t blake3_out[32];
    __shared__ uint8_t argon2_out[32];
    // Only one thread should do this work
    if (threadIdx.x == 0) {
        // Step 1: BLAKE3 hash - now using light_hash_device
        light_hash_device(input, input_len, blake3_out);
        // Step 2: Argon2d hash
        uint32_t m_cost = 64000; // Example
        size_t memory_size = m_cost * sizeof(block);
        block* d_memory = (block*)malloc(memory_size);
        uint8_t salt[11] = { 'R','i','n','C','o','i','n','S','a','l','t' };
        device_argon2d_hash(argon2_out, blake3_out, 32, 2, 64000, 1, d_memory, salt, 11);
        
        // Step 3: SHA3-256 hash
        uint8_t sha3_out[32];
        sha3_256_device(argon2_out, 32, sha3_out);
        
    }
    
    // Use syncthreads to ensure all threads wait for the computation to complete
    __syncthreads();
}

// RinHash CUDA implementation
extern "C" void rinhash_cuda(const uint8_t* input, size_t input_len, uint8_t* output) {
    // Allocate device memory
    uint8_t *d_input = nullptr;
    uint8_t *d_output = nullptr;

    cudaError_t err;

    // Allocate memory on device
    err = cudaMalloc(&d_input, input_len);
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA error: Failed to allocate input memory: %s\n", cudaGetErrorString(err));
        return;
    }

    err = cudaMalloc(&d_output, 32);
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA error: Failed to allocate output memory: %s\n", cudaGetErrorString(err));
        cudaFree(d_input);
        return;
    }

    // Copy input data to device
    err = cudaMemcpy(d_input, input, input_len, cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA error: Failed to copy input to device: %s\n", cudaGetErrorString(err));
        cudaFree(d_input);
        cudaFree(d_output);
        return;
    }

    // Launch the kernel
    rinhash_cuda_kernel<<<1, 1>>>(d_input, input_len, d_output);

    // Wait for kernel to finish
    err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA error during kernel execution: %s\n", cudaGetErrorString(err));
        cudaFree(d_input);
        cudaFree(d_output);
        return;
    }

    // Copy result back to host
    err = cudaMemcpy(output, d_output, 32, cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA error: Failed to copy output from device: %s\n", cudaGetErrorString(err));
    }

    // Free device memory
    cudaFree(d_input);
    cudaFree(d_output);
}


// Helper function to convert a block header to bytes
extern "C" void blockheader_to_bytes(
    const uint32_t* version,
    const uint32_t* prev_block,
    const uint32_t* merkle_root,
    const uint32_t* timestamp,
    const uint32_t* bits,
    const uint32_t* nonce,
    uint8_t* output,
    size_t* output_len
) {
    size_t offset = 0;
    
    // Version (4 bytes)
    memcpy(output + offset, version, 4);
    offset += 4;
    
    // Previous block hash (32 bytes)
    memcpy(output + offset, prev_block, 32);
    offset += 32;
    
    // Merkle root (32 bytes)
    memcpy(output + offset, merkle_root, 32);
    offset += 32;
    
    // Timestamp (4 bytes)
    memcpy(output + offset, timestamp, 4);
    offset += 4;
    
    // Bits (4 bytes)
    memcpy(output + offset, bits, 4);
    offset += 4;
    
    // Nonce (4 bytes)
    memcpy(output + offset, nonce, 4);
    offset += 4;
    
    *output_len = offset;
}

// Batch processing version for mining
extern "C" void rinhash_cuda_batch(
    const uint8_t* block_headers,
    size_t block_header_len,
    uint8_t* outputs,
    uint32_t num_blocks
) {
    // Reset device to clear any previous errors
    cudaError_t err = cudaDeviceReset();
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA error: Failed to reset device: %s\n", 
                cudaGetErrorString(err));
        return;
    }
    
    
    // Check available memory
    size_t free_mem, total_mem;
    err = cudaMemGetInfo(&free_mem, &total_mem);
    if (err != cudaSuccess) {
        //fprintf(stderr, "CUDA error: Failed to get memory info: %s\n", 
        //        cudaGetErrorString(err));
        return;
    }
    
    size_t headers_size = num_blocks * block_header_len;
    size_t outputs_size = num_blocks * 32;
    size_t required_mem = headers_size + outputs_size;
    
    if (required_mem > free_mem) {
        fprintf(stderr, "CUDA error: Not enough memory (required: %zu, free: %zu)\n", 
                required_mem, free_mem);
        return;
    }
    
    // Allocate device memory
    uint8_t *d_headers = NULL;
    uint8_t *d_outputs = NULL;
    
    // Allocate memory for input block headers with error check
    err = cudaMalloc((void**)&d_headers, headers_size);
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA error: Failed to allocate device memory for headers (%zu bytes): %s\n", 
                headers_size, cudaGetErrorString(err));
        return;
    }
    
    // Allocate memory for output hashes with error check
    err = cudaMalloc((void**)&d_outputs, outputs_size);
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA error: Failed to allocate device memory for outputs (%zu bytes): %s\n",
                outputs_size, cudaGetErrorString(err));
        cudaFree(d_headers);
        return;
    }
    
    // Copy block headers from host to device
    err = cudaMemcpy(d_headers, block_headers, headers_size, cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA error: Failed to copy headers to device: %s\n",
                cudaGetErrorString(err));
        cudaFree(d_headers);
        cudaFree(d_outputs);
        return;
    }
    
    // Process one header at a time to isolate any issues
    for (uint32_t i = 0; i < num_blocks; i++) {
        const uint8_t* input = d_headers + i * block_header_len;
        uint8_t* output = d_outputs + i * 32;
        
        // Call rinhash_cuda_kernel with device pointers and proper launch configuration
        rinhash_cuda_kernel<<<1, 32>>>(input, block_header_len, output);
        
        // Check for errors after each processing
        err = cudaGetLastError();
        if (err != cudaSuccess) {
            fprintf(stderr, "CUDA error in block %u: %s\n", i, cudaGetErrorString(err));
            cudaFree(d_headers);
            cudaFree(d_outputs);
            return;
        }
    }
    
    // Synchronize device to ensure all operations are complete
    err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA error during synchronization: %s\n", cudaGetErrorString(err));
        cudaFree(d_headers);
        cudaFree(d_outputs);
        return;
    }
    
    // Copy results back from device to host
    err = cudaMemcpy(outputs, d_outputs, outputs_size, cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA error: Failed to copy results from device: %s\n",
                cudaGetErrorString(err));
    }
    
    // Free device memory
    cudaFree(d_headers);
    cudaFree(d_outputs);
}

// Main RinHash function that would be called from outside
extern "C" void RinHash(
    const uint32_t* version,
    const uint32_t* prev_block,
    const uint32_t* merkle_root,
    const uint32_t* timestamp,
    const uint32_t* bits,
    const uint32_t* nonce,
    uint8_t* output
) {
    uint8_t block_header[80]; // Standard block header size
    size_t block_header_len;
    
    // Convert block header to bytes
    blockheader_to_bytes(
        version,
        prev_block,
        merkle_root,
        timestamp,
        bits,
        nonce,
        block_header,
        &block_header_len
    );
    
    // Calculate RinHash
    rinhash_cuda(block_header, block_header_len, output);
}

// Mining function that tries different nonces
extern "C" void RinHash_mine(
    const uint32_t* version,
    const uint32_t* prev_block,
    const uint32_t* merkle_root,
    const uint32_t* timestamp,
    const uint32_t* bits,
    uint32_t start_nonce,
    uint32_t num_nonces,
    uint32_t* found_nonce,
    uint8_t* target_hash,
    uint8_t* best_hash
) {
    const size_t block_header_len = 80;
    std::vector<uint8_t> block_headers(block_header_len * num_nonces);
    std::vector<uint8_t> hashes(32 * num_nonces);
    
    // Prepare block headers with different nonces
    for (uint32_t i = 0; i < num_nonces; i++) {
        uint32_t current_nonce = start_nonce + i;
        
        // Fill in the common parts of the header
        uint8_t* header = block_headers.data() + i * block_header_len;
        size_t header_len;
        
        blockheader_to_bytes(
            version,
            prev_block,
            merkle_root,
            timestamp,
            bits,
            &current_nonce,
            header,
            &header_len
        );
    }
    
    // Calculate hashes for all nonces
    rinhash_cuda_batch(block_headers.data(), block_header_len, hashes.data(), num_nonces);
    
    // Find the best hash (lowest value)
    memcpy(best_hash, hashes.data(), 32);
    *found_nonce = start_nonce;
    
    for (uint32_t i = 1; i < num_nonces; i++) {
        uint8_t* current_hash = hashes.data() + i * 32;
        
        // Compare current hash with best hash (byte by byte, from most significant to least)
        bool is_better = false;
        for (int j = 0; j < 32; j++) {
            if (current_hash[j] < best_hash[j]) {
                is_better = true;
                break;
            }
            else if (current_hash[j] > best_hash[j]) {
                break;
            }
        }
        
        if (is_better) {
            memcpy(best_hash, current_hash, 32);
            *found_nonce = start_nonce + i;
        }
    }
}
