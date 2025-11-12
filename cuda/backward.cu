#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>      
#include <cuda.h>
#include <cuda_runtime.h>
#include "../cpp/manual_att3ntion.h"
#include <tuple>
    
#ifndef TILE_I
#define TILE_I 8
#define TILE_J 8
#define TILE_K 8
#endif

#ifndef MAX_D_REG
#define MAX_D_REG 64    // max hidden dim the register arrays can hold; bump if needed
#endif

// --- Backward Pass ---

__global__ void gather_grad_Vq1_kernel_optimized(
    const float* __restrict__ gradY_slice,     // [N, D]
    const float* __restrict__ Vr_1_slice,      // [J, D]
    const float* __restrict__ Vs_1_slice,      // [K, D]
    const float* __restrict__ Ar_slice,        // [I, J, K]
    const float* __restrict__ As_slice,        // [I, J, K]
    float*       __restrict__ gradVq1_slice_out, // [I, D]
    int I, int J, int K, int D, int N_grad){
    // Grid of I*D threads, each computes one element of the output gradVq1_slice
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= (int64_t)I * D) return;

    int d = idx % D;
    int i_target = idx / D;

    float grad_accum = 0.0f;

    // --- 1. Contribution from Y_r path (dL/dY_r) ---
    // dL/dVq_1[i] += sum_{j} [ dL/dY_r[j,d] * sum_{k} ( Ar[i,j,k] * Vs_1[k,d] ) ]
    for (int j = 0; j < J; ++j) {
        if (j >= N_grad) continue;
        float dy_r = gradY_slice[(int64_t)j * D + d];
        if (dy_r == 0.0f) continue;

        float inner_sum_k = 0.0f;
        for (int k = 0; k < K; ++k) {
            int64_t ijk_idx = (int64_t)i_target * J * K + (int64_t)j * K + k;
            inner_sum_k += Ar_slice[ijk_idx] * Vs_1_slice[(int64_t)k * D + d];
        }
        grad_accum += dy_r * inner_sum_k;
    }

    // --- 2. Contribution from Y_s path (dL/dY_s) ---
    // dL/dVq_1[i] += sum_{k} [ dL/dY_s[k,d] * sum_{j} ( As[i,j,k] * Vr_1[j,d] ) ]
    for (int k = 0; k < K; ++k) {
        if (k >= N_grad) continue;
        float dy_s = gradY_slice[(int64_t)k * D + d];
        if (dy_s == 0.0f) continue;
        
        float inner_sum_j = 0.0f;
        for (int j = 0; j < J; ++j) {
            int64_t ijk_idx = (int64_t)i_target * J * K + (int64_t)j * K + k;
            inner_sum_j += As_slice[ijk_idx] * Vr_1_slice[(int64_t)j * D + d];
        }
        grad_accum += dy_s * inner_sum_j;
    }

    gradVq1_slice_out[idx] = grad_accum;}

__global__ void gather_grad_Vr1_kernel_optimized(
    const float* __restrict__ gradY_slice,       // [N, D]
    const float* __restrict__ Vq_1_slice,        // [I, D]
    const float* __restrict__ Vs_1_slice,        // [K, D]
    const float* __restrict__ Aq_slice,          // [I, J, K]
    const float* __restrict__ As_slice,          // [I, J, K]
    float*       __restrict__ gradVr1_slice_out,   // [J, D]
    int I, int J, int K, int D, int N_grad){
    // Grid of J*D threads
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= (int64_t)J * D) return;

    int d = idx % D;
    int j_target = idx / D;

    float grad_accum = 0.0f;

    // --- 1. Contribution from Y_q path (dL/dY_q) ---
    for (int i = 0; i < I; ++i) {
        if (i >= N_grad) continue;
        float dy_q = gradY_slice[(int64_t)i * D + d];
        if (dy_q == 0.0f) continue;
        
        float inner_sum_k = 0.0f;
        for (int k = 0; k < K; ++k) {
            int64_t ijk_idx = (int64_t)i * J * K + (int64_t)j_target * K + k;
            inner_sum_k += Aq_slice[ijk_idx] * Vs_1_slice[(int64_t)k * D + d];
        }
        grad_accum += dy_q * inner_sum_k;
    }

    // --- 2. Contribution from Y_s path (dL/dY_s) ---
    for (int k = 0; k < K; ++k) {
        if (k >= N_grad) continue;
        float dy_s = gradY_slice[(int64_t)k * D + d];
        if (dy_s == 0.0f) continue;
        
        float inner_sum_i = 0.0f;
        for (int i = 0; i < I; ++i) {
            int64_t ijk_idx = (int64_t)i * J * K + (int64_t)j_target * K + k;
            inner_sum_i += As_slice[ijk_idx] * Vq_1_slice[(int64_t)i * D + d];
        }
        grad_accum += dy_s * inner_sum_i;
    }

    gradVr1_slice_out[idx] = grad_accum;}

__global__ void gather_grad_Vs1_kernel_optimized(
    const float* __restrict__ gradY_slice,     // [N, D]
    const float* __restrict__ Vq_1_slice,      // [I, D]
    const float* __restrict__ Vr_1_slice,      // [J, D]
    const float* __restrict__ Aq_slice,        // [I, J, K]
    const float* __restrict__ Ar_slice,        // [I, J, K]
    float*       __restrict__ gradVs1_slice_out, // [K, D]
    int I, int J, int K, int D, int N_grad){
    // Grid of K*D threads
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= (int64_t)K * D) return;

    int d = idx % D;
    int k_target = idx / D;

    float grad_accum = 0.0f;

    // --- 1. Contribution from Y_q path (dL/dY_q) ---
    for (int i = 0; i < I; ++i) {
        if (i >= N_grad) continue;
        float dy_q = gradY_slice[(int64_t)i * D + d];
        if (dy_q == 0.0f) continue;

        float inner_sum_j = 0.0f;
        for (int j = 0; j < J; ++j) {
            int64_t ijk_idx = (int64_t)i * J * K + (int64_t)j * K + k_target;
            inner_sum_j += Aq_slice[ijk_idx] * Vr_1_slice[(int64_t)j * D + d];
        }
        grad_accum += dy_q * inner_sum_j;
    }

    // --- 2. Contribution from Y_r path (dL/dY_r) ---
    for (int j = 0; j < J; ++j) {
        if (j >= N_grad) continue;
        float dy_r = gradY_slice[(int64_t)j * D + d];
        if (dy_r == 0.0f) continue;

        float inner_sum_i = 0.0f;
        for (int i = 0; i < I; ++i) {
            int64_t ijk_idx = (int64_t)i * J * K + (int64_t)j * K + k_target;
            inner_sum_i += Ar_slice[ijk_idx] * Vq_1_slice[(int64_t)i * D + d];
        }
        grad_accum += dy_r * inner_sum_i;
    }

    gradVs1_slice_out[idx] = grad_accum;}

__global__ void scatter_grad_Vq2_kernel_optimized(
    const float* __restrict__ gradY_slice,
    const float* __restrict__ Aq_slice,
    const float* __restrict__ Ar_slice,
    const float* __restrict__ As_slice,
    const float* __restrict__ Vr_2_slice,
    const float* __restrict__ Vs_2_slice,
    float*       __restrict__ gradVq2_slice_out,
    int I, int J, int K, int D, int N_grad) {

    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= (int64_t)I * D) return;

    int d = idx % D;
    int i_target = idx / D;

    float grad_accum = 0.0f;

    // Term 1: from dL/dY_r'
    for (int j = 0; j < J; ++j) {
        if (j >= N_grad) continue;
        float dy_r = gradY_slice[(int64_t)j * D + d];
        if (dy_r == 0.0f) continue;

        for (int k = 0; k < K; ++k) {
            int64_t ijk_idx = (int64_t)i_target * J * K + (int64_t)j * K + k;
            grad_accum += dy_r * Aq_slice[ijk_idx] * As_slice[ijk_idx] * Vs_2_slice[(int64_t)k * D + d];
        }
    }

    // Term 2: from dL/dY_s'
    for (int k = 0; k < K; ++k) {
        if (k >= N_grad) continue;
        float dy_s = gradY_slice[(int64_t)k * D + d];
        if (dy_s == 0.0f) continue;
        
        for (int j = 0; j < J; ++j) {
            int64_t ijk_idx = (int64_t)i_target * J * K + (int64_t)j * K + k;
            grad_accum += dy_s * Aq_slice[ijk_idx] * Ar_slice[ijk_idx] * Vr_2_slice[(int64_t)j * D + d];
        }
    }

    gradVq2_slice_out[idx] = grad_accum;}

__global__ void scatter_grad_Vr2_kernel_optimized(
    const float* __restrict__ gradY_slice,
    const float* __restrict__ Aq_slice,
    const float* __restrict__ Ar_slice,
    const float* __restrict__ As_slice,
    const float* __restrict__ Vq_2_slice,
    const float* __restrict__ Vs_2_slice,
    float*       __restrict__ gradVr2_slice_out,
    int I, int J, int K, int D, int N_grad) {

    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= (int64_t)J * D) return;

    int d = idx % D;
    int j_target = idx / D;

    float grad_accum = 0.0f;

    // Term 1: from dL/dY_q'
    for (int i = 0; i < I; ++i) {
        if (i >= N_grad) continue;
        float dy_q = gradY_slice[(int64_t)i * D + d];
        if (dy_q == 0.0f) continue;

        for (int k = 0; k < K; ++k) {
            int64_t ijk_idx = (int64_t)i * J * K + (int64_t)j_target * K + k;
            grad_accum += dy_q * Ar_slice[ijk_idx] * As_slice[ijk_idx] * Vs_2_slice[(int64_t)k * D + d];
        }
    }

    // Term 2: from dL/dY_s'
    for (int k = 0; k < K; ++k) {
        if (k >= N_grad) continue;
        float dy_s = gradY_slice[(int64_t)k * D + d];
        if (dy_s == 0.0f) continue;
        
        for (int i = 0; i < I; ++i) {
            int64_t ijk_idx = (int64_t)i * J * K + (int64_t)j_target * K + k;
            grad_accum += dy_s * Aq_slice[ijk_idx] * Ar_slice[ijk_idx] * Vq_2_slice[(int64_t)i * D + d];
        }
    }

    gradVr2_slice_out[idx] = grad_accum;}

__global__ void scatter_grad_Vs2_kernel_optimized(
    const float* __restrict__ gradY_slice,
    const float* __restrict__ Aq_slice,
    const float* __restrict__ Ar_slice,
    const float* __restrict__ As_slice,
    const float* __restrict__ Vq_2_slice,
    const float* __restrict__ Vr_2_slice,
    float*       __restrict__ gradVs2_slice_out,
    int I, int J, int K, int D, int N_grad) {
    
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= (int64_t)K * D) return;

    int d = idx % D;
    int k_target = idx / D;

    float grad_accum = 0.0f;

    // Term 1: from dL/dY_q'
    for (int i = 0; i < I; ++i) {
        if (i >= N_grad) continue;
        float dy_q = gradY_slice[(int64_t)i * D + d];
        if (dy_q == 0.0f) continue;

        for (int j = 0; j < J; ++j) {
            int64_t ijk_idx = (int64_t)i * J * K + (int64_t)j * K + k_target;
            grad_accum += dy_q * Ar_slice[ijk_idx] * As_slice[ijk_idx] * Vr_2_slice[(int64_t)j * D + d];
        }
    }

    // Term 2: from dL/dY_r'
    for (int j = 0; j < J; ++j) {
        if (j >= N_grad) continue;
        float dy_r = gradY_slice[(int64_t)j * D + d];
        if (dy_r == 0.0f) continue;

        for (int i = 0; i < I; ++i) {
            int64_t ijk_idx = (int64_t)i * J * K + (int64_t)j * K + k_target;
            grad_accum += dy_r * Aq_slice[ijk_idx] * As_slice[ijk_idx] * Vq_2_slice[(int64_t)i * D + d];
        }
    }

    gradVs2_slice_out[idx] = grad_accum;}


__global__ void compute_A_slice_kernel(
    const float* __restrict__ Q_slice_global, // Input Q_slice [I,D] on GPU
    const float* __restrict__ R_slice_global, // Input R_slice [J,D] on GPU
    const float* __restrict__ S_slice_global, // Input S_slice [K,D] on GPU
    float*       __restrict__ A_out_global,   // Output A_slice [I,J,K] on GPU
    int I, int J, int K, int D,
    float scale
) {
    // Map 3D block and thread indices to I, J, K
    // This kernel is intended to be launched with a 3D grid/block structure
    // that covers all elements of A_slice [I,J,K].
    // E.g., GridDim could be ( (I+BlockDim.x-1)/BlockDim.x, (J+BlockDim.y-1)/BlockDim.y, (K+BlockDim.z-1)/BlockDim.z )
    // BlockDim could be (DIM_I_THREADS, DIM_J_THREADS, DIM_K_THREADS)
    
    int i_idx = blockIdx.x * blockDim.x + threadIdx.x;
    int j_idx = blockIdx.y * blockDim.y + threadIdx.y;
    int k_idx = blockIdx.z * blockDim.z + threadIdx.z;

    // Boundary check
    if (i_idx >= I || j_idx >= J || k_idx >= K) {
        return;
    }

    // Compute dot product Q_slice[i_idx,:] * R_slice[j_idx,:] * S_slice[k_idx,:]
    float dot_product = 0.0f;
    for (int d_loop = 0; d_loop < D; ++d_loop) {
        // Q_slice_global is [I, D], so access Q_slice_global[i_idx * D + d_loop]
        // R_slice_global is [J, D], so access R_slice_global[j_idx * D + d_loop]
        // S_slice_global is [K, D], so access S_slice_global[k_idx * D + d_loop]
        dot_product += Q_slice_global[i_idx * D + d_loop] * \
                       R_slice_global[j_idx * D + d_loop] * \
                       S_slice_global[k_idx * D + d_loop];
    }

    // Apply scale and store in A_out_global
    // A_out_global is [I,J,K]. Index is i_idx * (J*K) + j_idx * K + k_idx
    A_out_global[i_idx * J * K + j_idx * K + k_idx] = dot_product * scale;
}
// C++ Wrapper for compute_A_slice_kernel
torch::Tensor compute_A_slice_cuda_wrapper(
    const torch::Tensor& Q_slice_gpu, // Assumed to be on GPU, shape [I,D]
    const torch::Tensor& R_slice_gpu, // Assumed to be on GPU, shape [J,D]
    const torch::Tensor& S_slice_gpu, // Assumed to be on GPU, shape [K,D]
    float scale
) {
    TORCH_CHECK(Q_slice_gpu.is_cuda(), "Q_slice_gpu must be a CUDA tensor");
    TORCH_CHECK(R_slice_gpu.is_cuda(), "R_slice_gpu must be a CUDA tensor");
    TORCH_CHECK(S_slice_gpu.is_cuda(), "S_slice_gpu must be a CUDA tensor");

    TORCH_CHECK(Q_slice_gpu.dim() == 2, "Q_slice_gpu must be 2D");
    TORCH_CHECK(R_slice_gpu.dim() == 2, "R_slice_gpu must be 2D");
    TORCH_CHECK(S_slice_gpu.dim() == 2, "S_slice_gpu must be 2D");

    const int I = Q_slice_gpu.size(0);
    const int D_q = Q_slice_gpu.size(1);
    const int J = R_slice_gpu.size(0);
    const int D_r = R_slice_gpu.size(1);
    const int K = S_slice_gpu.size(0);
    const int D_s = S_slice_gpu.size(1);

    TORCH_CHECK(D_q == D_r && D_r == D_s, "Dimension D must match for Q, R, S slices");
    const int D = D_q;

    // Allocate output tensor A_slice_out_gpu [I,J,K] on GPU
    auto options = Q_slice_gpu.options(); // Inherit dtype and device
    torch::Tensor A_slice_out_gpu = torch::zeros({I, J, K}, options);

    // Define block dimensions (e.g., 8x8x8 = 512 threads, adjust as needed)
    // These are example values, can be tuned.
    // Max threads per block is 1024.
    // Block dimensions should ideally be multiples of warp size (32 for NVIDIA GPUs) for one dim.
    constexpr int BLOCK_DIM_I = 8;
    constexpr int BLOCK_DIM_J = 8;
    constexpr int BLOCK_DIM_K = 8; 
    
    dim3 blockDim(BLOCK_DIM_I, BLOCK_DIM_J, BLOCK_DIM_K);

    // Calculate grid dimensions
    dim3 gridDim(
        (I + BLOCK_DIM_I - 1) / BLOCK_DIM_I,
        (J + BLOCK_DIM_J - 1) / BLOCK_DIM_J,
        (K + BLOCK_DIM_K - 1) / BLOCK_DIM_K
    );
    
    // Ensure inputs are contiguous (important for direct pointer access in kernel)
    auto Q_cont = Q_slice_gpu.contiguous();
    auto R_cont = R_slice_gpu.contiguous();
    auto S_cont = S_slice_gpu.contiguous();

    // Launch Kernel
    compute_A_slice_kernel<<<gridDim, blockDim>>>(
        Q_cont.data_ptr<float>(),
        R_cont.data_ptr<float>(),
        S_cont.data_ptr<float>(),
        A_slice_out_gpu.data_ptr<float>(),
        I, J, K, D,
        scale
    );

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA error in compute_A_slice_cuda_wrapper: %s\n", cudaGetErrorString(err));
        // Optionally throw an exception
        // throw std::runtime_error(std::string("CUDA kernel launch failed in compute_A_slice_cuda_wrapper: ") + cudaGetErrorString(err));
    }
    // cudaDeviceSynchronize(); // For debugging, to ensure kernel finishes before wrapper returns

    return A_slice_out_gpu;
}
// Kernel to compute Aq_slice (softmax over j, k for each fixed i)
// Each block processes one 'i' plane.
__global__ void compute_Aq_slice_kernel(
    const float* __restrict__ A_slice_global, // Input A_slice [I,J,K] (global mem)
    float*       __restrict__ Aq_out_global,  // Output Aq_slice [I,J,K] (global mem)
    int I_dim, int J_dim, int K_dim // Use _dim to avoid conflict with loop vars
) {
    // Shared memory is now only for reduction, not for the entire data plane.
    extern __shared__ float s_reduction_pad[]; 
    
    // Current 'i' index this block is responsible for
    int i_current = blockIdx.x;
    if (i_current >= I_dim) {
        return;
    }

    const float* current_A_plane_global = A_slice_global + (int64_t)i_current * J_dim * K_dim;
    float* current_Aq_plane_global = Aq_out_global + (int64_t)i_current * J_dim * K_dim;

    int plane_size = J_dim * K_dim;
    int tid_in_block = threadIdx.x;
    int threads_in_block = blockDim.x;

    // --- Pass 1: Find max_val in the plane using a parallel reduction ---
    float thread_max_val = -FLT_MAX;
    for (int idx = tid_in_block; idx < plane_size; idx += threads_in_block) {
        thread_max_val = fmaxf(thread_max_val, current_A_plane_global[idx]);
    }
    s_reduction_pad[tid_in_block] = thread_max_val;
    __syncthreads();

    // Reduce to find the max value for the whole plane
    for (int offset = threads_in_block / 2; offset > 0; offset >>= 1) {
        if (tid_in_block < offset) {
            s_reduction_pad[tid_in_block] = fmaxf(s_reduction_pad[tid_in_block], s_reduction_pad[tid_in_block + offset]);
        }
        __syncthreads();
    }
    float plane_max_val = s_reduction_pad[0];
    __syncthreads(); // Ensure all threads see the correct max value

    // --- Pass 2: Compute sum_exp for the plane ---
    float thread_sum_exp = 0.0f;
    for (int idx = tid_in_block; idx < plane_size; idx += threads_in_block) {
        thread_sum_exp += expf(current_A_plane_global[idx] - plane_max_val);
    }
    s_reduction_pad[tid_in_block] = thread_sum_exp;
    __syncthreads();

    // Reduce to find the sum of exponents for the whole plane
    for (int offset = threads_in_block / 2; offset > 0; offset >>= 1) {
        if (tid_in_block < offset) {
            s_reduction_pad[tid_in_block] += s_reduction_pad[tid_in_block + offset];
        }
        __syncthreads();
    }
    float plane_sum_exp = s_reduction_pad[0];
    __syncthreads();

    // --- Pass 3: Compute softmax values and write to global memory ---
    if (plane_sum_exp == 0.0f) plane_sum_exp = 1e-20f; // Avoid division by zero
    for (int idx = tid_in_block; idx < plane_size; idx += threads_in_block) {
        current_Aq_plane_global[idx] = expf(current_A_plane_global[idx] - plane_max_val) / plane_sum_exp;
    }
}


torch::Tensor compute_Aq_slice_cuda_wrapper(
    const torch::Tensor& A_slice_gpu // Assumed to be on GPU, shape [I,J,K]
) {
    TORCH_CHECK(A_slice_gpu.is_cuda(), "A_slice_gpu must be a CUDA tensor");
    TORCH_CHECK(A_slice_gpu.dim() == 3, "A_slice_gpu must be 3D [I,J,K]");

    const int I = A_slice_gpu.size(0);
    const int J = A_slice_gpu.size(1);
    const int K = A_slice_gpu.size(2);

    auto options = A_slice_gpu.options();
    torch::Tensor Aq_slice_out_gpu = torch::zeros({I, J, K}, options);

    dim3 gridDim(I);
    int threads_per_block = 256;
    dim3 blockDim(threads_per_block);

    // Shared memory size is now fixed and small, proportional to block size
    size_t shared_mem_size = threads_per_block * sizeof(float);
    
    // The explicit call to cudaFuncSetAttribute is no longer needed, as the requested
    // shared memory size is small and constant. Passing it in the kernel launch is sufficient.

    auto A_cont = A_slice_gpu.contiguous();

    // Launch Kernel
    compute_Aq_slice_kernel<<<gridDim, blockDim, shared_mem_size>>>(
        A_cont.data_ptr<float>(),
        Aq_slice_out_gpu.data_ptr<float>(),
        I, J, K
    );

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA error in compute_Aq_slice_cuda_wrapper: %s\n", cudaGetErrorString(err));
    }
    return Aq_slice_out_gpu;
}

// Kernel to compute Ar_slice (softmax over i, k for each fixed j)
// Each block processes one 'j' plane.
__global__ void compute_Ar_slice_kernel(
    const float* __restrict__ A_slice_global, // Input A_slice [I,J,K] (global mem)
    float*       __restrict__ Ar_out_global,  // Output Ar_slice [I,J,K] (global mem)
    int I_dim, int J_dim, int K_dim
) {
    extern __shared__ float s_reduction_pad[]; 
    
    int j_current = blockIdx.x;
    if (j_current >= J_dim) return;

    int plane_size = I_dim * K_dim;
    int tid_in_block = threadIdx.x;
    int threads_in_block = blockDim.x;

    // --- Pass 1: Find max_val ---
    float thread_max_val = -FLT_MAX;
    for (int idx = tid_in_block; idx < plane_size; idx += threads_in_block) {
        int i_load = idx / K_dim;
        int k_load = idx % K_dim;
        int64_t global_idx = (int64_t)i_load * J_dim * K_dim + (int64_t)j_current * K_dim + k_load;
        thread_max_val = fmaxf(thread_max_val, A_slice_global[global_idx]);
    }
    s_reduction_pad[tid_in_block] = thread_max_val;
    __syncthreads();
    for (int offset = threads_in_block / 2; offset > 0; offset >>= 1) {
        if (tid_in_block < offset) s_reduction_pad[tid_in_block] = fmaxf(s_reduction_pad[tid_in_block], s_reduction_pad[tid_in_block + offset]);
        __syncthreads();
    }
    float plane_max_val = s_reduction_pad[0];
    __syncthreads();

    // --- Pass 2: Compute sum_exp ---
    float thread_sum_exp = 0.0f;
    for (int idx = tid_in_block; idx < plane_size; idx += threads_in_block) {
        int i_load = idx / K_dim;
        int k_load = idx % K_dim;
        int64_t global_idx = (int64_t)i_load * J_dim * K_dim + (int64_t)j_current * K_dim + k_load;
        thread_sum_exp += expf(A_slice_global[global_idx] - plane_max_val);
    }
    s_reduction_pad[tid_in_block] = thread_sum_exp;
    __syncthreads();
    for (int offset = threads_in_block / 2; offset > 0; offset >>= 1) {
        if (tid_in_block < offset) s_reduction_pad[tid_in_block] += s_reduction_pad[tid_in_block + offset];
        __syncthreads();
    }
    float plane_sum_exp = s_reduction_pad[0];
    __syncthreads();

    // --- Pass 3: Compute softmax and write ---
    if (plane_sum_exp == 0.0f) plane_sum_exp = 1e-20f;
    for (int idx = tid_in_block; idx < plane_size; idx += threads_in_block) {
        int i_write = idx / K_dim;
        int k_write = idx % K_dim;
        int64_t global_idx = (int64_t)i_write * J_dim * K_dim + (int64_t)j_current * K_dim + k_write;
        Ar_out_global[global_idx] = expf(A_slice_global[global_idx] - plane_max_val) / plane_sum_exp;
    }
}


// Kernel to compute As_slice (softmax over i, j for each fixed k)
// Each block processes one 'k' plane.
__global__ void compute_As_slice_kernel(
    const float* __restrict__ A_slice_global, // Input A_slice [I,J,K] (global mem)
    float*       __restrict__ As_out_global,  // Output As_slice [I,J,K] (global mem)
    int I_dim, int J_dim, int K_dim
) {
    extern __shared__ float s_reduction_pad[];

    int k_current = blockIdx.x;
    if (k_current >= K_dim) return;

    int plane_size = I_dim * J_dim;
    int tid_in_block = threadIdx.x;
    int threads_in_block = blockDim.x;

    // --- Pass 1: Find max_val ---
    float thread_max_val = -FLT_MAX;
    for (int idx = tid_in_block; idx < plane_size; idx += threads_in_block) {
        int i_load = idx / J_dim;
        int j_load = idx % J_dim;
        int64_t global_idx = (int64_t)i_load * J_dim * K_dim + (int64_t)j_load * K_dim + k_current;
        thread_max_val = fmaxf(thread_max_val, A_slice_global[global_idx]);
    }
    s_reduction_pad[tid_in_block] = thread_max_val;
    __syncthreads();
    for (int offset = threads_in_block / 2; offset > 0; offset >>= 1) {
        if (tid_in_block < offset) s_reduction_pad[tid_in_block] = fmaxf(s_reduction_pad[tid_in_block], s_reduction_pad[tid_in_block + offset]);
        __syncthreads();
    }
    float plane_max_val = s_reduction_pad[0];
    __syncthreads();

    // --- Pass 2: Compute sum_exp ---
    float thread_sum_exp = 0.0f;
    for (int idx = tid_in_block; idx < plane_size; idx += threads_in_block) {
        int i_load = idx / J_dim;
        int j_load = idx % J_dim;
        int64_t global_idx = (int64_t)i_load * J_dim * K_dim + (int64_t)j_load * K_dim + k_current;
        thread_sum_exp += expf(A_slice_global[global_idx] - plane_max_val);
    }
    s_reduction_pad[tid_in_block] = thread_sum_exp;
    __syncthreads();
    for (int offset = threads_in_block / 2; offset > 0; offset >>= 1) {
        if (tid_in_block < offset) s_reduction_pad[tid_in_block] += s_reduction_pad[tid_in_block + offset];
        __syncthreads();
    }
    float plane_sum_exp = s_reduction_pad[0];
    __syncthreads();

    // --- Pass 3: Compute softmax and write ---
    if (plane_sum_exp == 0.0f) plane_sum_exp = 1e-20f;
    for (int idx = tid_in_block; idx < plane_size; idx += threads_in_block) {
        int i_write = idx / J_dim;
        int j_write = idx % J_dim;
        int64_t global_idx = (int64_t)i_write * J_dim * K_dim + (int64_t)j_write * K_dim + k_current;
        As_out_global[global_idx] = expf(A_slice_global[global_idx] - plane_max_val) / plane_sum_exp;
    }
}

// C++ Wrapper for compute_Ar_slice_kernel
torch::Tensor compute_Ar_slice_cuda_wrapper(
    const torch::Tensor& A_slice_gpu // Assumed to be on GPU, shape [I,J,K]
) {
    TORCH_CHECK(A_slice_gpu.is_cuda(), "A_slice_gpu must be a CUDA tensor");
    TORCH_CHECK(A_slice_gpu.dim() == 3, "A_slice_gpu must be 3D [I,J,K]");

    const int I = A_slice_gpu.size(0);
    const int J = A_slice_gpu.size(1);
    const int K = A_slice_gpu.size(2);

    auto options = A_slice_gpu.options(); 
    torch::Tensor Ar_slice_out_gpu = torch::zeros({I, J, K}, options);

    dim3 gridDim(J); // J blocks, one for each j-plane

    int threads_per_block = 256;
    dim3 blockDim(threads_per_block); 

    size_t shared_mem_size = threads_per_block * sizeof(float);
    
    auto A_cont = A_slice_gpu.contiguous();

    compute_Ar_slice_kernel<<<gridDim, blockDim, shared_mem_size>>>(
        A_cont.data_ptr<float>(),
        Ar_slice_out_gpu.data_ptr<float>(),
        I, J, K
    );

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA error in compute_Ar_slice_cuda_wrapper: %s\n", cudaGetErrorString(err));
    }
    return Ar_slice_out_gpu;
}


// C++ Wrapper for compute_As_slice_kernel
torch::Tensor compute_As_slice_cuda_wrapper(
    const torch::Tensor& A_slice_gpu // Assumed to be on GPU, shape [I,J,K]
) {
    TORCH_CHECK(A_slice_gpu.is_cuda(), "A_slice_gpu must be a CUDA tensor");
    TORCH_CHECK(A_slice_gpu.dim() == 3, "A_slice_gpu must be 3D [I,J,K]");

    const int I = A_slice_gpu.size(0);
    const int J = A_slice_gpu.size(1);
    const int K = A_slice_gpu.size(2);

    auto options = A_slice_gpu.options(); 
    torch::Tensor As_slice_out_gpu = torch::zeros({I, J, K}, options);

    dim3 gridDim(K); // K blocks, one for each k-plane

    int threads_per_block = 256;
    dim3 blockDim(threads_per_block); 

    size_t shared_mem_size = threads_per_block * sizeof(float);
    
    auto A_cont = A_slice_gpu.contiguous();

    compute_As_slice_kernel<<<gridDim, blockDim, shared_mem_size>>>(
        A_cont.data_ptr<float>(),
        As_slice_out_gpu.data_ptr<float>(),
        I, J, K
    );

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA error in compute_As_slice_cuda_wrapper: %s\n", cudaGetErrorString(err));
    }
    return As_slice_out_gpu;
}

// Kernel to compute grad_Aq_slice, grad_Ar_slice, grad_As_slice (Phase 1 of C++ compute_grad_A_single)
__global__ void compute_interim_grads_kernel(
    // Inputs
    const float* __restrict__ grad_output_slice, // [N, D]
    const float* __restrict__ Vq_1_slice,        // [I, D]
    const float* __restrict__ Vq_2_slice,        // [I, D]
    const float* __restrict__ Vr_1_slice,        // [J, D]
    const float* __restrict__ Vr_2_slice,        // [J, D]
    const float* __restrict__ Vs_1_slice,        // [K, D]
    const float* __restrict__ Vs_2_slice,        // [K, D]
    const float* __restrict__ Aq_slice,          // [I, J, K]
    const float* __restrict__ Ar_slice,          // [I, J, K]
    const float* __restrict__ As_slice,          // [I, J, K]
    // Outputs (Temporary)
    float* __restrict__ grad_Aq_slice_out,     // [I, J, K]
    float* __restrict__ grad_Ar_slice_out,     // [I, J, K]
    float* __restrict__ grad_As_slice_out,     // [I, J, K]
    // Dimensions
    int I_dim, int J_dim, int K_dim, int D_dim, int N_dim // Use _dim suffix
) {
    // Map 3D thread indices to (i, j, k)
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    int k = blockIdx.z * blockDim.z + threadIdx.z;

    // Boundary check
    if (i >= I_dim || j >= J_dim || k >= K_dim) {
        return;
    }
    
    // Calculate linear index for output arrays [I,J,K]
    int64_t ijk_idx = (int64_t)i * J_dim * K_dim + (int64_t)j * K_dim + k;

    float grad_aq_val = 0.0f;
    float grad_ar_val = 0.0f;
    float grad_as_val = 0.0f;

    // Get pointers to relevant rows in V matrices (based on i, j, k)
    const float* vq1_row_i = Vq_1_slice + (int64_t)i * D_dim;
    const float* vq2_row_i = Vq_2_slice + (int64_t)i * D_dim;
    const float* vr1_row_j = Vr_1_slice + (int64_t)j * D_dim;
    const float* vr2_row_j = Vr_2_slice + (int64_t)j * D_dim;
    const float* vs1_row_k = Vs_1_slice + (int64_t)k * D_dim;
    const float* vs2_row_k = Vs_2_slice + (int64_t)k * D_dim;
    
    // Pre-fetch Aq, Ar, As values for this (i,j,k)
    float aq_ijk = Aq_slice[ijk_idx];
    float ar_ijk = Ar_slice[ijk_idx];
    float as_ijk = As_slice[ijk_idx];


    // --- Term 1.a: grad_Aq from Yq (gather) ---
    if (i < N_dim) {
        const float* grad_out_row_i = grad_output_slice + (int64_t)i * D_dim;
        for (int d = 0; d < D_dim; ++d) {
            grad_aq_val += grad_out_row_i[d] * vr1_row_j[d] * vs1_row_k[d];
        }
    }

    // --- Term 1.b: grad_Aq from Yr' (scatter) ---
    if (j < N_dim) {
        float grad_x_term_j = 0.0f;
        const float* grad_out_row_j = grad_output_slice + (int64_t)j * D_dim;
        for (int d = 0; d < D_dim; ++d) {
            grad_x_term_j += grad_out_row_j[d] * vq2_row_i[d] * vs2_row_k[d];
        }
        grad_aq_val += grad_x_term_j * as_ijk; // Multiply by As[i,j,k]
    }

    // --- Term 1.c: grad_Aq from Ys' (scatter) ---
    if (k < N_dim) {
        float grad_x_term_k = 0.0f;
        const float* grad_out_row_k = grad_output_slice + (int64_t)k * D_dim;
        for (int d = 0; d < D_dim; ++d) {
            grad_x_term_k += grad_out_row_k[d] * vq2_row_i[d] * vr2_row_j[d];
        }
        grad_aq_val += grad_x_term_k * ar_ijk; // Multiply by Ar[i,j,k]
    }

    // --- Term 2.a: grad_Ar from Yr (gather) ---
     if (j < N_dim) {
        const float* grad_out_row_j = grad_output_slice + (int64_t)j * D_dim;
        for (int d = 0; d < D_dim; ++d) {
            grad_ar_val += grad_out_row_j[d] * vq1_row_i[d] * vs1_row_k[d];
        }
    }

    // --- Term 2.b: grad_Ar from Yq' (scatter) ---
     if (i < N_dim) {
        float grad_x_term_i = 0.0f;
        const float* grad_out_row_i = grad_output_slice + (int64_t)i * D_dim;
        for (int d = 0; d < D_dim; ++d) {
            grad_x_term_i += grad_out_row_i[d] * vr2_row_j[d] * vs2_row_k[d];
        }
        grad_ar_val += grad_x_term_i * as_ijk; // Multiply by As[i,j,k]
    }
   
    // --- Term 2.c: grad_Ar from Ys' (scatter) ---
    if (k < N_dim) {
        float grad_x_term_k = 0.0f;
        const float* grad_out_row_k = grad_output_slice + (int64_t)k * D_dim;
         for (int d = 0; d < D_dim; ++d) {
             grad_x_term_k += grad_out_row_k[d] * vq2_row_i[d] * vr2_row_j[d];
         }
        grad_ar_val += grad_x_term_k * aq_ijk; // Multiply by Aq[i,j,k]
    }

    // --- Term 3.a: grad_As from Ys (gather) ---
     if (k < N_dim) {
        const float* grad_out_row_k = grad_output_slice + (int64_t)k * D_dim;
        for (int d = 0; d < D_dim; ++d) {
            grad_as_val += grad_out_row_k[d] * vq1_row_i[d] * vr1_row_j[d];
        }
    }

    // --- Term 3.b: grad_As from Yq' (scatter) ---
     if (i < N_dim) {
        float grad_x_term_i = 0.0f;
        const float* grad_out_row_i = grad_output_slice + (int64_t)i * D_dim;
        for (int d = 0; d < D_dim; ++d) {
            grad_x_term_i += grad_out_row_i[d] * vr2_row_j[d] * vs2_row_k[d];
        }
        grad_as_val += grad_x_term_i * ar_ijk; // Multiply by Ar[i,j,k]
    }
    
    // --- Term 3.c: grad_As from Yr' (scatter) ---
    if (j < N_dim) {
        float grad_x_term_j = 0.0f;
        const float* grad_out_row_j = grad_output_slice + (int64_t)j * D_dim;
        for (int d = 0; d < D_dim; ++d) {
            grad_x_term_j += grad_out_row_j[d] * vq2_row_i[d] * vs2_row_k[d];
        }
        grad_as_val += grad_x_term_j * aq_ijk; // Multiply by Aq[i,j,k]
    }

    // Write results to output tensors
    grad_Aq_slice_out[ijk_idx] = grad_aq_val;
    grad_Ar_slice_out[ijk_idx] = grad_ar_val;
    grad_As_slice_out[ijk_idx] = grad_as_val;
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor>
compute_interim_grads_cuda_wrapper(
    // Inputs (GPU tensors)
    const torch::Tensor& grad_output_slice, // [N, D]
    const torch::Tensor& Vq_1_slice,        // [I, D]
    const torch::Tensor& Vq_2_slice,        // [I, D]
    const torch::Tensor& Vr_1_slice,        // [J, D]
    const torch::Tensor& Vr_2_slice,        // [J, D]
    const torch::Tensor& Vs_1_slice,        // [K, D]
    const torch::Tensor& Vs_2_slice,        // [K, D]
    const torch::Tensor& Aq_slice,          // [I, J, K]
    const torch::Tensor& Ar_slice,          // [I, J, K]
    const torch::Tensor& As_slice           // [I, J, K]
) {
    // Input validation (basic checks)
    TORCH_CHECK(grad_output_slice.is_cuda(), "Input grad_output_slice must be CUDA tensor");
    // ... check other inputs are CUDA ...
    TORCH_CHECK(Aq_slice.dim() == 3, "Aq_slice must be 3D");
    // ... check other dimensions ...

    const int I = Vq_1_slice.size(0);
    const int J = Vr_1_slice.size(0);
    const int K = Vs_1_slice.size(0);
    const int D = Vq_1_slice.size(1);
    const int N = grad_output_slice.size(0);

    // Check dimension consistency
    TORCH_CHECK(Vq_2_slice.sizes() == Vq_1_slice.sizes(), "Vq shapes mismatch");
    TORCH_CHECK(Vr_2_slice.sizes() == Vr_1_slice.sizes(), "Vr shapes mismatch");
    TORCH_CHECK(Vs_2_slice.sizes() == Vs_1_slice.sizes(), "Vs shapes mismatch");
    TORCH_CHECK(Aq_slice.sizes() == torch::IntArrayRef({I, J, K}), "Aq shape mismatch");
    TORCH_CHECK(Ar_slice.sizes() == torch::IntArrayRef({I, J, K}), "Ar shape mismatch");
    TORCH_CHECK(As_slice.sizes() == torch::IntArrayRef({I, J, K}), "As shape mismatch");
    TORCH_CHECK(grad_output_slice.size(1) == D, "grad_output D mismatch");
    // ... add more checks as needed ...

    // Allocate output tensors (temporary grads)
    auto options = Aq_slice.options();
    torch::Tensor grad_Aq_slice_out = torch::zeros({I, J, K}, options);
    torch::Tensor grad_Ar_slice_out = torch::zeros({I, J, K}, options);
    torch::Tensor grad_As_slice_out = torch::zeros({I, J, K}, options);

    // Kernel launch configuration (similar to compute_A_slice_kernel)
    constexpr int BLOCK_DIM_I = 8;
    constexpr int BLOCK_DIM_J = 8;
    constexpr int BLOCK_DIM_K = 8; 
    dim3 blockDim(BLOCK_DIM_I, BLOCK_DIM_J, BLOCK_DIM_K);
    dim3 gridDim(
        (I + BLOCK_DIM_I - 1) / BLOCK_DIM_I,
        (J + BLOCK_DIM_J - 1) / BLOCK_DIM_J,
        (K + BLOCK_DIM_K - 1) / BLOCK_DIM_K
    );

    auto grad_output_cont = grad_output_slice.contiguous();
    auto Vq_1_cont = Vq_1_slice.contiguous();
    auto Vq_2_cont = Vq_2_slice.contiguous();
    auto Vr_1_cont = Vr_1_slice.contiguous();
    auto Vr_2_cont = Vr_2_slice.contiguous();
    auto Vs_1_cont = Vs_1_slice.contiguous();
    auto Vs_2_cont = Vs_2_slice.contiguous();
    auto Aq_cont = Aq_slice.contiguous();
    auto Ar_cont = Ar_slice.contiguous();
    auto As_cont = As_slice.contiguous();

    // Launch Kernel
    compute_interim_grads_kernel<<<gridDim, blockDim>>>(
        grad_output_cont.data_ptr<float>(),
        Vq_1_cont.data_ptr<float>(), Vq_2_cont.data_ptr<float>(),
        Vr_1_cont.data_ptr<float>(), Vr_2_cont.data_ptr<float>(),
        Vs_1_cont.data_ptr<float>(), Vs_2_cont.data_ptr<float>(),
        Aq_cont.data_ptr<float>(), Ar_cont.data_ptr<float>(), As_cont.data_ptr<float>(),
        grad_Aq_slice_out.data_ptr<float>(),
        grad_Ar_slice_out.data_ptr<float>(),
        grad_As_slice_out.data_ptr<float>(),
        I, J, K, D, N
    );

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA error in compute_interim_grads_cuda_wrapper: %s\n", cudaGetErrorString(err));
    }

    return std::make_tuple(grad_Aq_slice_out, grad_Ar_slice_out, grad_As_slice_out);
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor>
get_interim_grads_cpu(
    const torch::Tensor& grad_output_slice_cpu, // [N, D]
    const torch::Tensor& Q_slice_cpu,           // [I, D]
    const torch::Tensor& R_slice_cpu,           // [J, D]
    const torch::Tensor& S_slice_cpu,           // [K, D]
    const torch::Tensor& Vq_1_slice_cpu,        // [I, D]
    const torch::Tensor& Vq_2_slice_cpu,        // [I, D]
    const torch::Tensor& Vr_1_slice_cpu,        // [J, D]
    const torch::Tensor& Vr_2_slice_cpu,        // [J, D]
    const torch::Tensor& Vs_1_slice_cpu,        // [K, D]
    const torch::Tensor& Vs_2_slice_cpu,        // [K, D]
    const torch::Tensor& Aq_slice_cpu,          // [I, J, K]
    const torch::Tensor& Ar_slice_cpu,          // [I, J, K]
    const torch::Tensor& As_slice_cpu           // [I, J, K]
) {
    const int I = Q_slice_cpu.size(0);
    const int J = R_slice_cpu.size(0);
    const int K = S_slice_cpu.size(0);
    const int D = Q_slice_cpu.size(1);
    const int N = grad_output_slice_cpu.size(0); 
    auto options = Aq_slice_cpu.options(); 

    auto grad_Aq_slice_ref = torch::zeros({I, J, K}, options);
    auto grad_Ar_slice_ref = torch::zeros({I, J, K}, options);
    auto grad_As_slice_ref = torch::zeros({I, J, K}, options);

    auto grad_Aq_acc = grad_Aq_slice_ref.accessor<float, 3>();
    auto grad_Ar_acc = grad_Ar_slice_ref.accessor<float, 3>();
    auto grad_As_acc = grad_As_slice_ref.accessor<float, 3>();
    
    auto grad_output_acc = grad_output_slice_cpu.accessor<float, 2>();
    auto Vq_1_acc = Vq_1_slice_cpu.accessor<float, 2>();
    auto Vq_2_acc = Vq_2_slice_cpu.accessor<float, 2>();
    auto Vr_1_acc = Vr_1_slice_cpu.accessor<float, 2>();
    auto Vr_2_acc = Vr_2_slice_cpu.accessor<float, 2>();
    auto Vs_1_acc = Vs_1_slice_cpu.accessor<float, 2>();
    auto Vs_2_acc = Vs_2_slice_cpu.accessor<float, 2>();
    auto Aq_acc = Aq_slice_cpu.accessor<float, 3>();
    auto Ar_acc = Ar_slice_cpu.accessor<float, 3>();
    auto As_acc = As_slice_cpu.accessor<float, 3>();

    // Replicate Phase 1 logic from compute_grad_A_single here
    // Term 1.a
    if (I <= N) {
        for (int i = 0; i < I; ++i) { for (int j = 0; j < J; ++j) { for (int k = 0; k < K; ++k) { for (int d = 0; d < D; ++d) {
            grad_Aq_acc[i][j][k] += grad_output_acc[i][d] * Vr_1_acc[j][d] * Vs_1_acc[k][d];
        }}}}
    }
    // Term 1.b
    if (J <= N) {
        for (int i = 0; i < I; ++i) { for (int j = 0; j < J; ++j) { for (int k = 0; k < K; ++k) {
            float grad_x_vals = 0.0f;
            for (int d = 0; d < D; ++d) { grad_x_vals += grad_output_acc[j][d] * Vq_2_acc[i][d] * Vs_2_acc[k][d]; }
            grad_Aq_acc[i][j][k] += grad_x_vals * As_acc[i][j][k];
        }}}
    }
    // Term 1.c
    if (K <= N) {
         for (int i = 0; i < I; ++i) { for (int j = 0; j < J; ++j) { for (int k = 0; k < K; ++k) {
             float grad_x_vals = 0.0f;
             for (int d = 0; d < D; ++d) { grad_x_vals += grad_output_acc[k][d] * Vq_2_acc[i][d] * Vr_2_acc[j][d]; }
             grad_Aq_acc[i][j][k] += grad_x_vals * Ar_acc[i][j][k];
         }}}
     }

    // Term 2.a
     if (J <= N) {
         for (int i = 0; i < I; ++i) { for (int j = 0; j < J; ++j) { for (int k = 0; k < K; ++k) { for (int d = 0; d < D; ++d) {
             grad_Ar_acc[i][j][k] += grad_output_acc[j][d] * Vq_1_acc[i][d] * Vs_1_acc[k][d];
         }}}}
     }
    // Term 2.b
     if (I <= N) {
         for (int i = 0; i < I; ++i) { for (int j = 0; j < J; ++j) { for (int k = 0; k < K; ++k) {
             float grad_x_vals = 0.0f;
             for (int d = 0; d < D; ++d) { grad_x_vals += grad_output_acc[i][d] * Vr_2_acc[j][d] * Vs_2_acc[k][d]; }
             grad_Ar_acc[i][j][k] += grad_x_vals * As_acc[i][j][k];
         }}}
     }
    // Term 2.c
     if (K <= N) {
          for (int i = 0; i < I; ++i) { for (int j = 0; j < J; ++j) { for (int k = 0; k < K; ++k) {
              float grad_x_vals = 0.0f;
              for (int d = 0; d < D; ++d) { grad_x_vals += grad_output_acc[k][d] * Vq_2_acc[i][d] * Vr_2_acc[j][d]; }
              grad_Ar_acc[i][j][k] += grad_x_vals * Aq_acc[i][j][k];
          }}}
      }

    // Term 3.a
      if (K <= N) {
          for (int i = 0; i < I; ++i) { for (int j = 0; j < J; ++j) { for (int k = 0; k < K; ++k) { for (int d = 0; d < D; ++d) {
              grad_As_acc[i][j][k] += grad_output_acc[k][d] * Vq_1_acc[i][d] * Vr_1_acc[j][d];
          }}}}
      }
    // Term 3.b
      if (I <= N) {
          for (int i = 0; i < I; ++i) { for (int j = 0; j < J; ++j) { for (int k = 0; k < K; ++k) {
              float grad_x_vals = 0.0f;
              for (int d = 0; d < D; ++d) { grad_x_vals += grad_output_acc[i][d] * Vr_2_acc[j][d] * Vs_2_acc[k][d]; }
              grad_As_acc[i][j][k] += grad_x_vals * Ar_acc[i][j][k];
          }}}
      }
    // Term 3.c
      if (J <= N) {
          for (int i = 0; i < I; ++i) { for (int j = 0; j < J; ++j) { for (int k = 0; k < K; ++k) {
              float grad_x_vals = 0.0f;
              for (int d = 0; d < D; ++d) { grad_x_vals += grad_output_acc[j][d] * Vq_2_acc[i][d] * Vs_2_acc[k][d]; }
              grad_As_acc[i][j][k] += grad_x_vals * Aq_acc[i][j][k];
          }}}
      }

    return std::make_tuple(grad_Aq_slice_ref, grad_Ar_slice_ref, grad_As_slice_ref);
}

//     cudaError_t err = cudaGetLastError();
//     if (err != cudaSuccess) {
//         fprintf(stderr, "CUDA error in apply_softmax_backward_cuda_wrapper: %s\n", cudaGetErrorString(err));
//     }

//     return grad_A_slice_out_gpu;
// }

// --- Optimized Softmax Backward Implementation ---

// Stage 1, Kernel 1: Compute sum_q[i] = sum_{j,k} (grad_Aq[i,j,k] * Aq[i,j,k])
__global__ void compute_softmax_backward_sum_q_kernel(
    const float* __restrict__ grad_Aq_slice_in,
    const float* __restrict__ Aq_slice_in,
    float* __restrict__ sum_q_vec_out, // [I]
    int I_dim, int J_dim, int K_dim)
{
    extern __shared__ float s_reduction_pad[];
    int i_current = blockIdx.x;
    if (i_current >= I_dim) return;

    const float* grad_Aq_plane = grad_Aq_slice_in + (int64_t)i_current * J_dim * K_dim;
    const float* Aq_plane = Aq_slice_in + (int64_t)i_current * J_dim * K_dim;

    int plane_size = J_dim * K_dim;
    int tid_in_block = threadIdx.x;
    int threads_in_block = blockDim.x;

    // --- Pass 1: Compute sum of products for the plane ---
    float thread_sum = 0.0f;
    for (int idx = tid_in_block; idx < plane_size; idx += threads_in_block) {
        thread_sum += grad_Aq_plane[idx] * Aq_plane[idx];
    }
    s_reduction_pad[tid_in_block] = thread_sum;
    __syncthreads();

    // --- Pass 2: Reduce sums in shared memory ---
    for (int offset = threads_in_block / 2; offset > 0; offset >>= 1) {
        if (tid_in_block < offset) {
            s_reduction_pad[tid_in_block] += s_reduction_pad[tid_in_block + offset];
        }
        __syncthreads();
    }

    if (tid_in_block == 0) {
        sum_q_vec_out[i_current] = s_reduction_pad[0];
    }
}

// Stage 1, Kernel 2: Compute sum_r[j] = sum_{i,k} (grad_Ar[i,j,k] * Ar[i,j,k])
__global__ void compute_softmax_backward_sum_r_kernel(
    const float* __restrict__ grad_Ar_slice_in,
    const float* __restrict__ Ar_slice_in,
    float* __restrict__ sum_r_vec_out, // [J]
    int I_dim, int J_dim, int K_dim)
{
    extern __shared__ float s_reduction_pad[];
    int j_current = blockIdx.x;
    if (j_current >= J_dim) return;

    int plane_size = I_dim * K_dim;
    int tid_in_block = threadIdx.x;
    int threads_in_block = blockDim.x;

    float thread_sum = 0.0f;
    for (int idx = tid_in_block; idx < plane_size; idx += threads_in_block) {
        int i = idx / K_dim;
        int k = idx % K_dim;
        int64_t global_idx = (int64_t)i * J_dim * K_dim + (int64_t)j_current * K_dim + k;
        thread_sum += grad_Ar_slice_in[global_idx] * Ar_slice_in[global_idx];
    }
    s_reduction_pad[tid_in_block] = thread_sum;
    __syncthreads();

    for (int offset = threads_in_block / 2; offset > 0; offset >>= 1) {
        if (tid_in_block < offset) {
            s_reduction_pad[tid_in_block] += s_reduction_pad[tid_in_block + offset];
        }
        __syncthreads();
    }

    if (tid_in_block == 0) {
        sum_r_vec_out[j_current] = s_reduction_pad[0];
    }
}

// Stage 1, Kernel 3: Compute sum_s[k] = sum_{i,j} (grad_As[i,j,k] * As[i,j,k])
__global__ void compute_softmax_backward_sum_s_kernel(
    const float* __restrict__ grad_As_slice_in,
    const float* __restrict__ As_slice_in,
    float* __restrict__ sum_s_vec_out, // [K]
    int I_dim, int J_dim, int K_dim)
{
    extern __shared__ float s_reduction_pad[];
    int k_current = blockIdx.x;
    if (k_current >= K_dim) return;

    int plane_size = I_dim * J_dim;
    int tid_in_block = threadIdx.x;
    int threads_in_block = blockDim.x;

    float thread_sum = 0.0f;
    for (int idx = tid_in_block; idx < plane_size; idx += threads_in_block) {
        int i = idx / J_dim;
        int j = idx % J_dim;
        int64_t global_idx = (int64_t)i * J_dim * K_dim + (int64_t)j * K_dim + k_current;
        thread_sum += grad_As_slice_in[global_idx] * As_slice_in[global_idx];
    }
    s_reduction_pad[tid_in_block] = thread_sum;
    __syncthreads();

    for (int offset = threads_in_block / 2; offset > 0; offset >>= 1) {
        if (tid_in_block < offset) {
            s_reduction_pad[tid_in_block] += s_reduction_pad[tid_in_block + offset];
        }
        __syncthreads();
    }

    if (tid_in_block == 0) {
        sum_s_vec_out[k_current] = s_reduction_pad[0];
    }
}


// Stage 2: Final combination kernel
__global__ void apply_softmax_backward_optimized_kernel(
    const float* __restrict__ grad_Aq_slice_in,
    const float* __restrict__ grad_Ar_slice_in,
    const float* __restrict__ grad_As_slice_in,
    const float* __restrict__ Aq_slice_in,
    const float* __restrict__ Ar_slice_in,
    const float* __restrict__ As_slice_in,
    const float* __restrict__ sum_q_vec, // [I]
    const float* __restrict__ sum_r_vec, // [J]
    const float* __restrict__ sum_s_vec, // [K]
    float* __restrict__ grad_A_slice_out, // [I, J, K]
    int I_dim, int J_dim, int K_dim
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    int k = blockIdx.z * blockDim.z + threadIdx.z;

    if (i >= I_dim || j >= J_dim || k >= K_dim) {
        return;
    }

    int64_t ijk_idx = (int64_t)i * J_dim * K_dim + (int64_t)j * K_dim + k;

    // Each component is calculated without loops using the pre-computed sums
    float grad_A_q_comp = (grad_Aq_slice_in[ijk_idx] - sum_q_vec[i]) * Aq_slice_in[ijk_idx];
    float grad_A_r_comp = (grad_Ar_slice_in[ijk_idx] - sum_r_vec[j]) * Ar_slice_in[ijk_idx];
    float grad_A_s_comp = (grad_As_slice_in[ijk_idx] - sum_s_vec[k]) * As_slice_in[ijk_idx];

    grad_A_slice_out[ijk_idx] = grad_A_q_comp + grad_A_r_comp + grad_A_s_comp;
}


torch::Tensor apply_softmax_backward_cuda_wrapper(
    const torch::Tensor& grad_Aq_slice_gpu,
    const torch::Tensor& grad_Ar_slice_gpu,
    const torch::Tensor& grad_As_slice_gpu,
    const torch::Tensor& Aq_slice_gpu,
    const torch::Tensor& Ar_slice_gpu,
    const torch::Tensor& As_slice_gpu
) {
    TORCH_CHECK(grad_Aq_slice_gpu.is_cuda(), "grad_Aq_slice_gpu must be CUDA");
    TORCH_CHECK(grad_Aq_slice_gpu.dim() == 3 && Aq_slice_gpu.dim() == 3, "Inputs must be 3D");
    
    const int I = grad_Aq_slice_gpu.size(0);
    const int J = grad_Aq_slice_gpu.size(1);
    const int K = grad_Aq_slice_gpu.size(2);
    auto options = grad_Aq_slice_gpu.options();

    // Ensure inputs are contiguous
    auto grad_Aq_cont = grad_Aq_slice_gpu.contiguous();
    auto grad_Ar_cont = grad_Ar_slice_gpu.contiguous();
    auto grad_As_cont = grad_As_slice_gpu.contiguous();
    auto Aq_cont = Aq_slice_gpu.contiguous();
    auto Ar_cont = Ar_slice_gpu.contiguous();
    auto As_cont = As_slice_gpu.contiguous();

    // --- Stage 1: Pre-compute sums ---
    torch::Tensor sum_q_vec = torch::zeros({I}, options);
    torch::Tensor sum_r_vec = torch::zeros({J}, options);
    torch::Tensor sum_s_vec = torch::zeros({K}, options);
    
    int threads_per_block = 256; // Common choice for reduction kernels

    // Launch sum_q kernel
    dim3 gridDim_q(I);
    dim3 blockDim(threads_per_block);
    size_t shmem_size = threads_per_block * sizeof(float);
    compute_softmax_backward_sum_q_kernel<<<gridDim_q, blockDim, shmem_size>>>(
        grad_Aq_cont.data_ptr<float>(), Aq_cont.data_ptr<float>(), sum_q_vec.data_ptr<float>(), I, J, K
    );

    // Launch sum_r kernel
    dim3 gridDim_r(J);
    compute_softmax_backward_sum_r_kernel<<<gridDim_r, blockDim, shmem_size>>>(
        grad_Ar_cont.data_ptr<float>(), Ar_cont.data_ptr<float>(), sum_r_vec.data_ptr<float>(), I, J, K
    );

    // Launch sum_s kernel
    dim3 gridDim_s(K);
    compute_softmax_backward_sum_s_kernel<<<gridDim_s, blockDim, shmem_size>>>(
        grad_As_cont.data_ptr<float>(), As_cont.data_ptr<float>(), sum_s_vec.data_ptr<float>(), I, J, K
    );

    // --- Stage 2: Final Combination ---
    torch::Tensor grad_A_slice_out_gpu = torch::zeros({I, J, K}, options);
    constexpr int BLOCK_DIM_I = 8;
    constexpr int BLOCK_DIM_J = 8;
    constexpr int BLOCK_DIM_K = 8; 
    dim3 blockDim_final(BLOCK_DIM_I, BLOCK_DIM_J, BLOCK_DIM_K);
    dim3 gridDim_final(
        (I + BLOCK_DIM_I - 1) / BLOCK_DIM_I,
        (J + BLOCK_DIM_J - 1) / BLOCK_DIM_J,
        (K + BLOCK_DIM_K - 1) / BLOCK_DIM_K
    );

    apply_softmax_backward_optimized_kernel<<<gridDim_final, blockDim_final>>>(
        grad_Aq_cont.data_ptr<float>(), grad_Ar_cont.data_ptr<float>(), grad_As_cont.data_ptr<float>(),
        Aq_cont.data_ptr<float>(), Ar_cont.data_ptr<float>(), As_cont.data_ptr<float>(),
        sum_q_vec.data_ptr<float>(), sum_r_vec.data_ptr<float>(), sum_s_vec.data_ptr<float>(),
        grad_A_slice_out_gpu.data_ptr<float>(),
        I, J, K
    );
    
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA error in apply_softmax_backward_cuda_wrapper (optimized): %s\n", cudaGetErrorString(err));
    }

    return grad_A_slice_out_gpu;
}

// Kernel to compute gradient for Q, assuming grad_A is pre-computed
__global__ void grad_Q_kernel(
    const float* __restrict__ grad_A, // Shape [B, H, I, J, K]
    const float* __restrict__ R,      // Shape [B, H, J, D]
    const float* __restrict__ S,      // Shape [B, H, K, D]
    float*       __restrict__ grad_Q, // Shape [B, H, I, D] - Output
    const int B, const int H, const int I, const int J, const int K, const int D,
    const float scale
) {
    // --- Calculate indices for this thread ---
    // Map threads to output elements (b, h, i, d)
    // Using 1D grid/block for simplicity, can be optimized later
    int64_t idx = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    int64_t total_elements = (int64_t)B * H * I * D;
    if (idx >= total_elements) {
        return;
    }

    // Decode indices (b, h, i, d) from linear index idx
    const int d = idx % D;
    const int64_t temp_idx_d = idx / D;
    const int i = temp_idx_d % I;
    const int64_t temp_idx_i = temp_idx_d / I;
    const int h = temp_idx_i % H;
    const int b = temp_idx_i / H;


    // --- Calculate strides ---
    const int64_t stride_A_B = (int64_t)H * I * J * K;
    const int64_t stride_A_H = (int64_t)I * J * K;
    const int64_t stride_A_I = (int64_t)J * K;
    const int64_t stride_A_J = (int64_t)K;
    const int64_t stride_A_K = 1;

    const int64_t stride_R_B = (int64_t)H * J * D;
    const int64_t stride_R_H = (int64_t)J * D;
    const int64_t stride_R_J = (int64_t)D;
    const int64_t stride_R_D = 1;

    const int64_t stride_S_B = (int64_t)H * K * D;
    const int64_t stride_S_H = (int64_t)K * D;
    const int64_t stride_S_K = (int64_t)D;
    const int64_t stride_S_D = 1;

    // Output stride calculation not needed as we write to grad_Q[idx]


    // --- Calculate base pointers for this slice (b, h) ---
    // Note: We need the base for the entire tensors here, not just slices,
    // because the loops below access elements across different j and k.
    const float* grad_A_base = grad_A + b * stride_A_B + h * stride_A_H;
    const float* R_base = R + b * stride_R_B + h * stride_R_H;
    const float* S_base = S + b * stride_S_B + h * stride_S_H;


    // --- Compute sum for grad_Q[b, h, i, d] ---
    // grad_Q[i,d] = scale * sum_{j,k} ( grad_A[i,j,k] * R[j,d] * S[k,d] )
    float sum_for_grad_q = 0.0f;
    for (int j_idx = 0; j_idx < J; ++j_idx) {
        for (int k_idx = 0; k_idx < K; ++k_idx) {
            // Calculate linear indices relative to slice base pointers (b,h)
            // grad_A[i, j, k] within the (b,h) slice
            int64_t idx_A = (int64_t)i * stride_A_I + (int64_t)j_idx * stride_A_J + (int64_t)k_idx * stride_A_K;
            // R[j, d] within the (b,h) slice
            int64_t idx_R = (int64_t)j_idx * stride_R_J + (int64_t)d * stride_R_D;
            // S[k, d] within the (b,h) slice
            int64_t idx_S = (int64_t)k_idx * stride_S_K + (int64_t)d * stride_S_D;

            // Accumulate directly as float
            sum_for_grad_q += grad_A_base[idx_A] * R_base[idx_R] * S_base[idx_S];
        }
    }

    grad_Q[idx] = scale * sum_for_grad_q;
}

// Kernel to compute gradient for R, following grad_Q pattern
__global__ void grad_R_kernel(
    const float* __restrict__ grad_A, // Shape [B, H, I, J, K]
    const float* __restrict__ Q,      // Shape [B, H, I, D]
    const float* __restrict__ S,      // Shape [B, H, K, D]
    float*       __restrict__ grad_R, // Shape [B, H, J, D] - Output
    const int B_dim, const int H_dim, const int I_dim, const int J_dim, const int K_dim, const int D_dim,
    const float scale
) {
    // --- Calculate indices for this thread ---
    // Map threads to output elements (b, h, j, d)
    int64_t idx = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    int64_t total_elements = (int64_t)B_dim * H_dim * J_dim * D_dim; // Output shape B*H*J*D
    if (idx >= total_elements) {
        return;
    }

    // Decode indices (b, h, j, d) from linear index idx
    const int d = idx % D_dim;
    const int64_t temp_idx_d = idx / D_dim;
    const int j = temp_idx_d % J_dim;
    const int64_t temp_idx_j = temp_idx_d / J_dim;
    const int h = temp_idx_j % H_dim;
    const int b = temp_idx_j / H_dim;


    // --- Calculate strides ---
    const int64_t stride_A_B = (int64_t)H_dim * I_dim * J_dim * K_dim;
    const int64_t stride_A_H = (int64_t)I_dim * J_dim * K_dim;
    const int64_t stride_A_I = (int64_t)J_dim * K_dim;
    const int64_t stride_A_J = (int64_t)K_dim;
    const int64_t stride_A_K = 1;

    const int64_t stride_Q_B = (int64_t)H_dim * I_dim * D_dim;
    const int64_t stride_Q_H = (int64_t)I_dim * D_dim;
    const int64_t stride_Q_I = (int64_t)D_dim;
    const int64_t stride_Q_D = 1;

    const int64_t stride_S_B = (int64_t)H_dim * K_dim * D_dim;
    const int64_t stride_S_H = (int64_t)K_dim * D_dim;
    const int64_t stride_S_K = (int64_t)D_dim;
    const int64_t stride_S_D = 1;

    // Output stride calculation not needed as we write to grad_R[idx]


    // --- Calculate base pointers for this slice (b, h) ---
    // Note: We need the base for the entire tensors here, not just slices,
    // because the loops below access elements across different i and k.
    const float* grad_A_base = grad_A + b * stride_A_B + h * stride_A_H;
    const float* Q_base = Q + b * stride_Q_B + h * stride_Q_H;
    const float* S_base = S + b * stride_S_B + h * stride_S_H;


    // --- Compute sum for grad_R[b, h, j, d] ---
    // grad_R[b, h, j, d] = scale * sum_{i,k} ( grad_A[i,j,k] * Q[i,d] * S[k,d] )
    float sum_for_grad_r = 0.0f;
    for (int i_idx = 0; i_idx < I_dim; ++i_idx) {
        for (int k_idx = 0; k_idx < K_dim; ++k_idx) {
            // Calculate linear indices relative to slice base pointers (b,h)
            // grad_A[i, j, k] within the (b,h) slice
            int64_t idx_A = (int64_t)i_idx * stride_A_I + (int64_t)j * stride_A_J + (int64_t)k_idx * stride_A_K;
            // Q[i, d] within the (b,h) slice
            int64_t idx_Q = (int64_t)i_idx * stride_Q_I + (int64_t)d * stride_Q_D;
            // S[k, d] within the (b,h) slice
            int64_t idx_S = (int64_t)k_idx * stride_S_K + (int64_t)d * stride_S_D;

            // Accumulate directly as float
            sum_for_grad_r += grad_A_base[idx_A] * Q_base[idx_Q] * S_base[idx_S];
        }
    }

    grad_R[idx] = scale * sum_for_grad_r;
}

// Kernel to compute gradient for S, following grad_Q/R pattern
__global__ void grad_S_kernel(
    const float* __restrict__ grad_A, // Shape [B, H, I, J, K]
    const float* __restrict__ Q,      // Shape [B, H, I, D]
    const float* __restrict__ R,      // Shape [B, H, J, D]
    float*       __restrict__ grad_S, // Shape [B, H, K, D] - Output
    const int B_dim, const int H_dim, const int I_dim, const int J_dim, const int K_dim, const int D_dim,
    const float scale
) {
    // Map threads to output elements (b, h, k, d)
    int64_t idx = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    int64_t total_elements = (int64_t)B_dim * H_dim * K_dim * D_dim; // Output shape B*H*K*D
    if (idx >= total_elements) {
        return;
    }

    // Decode indices (b, h, k, d) from linear index idx
    const int d = idx % D_dim;
    const int64_t temp_idx_d = idx / D_dim;
    const int k = temp_idx_d % K_dim; // This thread calculates for grad_S[k,d]
    const int64_t temp_idx_k = temp_idx_d / K_dim;
    const int h = temp_idx_k % H_dim;
    const int b = temp_idx_k / H_dim;

    // Strides for grad_A[B, H, I, J, K]
    const int64_t stride_A_B = (int64_t)H_dim * I_dim * J_dim * K_dim;
    const int64_t stride_A_H = (int64_t)I_dim * J_dim * K_dim;
    const int64_t stride_A_I = (int64_t)J_dim * K_dim;
    const int64_t stride_A_J = (int64_t)K_dim;
    const int64_t stride_A_K = 1;

    // Strides for Q[B, H, I, D]
    const int64_t stride_Q_B = (int64_t)H_dim * I_dim * D_dim;
    const int64_t stride_Q_H = (int64_t)I_dim * D_dim;
    const int64_t stride_Q_I = (int64_t)D_dim;
    const int64_t stride_Q_D = 1;

    // Strides for R[B, H, J, D]
    const int64_t stride_R_B = (int64_t)H_dim * J_dim * D_dim;
    const int64_t stride_R_H = (int64_t)J_dim * D_dim;
    const int64_t stride_R_J = (int64_t)D_dim;
    const int64_t stride_R_D = 1;

    // Calculate base pointers for this batch/head (b, h)
    const float* grad_A_base = grad_A + b * stride_A_B + h * stride_A_H;
    const float* Q_base = Q + b * stride_Q_B + h * stride_Q_H;
    const float* R_base = R + b * stride_R_B + h * stride_R_H;

    // Compute sum for grad_S[b, h, k, d]
    // grad_S[k,d] = scale * sum_{i,j} ( grad_A[i,j,k] * Q[i,d] * R[j,d] )
    float sum_for_grad_s = 0.0f;
    for (int i_loop = 0; i_loop < I_dim; ++i_loop) { // Sum over i
        for (int j_loop = 0; j_loop < J_dim; ++j_loop) { // Sum over j
            // Calculate linear indices relative to slice base pointers (b,h)
            int64_t idx_A = (int64_t)i_loop * stride_A_I + (int64_t)j_loop * stride_A_J + (int64_t)k * stride_A_K;
            int64_t idx_Q = (int64_t)i_loop * stride_Q_I + (int64_t)d * stride_Q_D;
            int64_t idx_R = (int64_t)j_loop * stride_R_J + (int64_t)d * stride_R_D;

            sum_for_grad_s += grad_A_base[idx_A] * Q_base[idx_Q] * R_base[idx_R];
        }
    }
    grad_S[idx] = scale * sum_for_grad_s;
}



/*********************************************************************
*  grad_Vq1_tbIK_kernel_v2  (Approach B, with shared memory reduction)
*********************************************************************/
#ifndef T_I
  #define T_I  8
#endif
#ifndef T_K
  #define T_K  8
#endif
#ifndef T_J
  #define T_J  8
#endif

__global__ void grad_Vq1_tbIK_kernel(
    const float* __restrict__ Q,        // [B,H,N,D]
    const float* __restrict__ R,        // [B,H,N,D]
    const float* __restrict__ S,        // [B,H,N,D]
    const float* __restrict__ Vs,       // [B,H,N,D]   (Vs_1 slice)
    const float* __restrict__ Vr,       // [B,H,N,D]   (Vr_1 slice)
    const float* __restrict__ gradY,    // [B,H,N,D]   (grad_output slice)   <-- single source
    const float* __restrict__ m_j,      // [B,H,N]
    const float* __restrict__ l_j,      // [B,H,N]
    const float* __restrict__ m_k,      // [B,H,N]
    const float* __restrict__ l_k,      // [B,H,N]
    float*       __restrict__ gradVq,   // [B,H,N,D]   (output)
    int  N, int D,
    float scale )
{
    /* ---- decode indices ------------------------------------------------ */
    int bh  = blockIdx.z;          // flattened (batch, head)
    int i0  = blockIdx.x * T_I + threadIdx.x;
    int k0  = blockIdx.y * T_K + threadIdx.y;
    if (i0 >= N || k0 >= N) return;

    const int64_t stride_BH = (int64_t)N * D;
    const float* QBH   = Q       + bh * stride_BH;
    const float* RBH   = R       + bh * stride_BH;
    const float* SBH   = S       + bh * stride_BH;
    const float* VsBH  = Vs      + bh * stride_BH;
    const float* VrBH  = Vr      + bh * stride_BH;
    const float* gYBH  = gradY   + bh * stride_BH;
    const float* mJBH  = m_j     + (int64_t)bh * N;
    const float* lJBH  = l_j     + (int64_t)bh * N;
    const float* mKBH  = m_k     + (int64_t)bh * N;
    const float* lKBH  = l_k     + (int64_t)bh * N;
          float* gVqBH = gradVq  + bh * stride_BH;

    /* ---- load Q[i,:]  S[k,:]  Vs[k,:] into registers ------------------- */
    float q_vec[MAX_D_REG];    // assumes D ≤ 16 (adjust if larger)
    float s_vec[MAX_D_REG], vs_vec[MAX_D_REG];
    #pragma unroll
    for (int d=0; d<D; ++d){
        q_vec[d]  = QBH[i0*D + d];
        s_vec[d]  = SBH[k0*D + d];
        vs_vec[d] = VsBH[k0*D + d];
    }
    float grad_acc[MAX_D_REG] = {0.0f};

    /* ---- loop over J in chunks ---------------------------------------- */
    for (int jBase=0; jBase<N; jBase+=T_J){
        __shared__ float sh_R [T_J][MAX_D_REG];
        __shared__ float sh_Vr[T_J][MAX_D_REG];
        __shared__ float sh_gY[T_J][MAX_D_REG];
        __shared__ float sh_mj[T_J];
        __shared__ float sh_lj[T_J];

        int lj = threadIdx.y;                       // 0 … T_K-1 (≤T_J)
        if (lj < T_J && (jBase+lj) < N){
            int jGlob = jBase + lj;
            #pragma unroll
            for (int d=threadIdx.x; d<D; d+=T_I){
                sh_R [lj][d] = RBH [jGlob*D + d];
                sh_Vr[lj][d] = VrBH[jGlob*D + d];
                sh_gY[lj][d] = gYBH[jGlob*D + d];   // grad_Yr = grad_output
            }
            if (threadIdx.x == 0){
                sh_mj[lj] = mJBH[jGlob];
                sh_lj[lj] = lJBH[jGlob];
            }
        }
        __syncthreads();

        /* ---- iterate inside loaded j-chunk ----------------------------- */
        for (int jOff=0; jOff<T_J && (jBase+jOff)<N; ++jOff){
            float logits=0.f;
            #pragma unroll
            for (int d=0; d<D; ++d)
                logits += q_vec[d]*sh_R[jOff][d]*s_vec[d];
            logits *= scale;

            float wj = __expf(logits - sh_mj[jOff]) / sh_lj[jOff];
            float wk = __expf(logits - mKBH[k0])    / lKBH[k0];

            #pragma unroll
            for (int d=0; d<D; ++d){
                grad_acc[d] += wj * sh_gY[jOff][d] * vs_vec[d]        /* Yr path */
                              + wk * gYBH[k0*D + d]  * sh_Vr[jOff][d];/* Ys path */
            }
        }
        __syncthreads();
    }

    /* ---- atomic add to global grad ------------------------------------ */
    #pragma unroll
    for (int d=0; d<D; ++d)
        atomicAdd(&gVqBH[i0*D + d], grad_acc[d]);
}

// ===================== scatter-grad Vq2 kernel (tile-by-I,K) =====================
// Similar layout to grad_Vq1_tbIK_kernel but accumulates two terms:
//   w1 = Aq*As  -> dy_r * w1 * Vs2
//   w2 = Aq*Ar  -> dy_s * w2 * Vr2
// Requires m_i,l_i,m_j,l_j,m_k,l_k – m_i,l_i computed by new Ai_tiled_softmax.

__global__ void grad_Vq2_tbIK_kernel(
    const float* __restrict__ Q,      // [B,H,N,D]
    const float* __restrict__ R,      // [B,H,N,D]
    const float* __restrict__ S,      // [B,H,N,D]
    const float* __restrict__ Vr2,    // [B,H,N,D]
    const float* __restrict__ Vs2,    // [B,H,N,D]
    const float* __restrict__ gradY,  // [B,H,N,D]  (same tensor provides dYr & dYs)
    const float* __restrict__ m_i,    // [B,H,N]
    const float* __restrict__ l_i,    // [B,H,N]
    const float* __restrict__ m_j,    // [B,H,N]
    const float* __restrict__ l_j,    // [B,H,N]
    const float* __restrict__ m_k,    // [B,H,N]
    const float* __restrict__ l_k,    // [B,H,N]
    float*       __restrict__ gradVq, // [B,H,N,D]
    int N, int D, float scale)
{
    /* threadblock organisation = (i,k) tile  --------------------------- */
    const int i0 = blockIdx.x * T_I + threadIdx.x;   // 0..N-1 (column)
    const int k0 = blockIdx.y * T_K + threadIdx.y;   // 0..N-1 (row)
    const int bh = blockIdx.z;                       // flattened (B,H)
    if (i0 >= N || k0 >= N) return;

    // per-BH base pointers and strides
    const int64_t stride_BH = (int64_t)N * D;
    const float* QBH   = Q    + (int64_t)bh * stride_BH;
    const float* RBH   = R    + (int64_t)bh * stride_BH;
    const float* SBH   = S    + (int64_t)bh * stride_BH;
    const float* Vr2BH = Vr2  + (int64_t)bh * stride_BH;
    const float* Vs2BH = Vs2  + (int64_t)bh * stride_BH;
    const float* gYBH  = gradY+ (int64_t)bh * stride_BH;
    const float* m_iBH = m_i  + (int64_t)bh * N;
    const float* l_iBH = l_i  + (int64_t)bh * N;
    const float* m_jBH = m_j  + (int64_t)bh * N;
    const float* l_jBH = l_j  + (int64_t)bh * N;
    const float* m_kBH = m_k  + (int64_t)bh * N;
    const float* l_kBH = l_k  + (int64_t)bh * N;
          float* gVqBH = gradVq + (int64_t)bh * stride_BH;

    // ---- registers for Q[i0], S[k0], Vs2[k0] --------------------------
    float q_vec[MAX_D_REG];
    float s_vec[MAX_D_REG], vs2_vec[MAX_D_REG];
    #pragma unroll
    for (int d=0; d<D; ++d){
        q_vec[d]  = QBH[i0*D + d];
        s_vec[d]  = SBH[k0*D + d];
        vs2_vec[d]= Vs2BH[k0*D + d];
    }
    float grad_acc[MAX_D_REG] = {0.0f};

    // coefficients depending only on i or k  ----------------------------
    const float coeff_i = __expf(-m_iBH[i0]) / l_iBH[i0];
    const float coeff_k = __expf(-m_kBH[k0]) / l_kBH[k0];

    // ---- shared memory tiles for (j) ----------------------------------
    extern __shared__ float shmem[];
    float* sh_R   = shmem;                   // T_J * D
    float* sh_Vr2 = sh_R   + T_J * D;        // T_J * D
    float* sh_gYr = sh_Vr2 + T_J * D;        // T_J * D   (dYr[j,:])
    float* sh_mj  = (float*)(sh_gYr + T_J * D);  // T_J scalars
    float* sh_lj  = sh_mj  + T_J;

    /* iterate over J tiles --------------------------------------------- */
    for (int jBase=0; jBase < N; jBase+=T_J){
        // cooperative load by all (i,k) threads inside TB
        const int ld_idx = threadIdx.y;  // reuse y-dimension for co-load rows
        if (ld_idx < T_J && (jBase+ld_idx) < N){
            const int jGlob = jBase + ld_idx;
            for (int d=threadIdx.x; d<D; d+=T_I){
                sh_R  [ld_idx*D + d] = RBH  [jGlob*D + d];
                sh_Vr2[ld_idx*D + d] = Vr2BH[jGlob*D + d];
                sh_gYr[ld_idx*D + d] = gYBH [jGlob*D + d];
            }
            if (threadIdx.x==0){
                sh_mj[ld_idx] = m_jBH[jGlob];
                sh_lj[ld_idx] = l_jBH[jGlob];
            }
        }
        __syncthreads();

        // ---- loop inside J-tile --------------------------------------
        for (int jOff=0; jOff<T_J && (jBase+jOff)<N; ++jOff){
            // dot(Q[i],R[j],S[k])
            float dot = 0.f;
            #pragma unroll
            for (int d=0; d<D; ++d)
                dot += q_vec[d] * sh_R[jOff*D + d] * s_vec[d];
            float logits = dot * scale;
            float exp_logits = __expf(logits);

            // softmax numerators
            float w_aq   = exp_logits * coeff_i;                    // Aq numerator (omit /li factor later?) Actually Aq=exp(logits-m_i)/l_i = exp_logits*coeff_i
            float w_ar   = exp_logits * (__expf(-sh_mj[jOff]) / sh_lj[jOff]);
            float w_as   = exp_logits * coeff_k;

            // term 1: Aq*As
            float w1 = w_aq * w_as; // exp_logits^2 * ... large but okay FP32
            // term 2: Aq*Ar
            float w2 = w_aq * w_ar;

            // load vectors
            const float* dYr_vec = &sh_gYr[jOff*D];
            const float* Vr2_vec = &sh_Vr2[jOff*D];
            const float* dYs_vec = &gYBH[k0*D]; // contiguous in global, fine

            #pragma unroll
            for (int d=0; d<D; ++d){
                grad_acc[d] += w1 * dYr_vec[d] * vs2_vec[d] +
                               w2 * dYs_vec[d] * Vr2_vec[d];
            }
        }
        __syncthreads();
    }

    // ---- atomic add results ------------------------------------------
    #pragma unroll
    for (int d=0; d<D; ++d)
        atomicAdd(&gVqBH[i0*D + d], grad_acc[d]);
}

// ======================= end kernel =====================================
// ... existing code ...
// After grad_Vq1 launch, insert launch for new scatter kernel
// ... existing code up to grad_Vq1_tbIK_kernel launch ...
// -------------------------------------------------------------------------






// Ar = softmax_{i,k}(A)
static __global__ void Ar_tiled_softmax(
    const float* __restrict__ Q, const float* __restrict__ R, const float* __restrict__ S,
    float* __restrict__ m_j_out, float* __restrict__ l_j_out,
    int B, int H, int I, int J, int K, int D, float scale
) {
    // --- Grid/Block Mapping ---
    // Each block computes the stats for a single j_idx, for a single batch/head.
    const int j_idx = blockIdx.x;
    const int bh_idx = blockIdx.y;
    
    if (j_idx >= J) return;

    const int tid = threadIdx.x;
    const int block_size = blockDim.x;

    // --- Pointers ---
    const int64_t q_bh_offset = (int64_t)bh_idx * I * D;
    const int64_t r_bh_offset = (int64_t)bh_idx * J * D;
    const int64_t s_bh_offset = (int64_t)bh_idx * K * D;

    const float* Q_slice = Q + q_bh_offset;
    const float* R_slice = R + r_bh_offset;
    const float* S_slice = S + s_bh_offset;

    // --- Shared Memory ---
    extern __shared__ float smem[];
    float* r_vec_sh = smem;                              // D
    float* q_tile = r_vec_sh + D;                        // TILE_I * D
    float* s_tile = q_tile + TILE_I * D;                 // TILE_K * D
    float* p_tile = s_tile + TILE_K * D;                 // TILE_I * TILE_K (for intermediate dots)
    float* red_buf = p_tile + TILE_I * TILE_K;           // block_size (for reduction)

    // --- Load R[j] into shared memory ---
    for(int d=tid; d<D; d+=block_size) {
        r_vec_sh[d] = R_slice[j_idx * D + d];
    }
    
    // Each thread holds a local copy of the block-wide stats
    float m_block = -1e30f;
    float l_block = 0.0f;

    __syncthreads();

    // --- Main Loop: Iterate over all i and k tiles ---
    for (int i0 = 0; i0 < I; i0 += TILE_I) {
        for (int k0 = 0; k0 < K; k0 += TILE_K) {
            // Cooperatively load Q and S tiles
            for (int idx = tid; idx < TILE_I * D; idx += block_size) {
                int i_load = i0 + (idx/D);
                if(i_load < I) q_tile[idx] = Q_slice[i_load * D + (idx%D)];
            }
            for (int idx = tid; idx < TILE_K * D; idx += block_size) {
                int k_load = k0 + (idx/D);
                if(k_load < K) s_tile[idx] = S_slice[k_load * D + (idx%D)];
            }
            __syncthreads();

            // --- Compute dot products for this tile and store in p_tile ---
            for (int flat_idx = tid; flat_idx < TILE_I * TILE_K; flat_idx += block_size) {
                int i_tile_idx = flat_idx / TILE_K;
                int k_tile_idx = flat_idx % TILE_K;
                
                float dot = 0.0f;
                if (i0 + i_tile_idx < I && k0 + k_tile_idx < K) {
                    const float* q_vec_sh = q_tile + i_tile_idx * D;
                    const float* s_vec_sh = s_tile + k_tile_idx * D;
                    for(int d=0; d<D; ++d) dot += q_vec_sh[d] * r_vec_sh[d] * s_vec_sh[d];
                    p_tile[flat_idx] = dot * scale;
                } else {
                    p_tile[flat_idx] = -1e30f; // Padding
                }
            }
            __syncthreads();

            // --- Parallel reduction to find tile max (m_tile) ---
            float m_tile_thread = -1e30f;
            for (int flat_idx = tid; flat_idx < TILE_I * TILE_K; flat_idx += block_size) {
                m_tile_thread = fmaxf(m_tile_thread, p_tile[flat_idx]);
            }
            red_buf[tid] = m_tile_thread;
            __syncthreads();
            for (int s = block_size / 2; s > 0; s >>= 1) {
                if (tid < s) red_buf[tid] = fmaxf(red_buf[tid], red_buf[tid + s]);
                __syncthreads();
            }
            float m_tile = red_buf[0];

            // --- Parallel reduction to find tile sum_exp (l_tile) ---
            float l_tile_thread = 0.0f;
            for (int flat_idx = tid; flat_idx < TILE_I * TILE_K; flat_idx += block_size) {
                if (p_tile[flat_idx] > -1e29f) { // Check if it's not padding
                    l_tile_thread += expf(p_tile[flat_idx] - m_tile);
                }
            }
            red_buf[tid] = l_tile_thread;
            __syncthreads();
            for (int s = block_size / 2; s > 0; s >>= 1) {
                if (tid < s) red_buf[tid] += red_buf[tid + s];
                __syncthreads();
            }
            float l_tile = red_buf[0];
            
            // --- Online update of block-wide stats (each thread updates its copy) ---
            float m_new = fmaxf(m_block, m_tile);
            l_block = expf(m_block - m_new) * l_block + expf(m_tile - m_new) * l_tile;
            m_block = m_new;
            __syncthreads(); // Sync to ensure all threads are ready for the next tile
        }
    }
    
    // --- Write final stats to global memory (only thread 0) ---
    if (tid == 0) {
        m_j_out[bh_idx * J + j_idx] = m_block;
        l_j_out[bh_idx * J + j_idx] = l_block;
    }
}


// Aq = softmax_{j,k}(A)
static __global__ void Aq_tiled_softmax(
    const float* __restrict__ Q, const float* __restrict__ R, const float* __restrict__ S,
    float* __restrict__ m_i_out, float* __restrict__ l_i_out,
    int B, int H, int I, int J, int K, int D, float scale
) { 
    // --- Grid/Block Mapping ---
    // Each block computes the stats for a single i_idx, for a single batch/head.
    const int i_idx = blockIdx.x;
    const int bh_idx = blockIdx.y;
    
    if (i_idx >= I) return;

    const int tid = threadIdx.x;
    const int block_size = blockDim.x;

    // --- Pointers ---
    const int64_t q_bh_offset = (int64_t)bh_idx * I * D;
    const int64_t r_bh_offset = (int64_t)bh_idx * J * D;
    const int64_t s_bh_offset = (int64_t)bh_idx * K * D;

    const float* Q_slice = Q + q_bh_offset;
    const float* R_slice = R + r_bh_offset;
    const float* S_slice = S + s_bh_offset;

    // --- Shared Memory ---
    extern __shared__ float smem[];
    float* q_vec_sh = smem;                              // D
    float* r_tile = q_vec_sh + D;                        // TILE_J * D
    float* s_tile = r_tile + TILE_J * D;                 // TILE_K * D
    float* p_tile = s_tile + TILE_K * D;                 // TILE_J * TILE_K
    float* red_buf = p_tile + TILE_J * TILE_K;           // block_size

    // --- Load Q[i] into shared memory ---
    for(int d=tid; d<D; d+=block_size) {
        q_vec_sh[d] = Q_slice[i_idx * D + d];
    }
    
    // Each thread holds a local copy of the block-wide stats
    float m_block = -1e30f;
    float l_block = 0.0f;

    __syncthreads();

    // --- Main Loop: Iterate over all j and k tiles ---
    for (int j0 = 0; j0 < J; j0 += TILE_J) {
        for (int k0 = 0; k0 < K; k0 += TILE_K) {
            // Cooperatively load R and S tiles
            for (int idx = tid; idx < TILE_J * D; idx += block_size) {
                int j_load = j0 + (idx/D);
                if(j_load < J) r_tile[idx] = R_slice[j_load * D + (idx%D)];
            }
            for (int idx = tid; idx < TILE_K * D; idx += block_size) {
                int k_load = k0 + (idx/D);
                if(k_load < K) s_tile[idx] = S_slice[k_load * D + (idx%D)];
            }
            __syncthreads();

            // --- Compute dot products for this tile and store in p_tile ---
            for (int flat_idx = tid; flat_idx < TILE_J * TILE_K; flat_idx += block_size) {
                int j_tile_idx = flat_idx / TILE_K;
                int k_tile_idx = flat_idx % TILE_K;
                
                float dot = 0.0f;
                if (j0 + j_tile_idx < J && k0 + k_tile_idx < K) {
                    const float* r_vec_sh = r_tile + j_tile_idx * D;
                    const float* s_vec_sh = s_tile + k_tile_idx * D;
                    for(int d=0; d<D; ++d) dot += q_vec_sh[d] * r_vec_sh[d] * s_vec_sh[d];
                    p_tile[flat_idx] = dot * scale;
                } else {
                    p_tile[flat_idx] = -1e30f; // Padding
                }
            }
            __syncthreads();

            // --- Parallel reduction to find tile max (m_tile) ---
            float m_tile_thread = -1e30f;
            for (int flat_idx = tid; flat_idx < TILE_J * TILE_K; flat_idx += block_size) {
                m_tile_thread = fmaxf(m_tile_thread, p_tile[flat_idx]);
            }
            red_buf[tid] = m_tile_thread;
            __syncthreads();
            for (int s = block_size / 2; s > 0; s >>= 1) {
                if (tid < s) red_buf[tid] = fmaxf(red_buf[tid], red_buf[tid + s]);
                __syncthreads();
            }
            float m_tile = red_buf[0];

            // --- Parallel reduction to find tile sum_exp (l_tile) ---
            float l_tile_thread = 0.0f;
            for (int flat_idx = tid; flat_idx < TILE_J * TILE_K; flat_idx += block_size) {
                if (p_tile[flat_idx] > -1e29f) { // Check if it's not padding
                    l_tile_thread += expf(p_tile[flat_idx] - m_tile);
                }
            }
            red_buf[tid] = l_tile_thread;
            __syncthreads();
            for (int s = block_size / 2; s > 0; s >>= 1) {
                if (tid < s) red_buf[tid] += red_buf[tid + s];
                __syncthreads();
            }
            float l_tile = red_buf[0];
            
            // --- Online update of block-wide stats (each thread updates its copy) ---
            float m_new = fmaxf(m_block, m_tile);
            l_block = expf(m_block - m_new) * l_block + expf(m_tile - m_new) * l_tile;
            m_block = m_new;
            __syncthreads(); 
        }
    }
    
    // --- Write final stats to global memory (only thread 0) ---
    if (tid == 0) {
        m_i_out[bh_idx * I + i_idx] = m_block;
        l_i_out[bh_idx * I + i_idx] = l_block;
    }
}

// As = softmax_{i,j}(A)
static __global__ void As_tiled_softmax(
    const float* __restrict__ Q, const float* __restrict__ R, const float* __restrict__ S,
    float* __restrict__ m_k_out, float* __restrict__ l_k_out,
    int B, int H, int I, int J, int K, int D, float scale
) {
    // --- Grid/Block Mapping ---
    const int k_idx = blockIdx.x;
    const int bh_idx = blockIdx.y;
    if (k_idx >= K) return;

    const int tid = threadIdx.x;
    const int block_size = blockDim.x;

    // --- Pointers ---
    const int64_t q_bh_offset = (int64_t)bh_idx * I * D;
    const int64_t r_bh_offset = (int64_t)bh_idx * J * D;
    const int64_t s_bh_offset = (int64_t)bh_idx * K * D;

    const float* Q_slice = Q + q_bh_offset;
    const float* R_slice = R + r_bh_offset;
    const float* S_slice = S + s_bh_offset;

    // --- Shared Memory ---
    extern __shared__ float smem[];
    float* s_vec_sh = smem;                              // D
    float* q_tile = s_vec_sh + D;                        // TILE_I * D
    float* r_tile = q_tile + TILE_I * D;                 // TILE_J * D
    float* p_tile = r_tile + TILE_J * D;                 // TILE_I * TILE_J
    float* red_buf = p_tile + TILE_I * TILE_J;           // block_size

    for(int d=tid; d<D; d+=block_size) s_vec_sh[d] = S_slice[k_idx * D + d];
    
    float m_block = -1e30f;
    float l_block = 0.0f;
    __syncthreads();

    // --- Main Loop: Iterate over all i and j tiles ---
    for (int i0 = 0; i0 < I; i0 += TILE_I) {
        for (int j0 = 0; j0 < J; j0 += TILE_J) {
            // Cooperatively load Q and R tiles
            for (int idx = tid; idx < TILE_I * D; idx += block_size) {
                int i_load = i0 + (idx/D);
                if(i_load < I) q_tile[idx] = Q_slice[i_load * D + (idx%D)];
            }
            for (int idx = tid; idx < TILE_J * D; idx += block_size) {
                int j_load = j0 + (idx/D);
                if(j_load < J) r_tile[idx] = R_slice[j_load * D + (idx%D)];
            }
            __syncthreads();

            // Compute dot products
            for (int flat_idx = tid; flat_idx < TILE_I * TILE_J; flat_idx += block_size) {
                int i_tile_idx = flat_idx / TILE_J;
                int j_tile_idx = flat_idx % TILE_J;
                float dot = 0.0f;
                if (i0 + i_tile_idx < I && j0 + j_tile_idx < J) {
                    const float* q_vec_sh = q_tile + i_tile_idx * D;
                    const float* r_vec_sh = r_tile + j_tile_idx * D;
                    for(int d=0; d<D; ++d) dot += q_vec_sh[d] * r_vec_sh[d] * s_vec_sh[d];
                    p_tile[flat_idx] = dot * scale;
                } else {
                    p_tile[flat_idx] = -1e30f;
                }
            }
            __syncthreads();

            // Find tile max (m_tile)
            float m_tile_thread = -1e30f;
            for (int i = tid; i < TILE_I*TILE_J; i+=block_size) m_tile_thread = fmaxf(m_tile_thread, p_tile[i]);
            red_buf[tid] = m_tile_thread;
            __syncthreads();
            for (int s=block_size/2; s>0; s>>=1) if (tid<s) red_buf[tid] = fmaxf(red_buf[tid], red_buf[tid+s]);
            __syncthreads();
            float m_tile = red_buf[0];

            // Find tile sum_exp (l_tile)
            float l_tile_thread = 0.0f;
            for (int i = tid; i < TILE_I*TILE_J; i+=block_size) {
                if(p_tile[i] > -1e29f) l_tile_thread += expf(p_tile[i] - m_tile);
            }
            red_buf[tid] = l_tile_thread;
            __syncthreads();
            for (int s=block_size/2; s>0; s>>=1) if (tid<s) red_buf[tid] += red_buf[tid+s];
            __syncthreads();
            float l_tile = red_buf[0];

            // Online update
            float m_new = fmaxf(m_block, m_tile);
            l_block = expf(m_block - m_new) * l_block + expf(m_tile - m_new) * l_tile;
            m_block = m_new;
            __syncthreads();
        }
    }
    
    if (tid == 0) {
        m_k_out[bh_idx * K + k_idx] = m_block;
        l_k_out[bh_idx * K + k_idx] = l_block;
    }
}


// ============================================================================
// KERNEL IMPLEMENTATION: Tiled grad_Q (Pass 1)
//
// This section contains the CUDA kernel and C++ wrapper for the first pass
// of the flash-attention-style gradient computation for Q. It computes the
// intermediate reduction arrays `sum_q`, `sum_r`, and `sum_s`.
// ============================================================================

// Helper for integer division rounded up
__host__ __device__ inline int ceilDiv(int a, int b) {
    return (a + b - 1) / b;
}

// Device helper for 3-way element-wise product and sum reduction (dot product)
__device__ __forceinline__ float dot3(const float* a, const float* b, const float* c, int D) {
    float sum = 0.0f;
    #pragma unroll
    for (int d = 0; d < D; ++d) {
        sum += a[d] * b[d] * c[d];
    }
    return sum;
}

//
// --- grad_q_pass1_kernel ---
//
// This kernel computes the intermediate sums for the grad_Q calculation.
// Each block processes a tile of size (Bq, Br, Bk) and atomically
// accumulates the results into global memory.
//
template<int Bq, int Br, int Bk>
__global__ void grad_q_pass1_kernel(
    // Inputs
    const float* Q,   const float* R,   const float* S,
    const float* Vq1, const float* Vq2,
    const float* Vr1, const float* Vr2,
    const float* Vs1, const float* Vs2,
    const float* grad_out,
    const float* m_i, const float* l_i,
    const float* m_j, const float* l_j,
    const float* m_k, const float* l_k,
    // Outputs
    float* sum_q, float* sum_r, float* sum_s,
    // Dimensions & Scale
    int N, int D, float scale
) {
    // Shared memory for tensor tiles
    extern __shared__ float smem[];
    float* Qi_smem   = smem;                                  // Bq x D
    float* Rj_smem   = Qi_smem   + Bq * D;                    // Br x D
    float* Sk_smem   = Rj_smem   + Br * D;                    // Bk x D
    float* Vq1i_smem = Sk_smem   + Bk * D;                    // Bq x D
    float* Vq2i_smem = Vq1i_smem + Bq * D;                    // Bq x D
    float* Vr1j_smem = Vq2i_smem + Bq * D;                    // Br x D
    float* Vr2j_smem = Vr1j_smem + Br * D;                    // Br x D
    float* Vs1k_smem = Vr2j_smem + Br * D;                    // Bk x D
    float* Vs2k_smem = Vs1k_smem + Bk * D;                    // Bk x D
    float* dYi_smem  = Vs2k_smem + Bk * D;                    // Bq x D
    float* dYj_smem  = dYi_smem  + Bq * D;                    // Br x D
    float* dYk_smem  = dYj_smem  + Br * D;                    // Bk x D

    // Tile base indices
    const int i0 = blockIdx.x * Bq;
    const int j0 = blockIdx.y * Br;
    const int k0 = blockIdx.z * Bk;

    // Thread indices within the tile
    const int ti = threadIdx.x;
    const int tj = threadIdx.y;
    const int tk = threadIdx.z;

    // Linear thread ID for cooperative loading
    const int lid = tk * (Bq * Br) + tj * Bq + ti;
    const int num_threads = Bq * Br * Bk;

    // Load tiles into shared memory
    // Each loop iterates over a type of tile (i-indexed, j-indexed, k-indexed)
    for (int i = lid; i < Bq * D; i += num_threads) {
        int row = i / D;
        int col = i % D;
        if (i0 + row < N) {
            Qi_smem[i]   = Q[(i0 + row) * D + col];
            Vq1i_smem[i] = Vq1[(i0 + row) * D + col];
            Vq2i_smem[i] = Vq2[(i0 + row) * D + col];
            dYi_smem[i]  = grad_out[(i0 + row) * D + col];
        } else {
            // Zero padding if tile extends beyond N
            Qi_smem[i] = 0.0f; Vq1i_smem[i] = 0.0f; Vq2i_smem[i] = 0.0f; dYi_smem[i] = 0.0f;
        }
    }
    for (int i = lid; i < Br * D; i += num_threads) {
        int row = i / D;
        int col = i % D;
        if (j0 + row < N) {
            Rj_smem[i]   = R[(j0 + row) * D + col];
            Vr1j_smem[i] = Vr1[(j0 + row) * D + col];
            Vr2j_smem[i] = Vr2[(j0 + row) * D + col];
            dYj_smem[i]  = grad_out[(j0 + row) * D + col];
        } else {
            Rj_smem[i] = 0.0f; Vr1j_smem[i] = 0.0f; Vr2j_smem[i] = 0.0f; dYj_smem[i] = 0.0f;
        }
    }
    for (int i = lid; i < Bk * D; i += num_threads) {
        int row = i / D;
        int col = i % D;
        if (k0 + row < N) {
            Sk_smem[i]   = S[(k0 + row) * D + col];
            Vs1k_smem[i] = Vs1[(k0 + row) * D + col];
            Vs2k_smem[i] = Vs2[(k0 + row) * D + col];
            dYk_smem[i]  = grad_out[(k0 + row) * D + col];
        } else {
            Sk_smem[i] = 0.0f; Vs1k_smem[i] = 0.0f; Vs2k_smem[i] = 0.0f; dYk_smem[i] = 0.0f;
        }
    }
    __syncthreads();

    // Global indices for this thread's (i, j, k) point
    const int i = i0 + ti;
    const int j = j0 + tj;
    const int k = k0 + tk;

    if (i < N && j < N && k < N) {
        const float* Qi_vec   = &Qi_smem[ti * D];
        const float* Rj_vec   = &Rj_smem[tj * D];
        const float* Sk_vec   = &Sk_smem[tk * D];
        const float* Vq1i_vec = &Vq1i_smem[ti * D];
        const float* Vq2i_vec = &Vq2i_smem[ti * D];
        const float* Vr1j_vec = &Vr1j_smem[tj * D];
        const float* Vr2j_vec = &Vr2j_smem[tj * D];
        const float* Vs1k_vec = &Vs1k_smem[tk * D];
        const float* Vs2k_vec = &Vs2k_smem[tk * D];
        const float* dYi_vec  = &dYi_smem[ti * D];
        const float* dYj_vec  = &dYj_smem[tj * D];
        const float* dYk_vec  = &dYk_smem[tk * D];

        // ---> Logits & Numerators
        float logits = dot3(Qi_vec, Rj_vec, Sk_vec, D) * scale;
        float Aq_num = expf(logits - m_i[i]);
        float Ar_num = expf(logits - m_j[j]);
        float As_num = expf(logits - m_k[k]);
        float Aq = Aq_num / l_i[i];
        float Ar = Ar_num / l_j[j];
        float As = As_num / l_k[k];

        // ---> Compute grad_A components
        // --- grad_Aq ---
        float gAq  = dot3(dYi_vec, Vr1j_vec, Vs1k_vec, D);
              gAq += dot3(dYj_vec, Vq2i_vec, Vs2k_vec, D) * As;
              gAq += dot3(dYk_vec, Vq2i_vec, Vr2j_vec, D) * Ar;
        // --- grad_Ar ---
        float gAr  = dot3(dYj_vec, Vq1i_vec, Vs1k_vec, D);
              gAr += dot3(dYi_vec, Vr2j_vec, Vs2k_vec, D) * As;
              gAr += dot3(dYk_vec, Vq2i_vec, Vr2j_vec, D) * Aq;
        // --- grad_As ---
        float gAs  = dot3(dYk_vec, Vq1i_vec, Vr1j_vec, D);
              gAs += dot3(dYi_vec, Vr2j_vec, Vs2k_vec, D) * Ar;
              gAs += dot3(dYj_vec, Vq2i_vec, Vs2k_vec, D) * Aq;

        // ---> Accumulate dot products
        atomicAdd(&sum_q[i], gAq * Aq);
        atomicAdd(&sum_r[j], gAr * Ar);
        atomicAdd(&sum_s[k], gAs * As);
    }
}

//
// --- grad_q_pass1_cuda_wrapper ---
//
// C++ wrapper to launch the grad_q_pass1_kernel. It handles tensor slicing
// for batches and heads and allocates the output tensors.
//
std::tuple<torch::Tensor, torch::Tensor, torch::Tensor>
grad_q_pass1_cuda_wrapper(
    torch::Tensor Q, torch::Tensor R, torch::Tensor S,
    torch::Tensor Vq_1, torch::Tensor Vq_2,
    torch::Tensor Vr_1, torch::Tensor Vr_2,
    torch::Tensor Vs_1, torch::Tensor Vs_2,
    torch::Tensor grad_output,
    torch::Tensor m_i, torch::Tensor l_i,
    torch::Tensor m_j, torch::Tensor l_j,
    torch::Tensor m_k, torch::Tensor l_k,
    float scale
) {
    const int B = Q.size(0);
    const int H = Q.size(1);
    const int N = Q.size(2);
    const int D = Q.size(3);

    auto sum_q = torch::zeros({B, H, N}, Q.options());
    auto sum_r = torch::zeros({B, H, N}, Q.options());
    auto sum_s = torch::zeros({B, H, N}, Q.options());

    constexpr int Bq = 8;
    constexpr int Br = 8;
    constexpr int Bk = 8;

    dim3 grid(ceilDiv(N, Bq), ceilDiv(N, Br), ceilDiv(N, Bk));
    dim3 threads(Bq, Br, Bk);

    const size_t shmem_bytes = (4*Bq + 4*Br + 4*Bk) * D * sizeof(float);

    for (int b = 0; b < B; ++b) {
        for (int h = 0; h < H; ++h) {
            grad_q_pass1_kernel<Bq, Br, Bk><<<grid, threads, shmem_bytes, at::cuda::getCurrentCUDAStream()>>>(
                Q.select(0,b).select(0,h).data_ptr<float>(),
                R.select(0,b).select(0,h).data_ptr<float>(),
                S.select(0,b).select(0,h).data_ptr<float>(),
                Vq_1.select(0,b).select(0,h).data_ptr<float>(),
                Vq_2.select(0,b).select(0,h).data_ptr<float>(),
                Vr_1.select(0,b).select(0,h).data_ptr<float>(),
                Vr_2.select(0,b).select(0,h).data_ptr<float>(),
                Vs_1.select(0,b).select(0,h).data_ptr<float>(),
                Vs_2.select(0,b).select(0,h).data_ptr<float>(),
                grad_output.select(0,b).select(0,h).data_ptr<float>(),
                m_i.select(0,b).select(0,h).data_ptr<float>(),
                l_i.select(0,b).select(0,h).data_ptr<float>(),
                m_j.select(0,b).select(0,h).data_ptr<float>(),
                l_j.select(0,b).select(0,h).data_ptr<float>(),
                m_k.select(0,b).select(0,h).data_ptr<float>(),
                l_k.select(0,b).select(0,h).data_ptr<float>(),
                sum_q.select(0,b).select(0,h).data_ptr<float>(),
                sum_r.select(0,b).select(0,h).data_ptr<float>(),
                sum_s.select(0,b).select(0,h).data_ptr<float>(),
                N, D, scale
            );
        }
    }

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
      fprintf(stderr, "CUDA error in grad_q_pass1: %s\n", cudaGetErrorString(err));
    }

  return std::make_tuple(sum_q, sum_r, sum_s);
}

// ==========================================================================
//  grad_Q_pass2_tbIK_kernel  –  second pass of flash-style backward
//  Thread-block geometry:     (i , k) tile   [blockDim = (tileI , tileK)]
// ==========================================================================
template<int BLOCK_I, int BLOCK_J, int BLOCK_K, int REG_CAP = MAX_D_REG>
__global__ void grad_Q_pass2_tbIK_kernel(
    /* ---- operands ---------------------------------------------------- */
    const float* __restrict__ Q,   // [B,H,N,D]
    const float* __restrict__ R,   // [B,H,N,D]
    const float* __restrict__ S,   // [B,H,N,D]
    const float* __restrict__ Vq1, const float* __restrict__ Vq2,
    const float* __restrict__ Vr1, const float* __restrict__ Vr2,
    const float* __restrict__ Vs1, const float* __restrict__ Vs2,
    const float* __restrict__ gradY,   // [B,H,N,D]   (∂L/∂Y)
    /* ---- softmax statistics ----------------------------------------- */
    const float* __restrict__ m_i, const float* __restrict__ l_i,
    const float* __restrict__ m_j, const float* __restrict__ l_j,
    const float* __restrict__ m_k, const float* __restrict__ l_k,
    /* ---- correction scalars from pass-1 ----------------------------- */
    const float* __restrict__ sum_q,
    const float* __restrict__ sum_r,
    const float* __restrict__ sum_s,
    /* ---- output ------------------------------------------------------ */
    float* __restrict__ gradQ,          // [B,H,N,D]
    /* ---- sizes ------------------------------------------------------- */
    int  N, int D, float scale)
{
    /* ---- decode indices --------------------------------------------- */
    const int i0 = blockIdx.x * BLOCK_I + threadIdx.x;   // i-index owned by this lane
    const int k0 = blockIdx.y * BLOCK_K + threadIdx.y;   // k-index owned by this lane
    const int bh = blockIdx.z;                       // flattened (batch, head)

    if (i0 >= N || k0 >= N) return;

    /* ---- per (B,H) base pointers & strides -------------------------- */
    const int64_t stride_BH = (int64_t)N * D;
    const float* Qbh   = Q   + (int64_t)bh * stride_BH;
    const float* Rbh   = R   + (int64_t)bh * stride_BH;
    const float* Sbh   = S   + (int64_t)bh * stride_BH;
    const float* Vq1bh = Vq1 + (int64_t)bh * stride_BH;
    const float* Vq2bh = Vq2 + (int64_t)bh * stride_BH;
    const float* Vr1bh = Vr1 + (int64_t)bh * stride_BH;
    const float* Vr2bh = Vr2 + (int64_t)bh * stride_BH;
    const float* Vs1bh = Vs1 + (int64_t)bh * stride_BH;
    const float* Vs2bh = Vs2 + (int64_t)bh * stride_BH;
    const float* gYbh  = gradY + (int64_t)bh * stride_BH;

    const float* miBH  = m_i + (int64_t)bh * N;
    const float* liBH  = l_i + (int64_t)bh * N;
    const float* mjBH  = m_j + (int64_t)bh * N;
    const float* ljBH  = l_j + (int64_t)bh * N;
    const float* mkBH  = m_k + (int64_t)bh * N;
    const float* lkBH  = l_k + (int64_t)bh * N;

    const float* sum_qBH = sum_q + (int64_t)bh * N;
    const float* sum_rBH = sum_r + (int64_t)bh * N;
    const float* sum_sBH = sum_s + (int64_t)bh * N;

    float* gQbh = gradQ + (int64_t)bh * stride_BH;

    /* ---- registers for i-/k-local vectors --------------------------- */
    float Qi [REG_CAP];
    float Sk [REG_CAP];
    float Vq1i[REG_CAP], Vq2i[REG_CAP];
    float Vs1k[REG_CAP], Vs2k[REG_CAP];
    float dYi [REG_CAP], dYk [REG_CAP];

    #pragma unroll
    for (int d=0; d<D; ++d){
        Qi [d] = Qbh [i0*D + d];
        Sk [d] = Sbh [k0*D + d];
        Vq1i[d] = Vq1bh[i0*D + d];
        Vq2i[d] = Vq2bh[i0*D + d];
        Vs1k[d] = Vs1bh[k0*D + d];
        Vs2k[d] = Vs2bh[k0*D + d];
        dYi [d] = gYbh [i0*D + d];
        dYk [d] = gYbh [k0*D + d];
    }

    const float mi   = miBH [i0];
    const float li   = liBH [i0];
    const float mk   = mkBH [k0];
    const float lk   = lkBH [k0];
    const float sumQi= sum_qBH[i0];
    const float sumSk= sum_sBH[k0];

    /* ---- shared memory for J-tiles ---------------------------------- */
    extern __shared__ float shmem[];
    float* sh_R   = shmem;                    // BLOCK_J × D
    float* sh_Vr1 = sh_R   + BLOCK_J * D;
    float* sh_Vr2 = sh_Vr1 + BLOCK_J * D;
    float* sh_gYj = sh_Vr2 + BLOCK_J * D;
    float* sh_mj  = sh_gYj + BLOCK_J * D;         // BLOCK_J
    float* sh_lj  = sh_mj  + BLOCK_J;             // BLOCK_J
    float* sh_sumr= sh_lj  + BLOCK_J;             // BLOCK_J

    /* ---- accumulator for grad_Q[i,:] -------------------------------- */
    float grad_acc[REG_CAP] = {0.0f};

    /* ---- iterate over J in tiles ------------------------------------ */
    for (int jBase = 0; jBase < N; jBase += BLOCK_J)
    {
        /* cooperative load by all (i,k) threads ----------------------- */
        const int ld = threadIdx.y;           // use y-lane for row loading
        if (ld < BLOCK_J && (jBase + ld) < N){
            const int jGlob = jBase + ld;
            #pragma unroll
            for (int d = threadIdx.x; d < D; d += BLOCK_I){
                sh_R   [ld*D + d] = Rbh  [jGlob*D + d];
                sh_Vr1 [ld*D + d] = Vr1bh[jGlob*D + d];
                sh_Vr2 [ld*D + d] = Vr2bh[jGlob*D + d];
                sh_gYj [ld*D + d] = gYbh [jGlob*D + d];
            }
            if (threadIdx.x == 0){
                sh_mj [ld]  = mjBH  [jGlob];
                sh_lj [ld]  = ljBH  [jGlob];
                sh_sumr[ld] = sum_rBH[jGlob];
            }
        }
        __syncthreads();

        /* ---- loop inside loaded J-tile ------------------------------ */
        for (int jOff = 0; jOff < BLOCK_J && (jBase + jOff) < N; ++jOff){
            /* --- logits & softmax numerators ------------------------ */
            float dot = 0.f;
            #pragma unroll
            for (int d=0; d<D; ++d)
                dot += Qi[d] * sh_R[jOff*D + d] * Sk[d];
            float logits = dot * scale;

            float Aq = __expf(logits - mi        ) / li;
            float Ar = __expf(logits - sh_mj[jOff]) / sh_lj[jOff];
            float As = __expf(logits - mk        ) / lk;

            /* --- grad_Aq ------------------------------------------- */
            float gAq = 0.f;
            #pragma unroll
            for (int d=0; d<D; ++d)
                gAq += dYi[d] * sh_Vr1[jOff*D + d] * Vs1k[d];
            float tmp = 0.f;
            #pragma unroll
            for (int d=0; d<D; ++d)
                tmp += sh_gYj[jOff*D + d] * Vq2i[d] * Vs2k[d];
            gAq += tmp * As;
            tmp = 0.f;
            #pragma unroll
            for (int d=0; d<D; ++d)
                tmp += dYk[d] * Vq2i[d] * sh_Vr2[jOff*D + d];
            gAq += tmp * Ar;

            /* --- grad_Ar ------------------------------------------- */
            float gAr = 0.f;
            #pragma unroll
            for (int d=0; d<D; ++d)
                gAr += sh_gYj[jOff*D + d] * Vq1i[d] * Vs1k[d];
            tmp = 0.f;
            #pragma unroll
            for (int d=0; d<D; ++d)
                tmp += dYi[d] * sh_Vr2[jOff*D + d] * Vs2k[d];
            gAr += tmp * As;
            tmp = 0.f;
            #pragma unroll
            for (int d=0; d<D; ++d)
                tmp += dYk[d] * Vq2i[d] * sh_Vr2[jOff*D + d];
            gAr += tmp * Aq;

            /* --- grad_As ------------------------------------------- */
            float gAs = 0.f;
            #pragma unroll
            for (int d=0; d<D; ++d)
                gAs += dYk[d] * Vq1i[d] * sh_Vr1[jOff*D + d];
            tmp = 0.f;
            #pragma unroll
            for (int d=0; d<D; ++d)
                tmp += dYi[d] * sh_Vr2[jOff*D + d] * Vs2k[d];
            gAs += tmp * Ar;
            tmp = 0.f;
            #pragma unroll
            for (int d=0; d<D; ++d)
                tmp += sh_gYj[jOff*D + d] * Vq2i[d] * Vs2k[d];
            gAs += tmp * Aq;

            /* --- Jacobian-corrected grad_A ------------------------- */
            float grad_A = (gAq -  sumQi          ) * Aq
                         + (gAr -  sh_sumr[jOff]  ) * Ar
                         + (gAs -  sumSk          ) * As;

            /* --- accumulate contribution to grad_Q[i,:] ------------ */
            #pragma unroll
            for (int d=0; d<D; ++d)
                grad_acc[d] += grad_A * sh_R[jOff*D + d] * Sk[d];
        }
        __syncthreads();
    }

    /* ---- atomic add to global grad_Q ------------------------------- */
    #pragma unroll
    for (int d=0; d<D; ++d)
        atomicAdd(&gQbh[i0*D + d], scale * grad_acc[d]);
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor,
          torch::Tensor, torch::Tensor,
          torch::Tensor, torch::Tensor,
          torch::Tensor, torch::Tensor>
backward_cuda(
    torch::Tensor grad_output, 
    torch::Tensor Q,           
    torch::Tensor R,           
    torch::Tensor S,           
    torch::Tensor Vq_1,        
    torch::Tensor Vq_2,        
    torch::Tensor Vr_1,        
    torch::Tensor Vr_2,        
    torch::Tensor Vs_1,        
    torch::Tensor Vs_2,        
    double dropout_rate)
{
  // ============================================================================
  // 1. ENSURE ALL TENSORS ARE CONTIGUOUS
  // ============================================================================
  grad_output = grad_output.contiguous();
  Q = Q.contiguous();  
  R = R.contiguous();  
  S = S.contiguous();
  Vq_1 = Vq_1.contiguous();
  Vq_2 = Vq_2.contiguous(); 
  Vr_1 = Vr_1.contiguous();
  Vr_2 = Vr_2.contiguous(); 
  Vs_1 = Vs_1.contiguous();
  Vs_2 = Vs_2.contiguous(); 

  // ============================================================================
  // 2. ALLOCATE OUTPUT GRADIENT TENSORS
  // ============================================================================
  auto grad_Q    = torch::zeros_like(Q);
  auto grad_R    = torch::zeros_like(R);
  auto grad_S    = torch::zeros_like(S);
  auto grad_Vq_1 = torch::zeros_like(Vq_1);
  auto grad_Vq_2 = torch::zeros_like(Vq_2); 
  auto grad_Vr_1 = torch::zeros_like(Vr_1);
  auto grad_Vr_2 = torch::zeros_like(Vr_2);
  auto grad_Vs_1 = torch::zeros_like(Vs_1);
  auto grad_Vs_2 = torch::zeros_like(Vs_2);
  
  // ============================================================================
  // 3. EXTRACT DIMENSIONS AND COMPUTE SCALE
  // ============================================================================
  const int B      = Q.size(0);
  const int H      = Q.size(1);
  const int N      = Q.size(2);
  const int I      = Q.size(2);
  const int J      = R.size(2);
  const int K      = S.size(2);
  const int D      = Q.size(3);
  const int N_grad = grad_output.size(2); 
  const float scale = 1.0f / sqrtf((float)D);

  // ============================================================================
  // 4. ALLOCATE SOFTMAX STATISTICS TENSORS (m_i, l_i, m_j, l_j, m_k, l_k)
  // ============================================================================
  auto optionsBH = Q.options();
  auto m_i = torch::empty({B, H, N}, optionsBH);
  auto l_i = torch::empty({B, H, N}, optionsBH);
  auto m_j = torch::empty({B, H, N}, optionsBH);
  auto l_j = torch::empty({B, H, N}, optionsBH);
  auto m_k = torch::empty({B, H, N}, optionsBH);
  auto l_k = torch::empty({B, H, N}, optionsBH);

  // ============================================================================
  // 5. COMPUTE SOFTMAX STATISTICS: Aq (i-centric), Ar (j-centric), As (k-centric)
  // ============================================================================
constexpr int block_threads = 256;

  // --- Aq (i-centric) stats ---
  {
const size_t shmem_bytes_Aq =
(D + TILE_J*D + TILE_K*D + TILE_J*TILE_K + block_threads) * sizeof(float);
    dim3 aqBlocks(I, B*H);
dim3 aqThreads(block_threads);

Aq_tiled_softmax<<<aqBlocks, aqThreads, shmem_bytes_Aq,
             at::cuda::getCurrentCUDAStream()>>>(
Q.data_ptr<float>(), R.data_ptr<float>(), S.data_ptr<float>(),
m_i.data_ptr<float>(), l_i.data_ptr<float>(),
B, H, N, J, K, D, scale);

cudaError_t err = cudaGetLastError();
if (err != cudaSuccess) {
    fprintf(stderr, "CUDA error in Aq_tiled_softmax: %s\n", cudaGetErrorString(err));
}
  }
    
  // --- Ar (j-centric) stats ---
  {
  const size_t shmem_bytes_Ar =
        (D                                  /* r_vec */
         + TILE_I*D + TILE_K*D              /* q_tile + s_tile */
         + TILE_I*TILE_K                    /* p_tile */
         + block_threads) * sizeof(float);  /* red_buf */
    dim3 arBlocks(J, B*H);
    dim3 arThreads(block_threads);

    Ar_tiled_softmax<<<arBlocks, arThreads, shmem_bytes_Ar,
                       at::cuda::getCurrentCUDAStream()>>>(
        Q.data_ptr<float>(), R.data_ptr<float>(), S.data_ptr<float>(),
        m_j.data_ptr<float>(), l_j.data_ptr<float>(),
        B, H, N, J, K, D, scale);

    cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) {
    fprintf(stderr, "CUDA error in Ar_tiled_softmax: %s\n", cudaGetErrorString(err));
    }
  }

  // --- As (k-centric) stats ---
  {
  const size_t shmem_bytes_As =
  (D                                  /* s_vec */
         + TILE_I*D + TILE_J*D              /* q_tile + r_tile */
         + TILE_I*TILE_J                    /* p_tile */
         + block_threads) * sizeof(float);  /* red_buf */
    dim3 asBlocks(K, B*H);
dim3 asThreads(block_threads);

    As_tiled_softmax<<<asBlocks, asThreads, shmem_bytes_As,
                       at::cuda::getCurrentCUDAStream()>>>(
Q.data_ptr<float>(), R.data_ptr<float>(), S.data_ptr<float>(),
m_k.data_ptr<float>(), l_k.data_ptr<float>(),
        B, H, N, J, K, D, scale);

    cudaError_t err = cudaGetLastError();
if (err != cudaSuccess) {
    fprintf(stderr, "CUDA error in As_tiled_softmax: %s\n", cudaGetErrorString(err));
}
  }

    // ============================================================================
  // 5.5 COMPUTE grad_Q INTERMEDIATES (pass 1 of flash-style backward)
  // TODO: Use these sums in a second pass to compute grad_Q.
  // ============================================================================
  auto sum_q = torch::zeros({B, H, N}, Q.options());
  auto sum_r = torch::zeros({B, H, N}, Q.options());
  auto sum_s = torch::zeros({B, H, N}, Q.options());
  {
    constexpr int Bq = 8;
    constexpr int Br = 8;
    constexpr int Bk = 8;

    dim3 grid(ceilDiv(N, Bq), ceilDiv(N, Br), ceilDiv(N, Bk));
    dim3 threads(Bq, Br, Bk);

    const size_t shmem_bytes = (
        Bq*D + Br*D + Bk*D +  // Q, R, S
        Bq*D + Bq*D +        // Vq1, Vq2
        Br*D + Br*D +        // Vr1, Vr2
        Bk*D + Bk*D +        // Vs1, Vs2
        Bq*D + Br*D + Bk*D   // dY
    ) * sizeof(float);

    for (int b = 0; b < B; ++b) {
        for (int h = 0; h < H; ++h) {
            grad_q_pass1_kernel<Bq, Br, Bk><<<grid, threads, shmem_bytes, at::cuda::getCurrentCUDAStream()>>>(
                Q.select(0,b).select(0,h).data_ptr<float>(),
                R.select(0,b).select(0,h).data_ptr<float>(),
                S.select(0,b).select(0,h).data_ptr<float>(),
                Vq_1.select(0,b).select(0,h).data_ptr<float>(),
                Vq_2.select(0,b).select(0,h).data_ptr<float>(),
                Vr_1.select(0,b).select(0,h).data_ptr<float>(),
                Vr_2.select(0,b).select(0,h).data_ptr<float>(),
                Vs_1.select(0,b).select(0,h).data_ptr<float>(),
                Vs_2.select(0,b).select(0,h).data_ptr<float>(),
                grad_output.select(0,b).select(0,h).data_ptr<float>(),
                m_i.select(0,b).select(0,h).data_ptr<float>(),
                l_i.select(0,b).select(0,h).data_ptr<float>(),
                m_j.select(0,b).select(0,h).data_ptr<float>(),
                l_j.select(0,b).select(0,h).data_ptr<float>(),
                m_k.select(0,b).select(0,h).data_ptr<float>(),
                l_k.select(0,b).select(0,h).data_ptr<float>(),
                sum_q.select(0,b).select(0,h).data_ptr<float>(),
                sum_r.select(0,b).select(0,h).data_ptr<float>(),
                sum_s.select(0,b).select(0,h).data_ptr<float>(),
                N, D, scale
            );
        }
    }
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
      fprintf(stderr, "CUDA error in grad_q_pass1 launch: %s\n", cudaGetErrorString(err));
    }
  }

  // ============================================================================
// 5.6  COMPUTE grad_Q  (flash-style second pass, tile-by-I,K kernel)
// ============================================================================
{
    constexpr int tileI = TILE_I;   // reuse existing macro defaults (8)
    constexpr int tileK = TILE_K;
    constexpr int tileJ = 16;       // independent knob for J streaming

    TORCH_CHECK(D <= MAX_D_REG,
                "grad_Q_pass2_tbIK_kernel requires D <= ", MAX_D_REG,
                ", but got D = ", D);

    dim3 blockDim(tileI, tileK);                    // (threads.x , threads.y)
    dim3 gridDim( (N + tileI - 1) / tileI,
                  (N + tileK - 1) / tileK,
                  B * H );

    const size_t shmem_bytes =
        /* R, Vr1, Vr2, gYj */ 4 * tileJ * D * sizeof(float) +
        /* mj, lj, sum_r   */ 3 * tileJ * sizeof(float);

    grad_Q_pass2_tbIK_kernel<tileI, tileJ, tileK><<<gridDim, blockDim,
        shmem_bytes, at::cuda::getCurrentCUDAStream()>>>(
        /* ---- tensor arguments (all .data_ptr<float>()) ---- */
        Q.data_ptr<float>(), R.data_ptr<float>(), S.data_ptr<float>(),
        Vq_1.data_ptr<float>(), Vq_2.data_ptr<float>(),
        Vr_1.data_ptr<float>(), Vr_2.data_ptr<float>(),
        Vs_1.data_ptr<float>(), Vs_2.data_ptr<float>(),
        grad_output.data_ptr<float>(),
        m_i.data_ptr<float>(), l_i.data_ptr<float>(),
        m_j.data_ptr<float>(), l_j.data_ptr<float>(),
        m_k.data_ptr<float>(), l_k.data_ptr<float>(),
        sum_q.data_ptr<float>(), sum_r.data_ptr<float>(),
        sum_s.data_ptr<float>(),
        grad_Q.data_ptr<float>(),
        N, D, scale);

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess){
        fprintf(stderr,"CUDA error in grad_Q_pass2 launch: %s\n",
                cudaGetErrorString(err));
    }
}

  // ============================================================================
  // 6. COMPUTE grad_Vq_1 (GATHER-STYLE KERNEL)
  // ============================================================================
  {
  constexpr int TI = T_I;
  constexpr int TK = T_K;
    dim3 blockDim(TI, TK);
    dim3 gridDim((N+TI-1)/TI, (N+TK-1)/TK, B*H);
    size_t shmem_bytes = T_J * D * 3 * sizeof(float);  // R + Vr + gradY

  grad_Vq1_tbIK_kernel<<<gridDim, blockDim, shmem_bytes,
  at::cuda::getCurrentCUDAStream()>>>(
Q.data_ptr<float>(), R.data_ptr<float>(), S.data_ptr<float>(),
Vs_1.data_ptr<float>(), Vr_1.data_ptr<float>(),
grad_output.data_ptr<float>(),
m_j.data_ptr<float>(), l_j.data_ptr<float>(),
m_k.data_ptr<float>(), l_k.data_ptr<float>(),
grad_Vq_1.data_ptr<float>(),
N, D, scale);
  }

  // ============================================================================
  // 7. COMPUTE grad_Vq_2 (SCATTER-STYLE KERNEL)
  // ============================================================================
{
    constexpr int TI = T_I;
    constexpr int TK = T_K;
    dim3 blockDim(TI, TK);
    dim3 gridDim((N+TI-1)/TI, (N+TK-1)/TK, B*H);
    size_t shmem_bytes_vq2 = T_J * D * 3 * sizeof(float) + T_J * 2 * sizeof(float);

    grad_Vq2_tbIK_kernel<<<gridDim, blockDim, shmem_bytes_vq2,
                           at::cuda::getCurrentCUDAStream()>>>(
        Q.data_ptr<float>(), R.data_ptr<float>(), S.data_ptr<float>(),
        Vr_2.data_ptr<float>(), Vs_2.data_ptr<float>(),
        grad_output.data_ptr<float>(),
        m_i.data_ptr<float>(), l_i.data_ptr<float>(),
        m_j.data_ptr<float>(), l_j.data_ptr<float>(),
        m_k.data_ptr<float>(), l_k.data_ptr<float>(),
        grad_Vq_2.data_ptr<float>(),
        N, D, scale);
}

  // ============================================================================
  // 8. COMPUTE REMAINING V GRADIENTS (grad_Vr_1, grad_Vs_1, grad_Vr_2, grad_Vs_2)
  //    Using slice-by-slice approach with A materialization
  // ============================================================================
  const int threads = 256; 
  auto grad_A_batched_gpu = torch::zeros({B, H, I, J, K}, Q.options()); 

  for (int b = 0; b < B; ++b) {
      for (int h = 0; h < H; ++h) {
      // --- Extract slices for current (b,h) ---
      auto Q_slice_gpu           = Q.select(0, b).select(0, h);
      auto R_slice_gpu           = R.select(0, b).select(0, h);
      auto S_slice_gpu           = S.select(0, b).select(0, h);
          auto grad_output_slice_gpu = grad_output.select(0, b).select(0, h);
      auto Vq_1_slice_gpu        = Vq_1.select(0, b).select(0, h);
      auto Vq_2_slice_gpu        = Vq_2.select(0, b).select(0, h);
      auto Vr_1_slice_gpu        = Vr_1.select(0, b).select(0, h);
      auto Vr_2_slice_gpu        = Vr_2.select(0, b).select(0, h);
      auto Vs_1_slice_gpu        = Vs_1.select(0, b).select(0, h);
      auto Vs_2_slice_gpu        = Vs_2.select(0, b).select(0, h);
        
      // --- Compute A and its marginals (Aq, Ar, As) ---
      torch::Tensor A_slice_gpu  = compute_A_slice_cuda_wrapper(
          Q_slice_gpu, R_slice_gpu, S_slice_gpu, scale);
          torch::Tensor Aq_slice_gpu = compute_Aq_slice_cuda_wrapper(A_slice_gpu);
          torch::Tensor Ar_slice_gpu = compute_Ar_slice_cuda_wrapper(A_slice_gpu);
          torch::Tensor As_slice_gpu = compute_As_slice_cuda_wrapper(A_slice_gpu);

      // --- Compute grad_Vr_1 (gather-style) ---
          {
              auto gradVr1_slice = grad_Vr_1.select(0, b).select(0, h);
              const int64_t N_kernel = (int64_t)J * D;
              const dim3 blocks((N_kernel + threads - 1) / threads);
              gather_grad_Vr1_kernel_optimized<<<blocks, threads>>>(
                  grad_output_slice_gpu.data_ptr<float>(),
                  Vq_1_slice_gpu.data_ptr<float>(),
                  Vs_1_slice_gpu.data_ptr<float>(),
                  Aq_slice_gpu.data_ptr<float>(),
                  As_slice_gpu.data_ptr<float>(),
                  gradVr1_slice.data_ptr<float>(),
            I, J, K, D, N_grad);
          }

      // --- Compute grad_Vs_1 (gather-style) ---
          {
              auto gradVs1_slice = grad_Vs_1.select(0, b).select(0, h);
              const int64_t N_kernel = (int64_t)K * D;
              const dim3 blocks((N_kernel + threads - 1) / threads);
              gather_grad_Vs1_kernel_optimized<<<blocks, threads>>>(
                  grad_output_slice_gpu.data_ptr<float>(),
                  Vq_1_slice_gpu.data_ptr<float>(),
                  Vr_1_slice_gpu.data_ptr<float>(),
                  Aq_slice_gpu.data_ptr<float>(),
                  Ar_slice_gpu.data_ptr<float>(),
                  gradVs1_slice.data_ptr<float>(),
            I, J, K, D, N_grad);
          }
          
      // --- Compute grad_Vr_2 (scatter-style) ---
          {
              auto gradVr2_slice = grad_Vr_2.select(0, b).select(0, h);
              const int64_t N_kernel = (int64_t)J * D;
              const dim3 blocks((N_kernel + threads - 1) / threads);
              scatter_grad_Vr2_kernel_optimized<<<blocks, threads>>>(
                  grad_output_slice_gpu.data_ptr<float>(),
            Aq_slice_gpu.data_ptr<float>(),
            Ar_slice_gpu.data_ptr<float>(),
            As_slice_gpu.data_ptr<float>(),
            Vq_2_slice_gpu.data_ptr<float>(),
            Vs_2_slice_gpu.data_ptr<float>(),
                  gradVr2_slice.data_ptr<float>(),
            I, J, K, D, N_grad);
          }

      // --- Compute grad_Vs_2 (scatter-style) ---
          {
              auto gradVs2_slice = grad_Vs_2.select(0, b).select(0, h);
              const int64_t N_kernel = (int64_t)K * D;
              const dim3 blocks((N_kernel + threads - 1) / threads);
              scatter_grad_Vs2_kernel_optimized<<<blocks, threads>>>(
                  grad_output_slice_gpu.data_ptr<float>(),
            Aq_slice_gpu.data_ptr<float>(),
            Ar_slice_gpu.data_ptr<float>(),
            As_slice_gpu.data_ptr<float>(),
            Vq_2_slice_gpu.data_ptr<float>(),
            Vr_2_slice_gpu.data_ptr<float>(),
                  gradVs2_slice.data_ptr<float>(),
            I, J, K, D, N_grad);
          }

      // --- Compute interim gradients for Aq, Ar, As ---
          torch::Tensor grad_Aq_slice_gpu_tmp, grad_Ar_slice_gpu_tmp, grad_As_slice_gpu_tmp;
          std::tie(grad_Aq_slice_gpu_tmp, grad_Ar_slice_gpu_tmp, grad_As_slice_gpu_tmp) = 
              compute_interim_grads_cuda_wrapper(
                  grad_output_slice_gpu, 
                  Vq_1_slice_gpu, Vq_2_slice_gpu, 
                  Vr_1_slice_gpu, Vr_2_slice_gpu, 
                  Vs_1_slice_gpu, Vs_2_slice_gpu, 
                  Aq_slice_gpu, Ar_slice_gpu, As_slice_gpu);
          
      // --- Apply softmax backward to get final grad_A ---
          torch::Tensor final_grad_A_slice_gpu = apply_softmax_backward_cuda_wrapper(
              grad_Aq_slice_gpu_tmp, grad_Ar_slice_gpu_tmp, grad_As_slice_gpu_tmp,
          Aq_slice_gpu, Ar_slice_gpu, As_slice_gpu);

      // --- Copy result into batched grad_A tensor ---
          grad_A_batched_gpu.select(0, b).select(0, h).copy_(final_grad_A_slice_gpu);
      }
  }

  // Error check after grad_A computation loop
  cudaError_t ga_err = cudaGetLastError();
  if (ga_err != cudaSuccess) {
    fprintf(stderr, "CUDA error after grad_A loop in backward_cuda: %s\n",
            cudaGetErrorString(ga_err));
  }

  // ============================================================================
  // 9. COMPUTE grad_Q, grad_R, grad_S FROM grad_A
  // ============================================================================
  // COMMENTED OUT: Returning zeros for grad_Q, grad_R, grad_S to allow kernel rewrite
  //
  // {
  //   const int64_t N_kernel_Q = (int64_t)B * H * I * D;
  //   const dim3 blocks_Q((N_kernel_Q + threads - 1) / threads);
  //   grad_Q_kernel<<<blocks_Q, threads>>>(
  //       grad_A_batched_gpu.data_ptr<float>(),
  //       R.data_ptr<float>(),
  //       S.data_ptr<float>(),
  //       grad_Q.data_ptr<float>(),
  //       B, H, I, J, K, D, scale);
  // }
  //
  // {
  //   const int64_t N_kernel_R = (int64_t)B * H * J * D;
  //   const dim3 blocks_R((N_kernel_R + threads - 1) / threads);
  //   grad_R_kernel<<<blocks_R, threads>>>(
  //       grad_A_batched_gpu.data_ptr<float>(),
  //       Q.data_ptr<float>(),
  //       S.data_ptr<float>(),
  //       grad_R.data_ptr<float>(),
  //       B, H, I, J, K, D, scale);
  // }
  //
  // {
  //   const int64_t N_kernel_S = (int64_t)B * H * K * D;
  //   const dim3 blocks_S((N_kernel_S + threads - 1) / threads);
  //   grad_S_kernel<<<blocks_S, threads>>>(
  //       grad_A_batched_gpu.data_ptr<float>(),
  //       Q.data_ptr<float>(),
  //       R.data_ptr<float>(),
  //       grad_S.data_ptr<float>(),
  //       B, H, I, J, K, D, scale);
  // }

  // ============================================================================
  // 10. SYNCHRONIZE AND RETURN GRADIENTS
  // ============================================================================

  cudaDeviceSynchronize(); 

  return std::make_tuple(
      grad_Q, grad_R, grad_S,
      grad_Vq_1, grad_Vq_2, 
      grad_Vr_1, grad_Vr_2,
      grad_Vs_1, grad_Vs_2
  );
}

