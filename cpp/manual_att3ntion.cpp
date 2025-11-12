#include <torch/extension.h> 
#include <iostream> 
#include <cmath> 
#include <limits>
#include <vector> 
#include <tuple>  
#include <iomanip> 
#include <cuda_runtime.h> 
#include "manual_att3ntion.h"
// Forward declarations for CUDA 
std::tuple<torch::Tensor, torch::Tensor, torch::Tensor,
          torch::Tensor, torch::Tensor,
          torch::Tensor>
forward_cuda(
    torch::Tensor Q, torch::Tensor R, torch::Tensor S,
    torch::Tensor Vq_1, torch::Tensor Vq_2,
    torch::Tensor Vr_1, torch::Tensor Vr_2,
    torch::Tensor Vs_1, torch::Tensor Vs_2,
    double dropout_rate = 0.0);

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor,
          torch::Tensor, torch::Tensor,
          torch::Tensor, torch::Tensor,
          torch::Tensor, torch::Tensor>
backward_cuda(
    torch::Tensor grad_output,
    torch::Tensor Q, torch::Tensor R, torch::Tensor S,
    torch::Tensor Vq_1, torch::Tensor Vq_2,
    torch::Tensor Vr_1, torch::Tensor Vr_2,
    torch::Tensor Vs_1, torch::Tensor Vs_2,
    double dropout_rate = 0.0);


// Forward pass
// helper: returns dot product between three vectors at specific indices
template <typename T>
inline float compute_dot_product(
    const T& Q_acc,
    const T& R_acc,
    const T& S_acc,
    int b, int h, int i, int j, int k, int D) {
    
    float dot = 0.0f;
    for (int d = 0; d < D; d++) {
        dot += Q_acc[b][h][i][d] * R_acc[b][h][j][d] * S_acc[b][h][k][d];
    }
    return dot;
}
// helper: compute 3D softmax with configurable dimensions
template <typename T>
void compute_softmax_3d(
    const T& Q_acc,
    const T& R_acc,
    const T& S_acc,
    int b, int h, 
    int fixed_dim, int fixed_idx,
    int dim1_size, int dim2_size,
    int dim1_idx_fn, int dim2_idx_fn, int fixed_dim_idx_fn,
    float scale, float* attn_out) {
    
    // Find max for numerical stability
    float max_val = -std::numeric_limits<float>::infinity();
    for (int idx1 = 0; idx1 < dim1_size; idx1++) {
        for (int idx2 = 0; idx2 < dim2_size; idx2++) {
            int i = fixed_dim == 0 ? fixed_idx : (dim1_idx_fn == 0 ? idx1 : (dim2_idx_fn == 0 ? idx2 : 0));
            int j = fixed_dim == 1 ? fixed_idx : (dim1_idx_fn == 1 ? idx1 : (dim2_idx_fn == 1 ? idx2 : 0));
            int k = fixed_dim == 2 ? fixed_idx : (dim1_idx_fn == 2 ? idx1 : (dim2_idx_fn == 2 ? idx2 : 0));
            
            float dot = compute_dot_product(Q_acc, R_acc, S_acc, b, h, i, j, k, Q_acc.size(3));
            dot *= scale;
            if (dot > max_val) max_val = dot;
        }
    }
    
    // Compute sum of exponentials
    float sum_exp = 0.0f;
    for (int idx1 = 0; idx1 < dim1_size; idx1++) {
        for (int idx2 = 0; idx2 < dim2_size; idx2++) {
            int i = fixed_dim == 0 ? fixed_idx : (dim1_idx_fn == 0 ? idx1 : (dim2_idx_fn == 0 ? idx2 : 0));
            int j = fixed_dim == 1 ? fixed_idx : (dim1_idx_fn == 1 ? idx1 : (dim2_idx_fn == 1 ? idx2 : 0));
            int k = fixed_dim == 2 ? fixed_idx : (dim1_idx_fn == 2 ? idx1 : (dim2_idx_fn == 2 ? idx2 : 0));
            
            float dot = compute_dot_product(Q_acc, R_acc, S_acc, b, h, i, j, k, Q_acc.size(3));
            dot *= scale;
            sum_exp += std::exp(dot - max_val);
        }
    }
    
    // Compute softmax values for each position
    int idx = 0;
    for (int idx1 = 0; idx1 < dim1_size; idx1++) {
        for (int idx2 = 0; idx2 < dim2_size; idx2++) {
            int i = fixed_dim == 0 ? fixed_idx : (dim1_idx_fn == 0 ? idx1 : (dim2_idx_fn == 0 ? idx2 : 0));
            int j = fixed_dim == 1 ? fixed_idx : (dim1_idx_fn == 1 ? idx1 : (dim2_idx_fn == 1 ? idx2 : 0));
            int k = fixed_dim == 2 ? fixed_idx : (dim1_idx_fn == 2 ? idx1 : (dim2_idx_fn == 2 ? idx2 : 0));
            
            float dot = compute_dot_product(Q_acc, R_acc, S_acc, b, h, i, j, k, Q_acc.size(3));
            dot *= scale;
            attn_out[idx++] = std::exp(dot - max_val) / sum_exp;
        }
    }
}

void compute_Y_gather(
    torch::Tensor& Y_out,
    const torch::Tensor& Q,
    const torch::Tensor& R,
    const torch::Tensor& S,
    const torch::Tensor& V1,
    const torch::Tensor& V2,
    int fixed_dim
) {
    auto Q_acc = Q.accessor<float, 4>();
    auto R_acc = R.accessor<float, 4>();
    auto S_acc = S.accessor<float, 4>();
    auto V1_acc = V1.accessor<float, 4>();
    auto V2_acc = V2.accessor<float, 4>();
    auto Y_out_acc = Y_out.accessor<float, 4>();

    int B = Q.size(0);
    int H = Q.size(1);
    int I = Q.size(2);
    int J = R.size(2);
    int K = S.size(2);
    int D = Q.size(3);
    float scale = 1.0f / std::sqrt(static_cast<float>(D));
    int sizes[3] = {I, J, K};

    for (int b = 0; b < B; b++) {
        for (int h = 0; h < H; h++) {
            for (int fixed_idx = 0; fixed_idx < sizes[fixed_dim]; fixed_idx++) {
                int dim1, dim2;
                int dim1_size, dim2_size;
                if (fixed_dim == 0) { dim1 = 1; dim2 = 2; dim1_size = J; dim2_size = K; }
                else if (fixed_dim == 1) { dim1 = 0; dim2 = 2; dim1_size = I; dim2_size = K; }
                else { dim1 = 0; dim2 = 1; dim1_size = I; dim2_size = J; }

                
                for (int d = 0; d < D; d++) {
                    Y_out_acc[b][h][fixed_idx][d] = 0.0f;
                    float max_val = -std::numeric_limits<float>::infinity();

                    // compute max dot product used for numerical stability
                    for (int idx1 = 0; idx1 < dim1_size; idx1++) {
                        for (int idx2 = 0; idx2 < dim2_size; idx2++) {
                            int i = (fixed_dim == 0) ? fixed_idx : (dim1 == 0) ? idx1 : idx2;
                            int j = (fixed_dim == 1) ? fixed_idx : (dim1 == 1) ? idx1 : idx2;
                            int k = (fixed_dim == 2) ? fixed_idx : (dim1 == 2) ? idx1 : idx2;
                            float dot_prod = compute_dot_product(Q_acc, R_acc, S_acc, b, h, i, j, k, D);
                            dot_prod *= scale;
                            max_val = std::max(max_val, dot_prod);
                        }
                    }

                    // compute denominator for softmax
                    float sum_exp = 0.0f;
                    for (int idx1 = 0; idx1 < dim1_size; idx1++) {
                        for (int idx2 = 0; idx2 < dim2_size; idx2++) {
                            int i = (fixed_dim == 0) ? fixed_idx : (dim1 == 0) ? idx1 : idx2;
                            int j = (fixed_dim == 1) ? fixed_idx : (dim1 == 1) ? idx1 : idx2;
                            int k = (fixed_dim == 2) ? fixed_idx : (dim1 == 2) ? idx1 : idx2;
                            float dot_prod = compute_dot_product(Q_acc, R_acc, S_acc, b, h, i, j, k, D);
                            dot_prod *= scale;
                            sum_exp += std::exp(dot_prod - max_val);
                        }
                    }

                    // compute actual gather
                    for (int idx1 = 0; idx1 < dim1_size; idx1++) {
                        for (int idx2 = 0; idx2 < dim2_size; idx2++) {
                            int i = (fixed_dim == 0) ? fixed_idx : (dim1 == 0) ? idx1 : idx2;
                            int j = (fixed_dim == 1) ? fixed_idx : (dim1 == 1) ? idx1 : idx2;
                            int k = (fixed_dim == 2) ? fixed_idx : (dim1 == 2) ? idx1 : idx2;
                            int v1_idx = (dim1 == 0) ? i : (dim1 == 1) ? j : k;
                            int v2_idx = (dim2 == 0) ? i : (dim2 == 1) ? j : k;
                            float dot_prod = compute_dot_product(Q_acc, R_acc, S_acc, b, h, i, j, k, D);
                            dot_prod *= scale;
                            float attn = (sum_exp > 1e-9) ? (std::exp(dot_prod - max_val) / sum_exp) : 0.0f;
                            Y_out_acc[b][h][fixed_idx][d] += attn * V1_acc[b][h][v1_idx][d] * V2_acc[b][h][v2_idx][d];
                        }
                    }
                }
            }
        }
    }
}

void compute_Y_scatter_q(
    torch::Tensor& Y_q,
    const torch::Tensor& Q,
    const torch::Tensor& R,
    const torch::Tensor& S,
    const torch::Tensor& Vr_2,
    const torch::Tensor& Vs_2
) {
    auto Q_acc = Q.accessor<float, 4>();  
    auto R_acc = R.accessor<float, 4>();  
    auto S_acc = S.accessor<float, 4>();  
    auto Vr_2_acc = Vr_2.accessor<float, 4>();  
    auto Vs_2_acc = Vs_2.accessor<float, 4>();  
    auto Y_q_acc = Y_q.accessor<float, 4>();

    int B = Q.size(0);
    int H = Q.size(1);
    int I = Q.size(2);
    int J = R.size(2);
    int K = S.size(2);
    int D = Q.size(3);
    float scale = 1.0f / std::sqrt(static_cast<float>(D));

    for (int b = 0; b < B; b++) {
        for (int h = 0; h < H; h++) {
            for (int i = 0; i < I; i++) {
                //Compute the effect of ArAs * Vr_2 * Vs_2: For each j, softmax over i,k.
                std::vector<std::vector<float>> Ar_values(J, std::vector<float>(K, 0.0f));

                for (int j = 0; j < J; j++) {
                    std::vector<float> softmax_results(I * K); 
                    compute_softmax_3d(
                        Q_acc, R_acc, S_acc,
                        b, h,
                        1, j,  // fixed_dim
                        I, K,  // variable dims
                        0, 2, 1, 
                        scale, softmax_results.data()
                    );
                    
                    for (int k = 0; k < K; k++) {
                        int idx = i * K + k; // Find the index of (i,k) in the softmax_results array
                        Ar_values[j][k] = softmax_results[idx];
                    }
                }

                std::vector<std::vector<float>> As_values(K, std::vector<float>(J, 0.0f));
                for (int k = 0; k < K; k++) {
                    // Compute As values (softmax over i,j for fixed k)
                    std::vector<float> softmax_results(I * J);
                    compute_softmax_3d(
                        Q_acc, R_acc, S_acc,
                        b, h,
                        2, k,  
                        I, J,  
                        0, 1, 2,  
                        scale, softmax_results.data()
                    );
                    
                    for (int j = 0; j < J; j++) {
                        int idx = i * J + j;
                        As_values[k][j] = softmax_results[idx];
                    }
                }

                // scatter update for Y_q
                for (int d = 0; d < D; d++) {
                    float sum = 0.0f;
                    for (int j = 0; j < J; j++) {
                        for (int k = 0; k < K; k++) {
                            float attn = Ar_values[j][k] * As_values[k][j];
                            sum += attn * Vr_2_acc[b][h][j][d] * Vs_2_acc[b][h][k][d];
                        }
                    }
                    Y_q_acc[b][h][i][d] += sum;
                }
            }
        }
    }
}

void compute_Y_scatter_r(
    torch::Tensor& Y_r_,
    const torch::Tensor& Q,
    const torch::Tensor& R,
    const torch::Tensor& S,
    const torch::Tensor& Vq_2,
    const torch::Tensor& Vs_2
) {
    auto Q_acc = Q.accessor<float, 4>();  
    auto R_acc = R.accessor<float, 4>();  
    auto S_acc = S.accessor<float, 4>();  
    auto Vq_2_acc = Vq_2.accessor<float, 4>();  
    auto Vs_2_acc = Vs_2.accessor<float, 4>();  
    auto Y_r__acc = Y_r_.accessor<float, 4>();

    int B = Q.size(0);
    int H = Q.size(1);
    int I = Q.size(2);
    int J = R.size(2);
    int K = S.size(2);
    int D = Q.size(3);
    float scale = 1.0f / std::sqrt(static_cast<float>(D));

    for (int b = 0; b < B; b++) {
        for (int h = 0; h < H; h++) {
            for (int j = 0; j < J; j++) {
                
                std::vector<std::vector<float>> Aq_values(I, std::vector<float>(K, 0.0f));
                for (int i = 0; i < I; i++) {
                    std::vector<float> softmax_results(J * K);
                    compute_softmax_3d(
                        Q_acc, R_acc, S_acc,
                        b, h,
                        0, i,  
                        J, K,  
                        1, 2, 0, 
                        scale, softmax_results.data()
                    );
                    
                    for (int k = 0; k < K; k++) {
                        int idx = j * K + k;
                        Aq_values[i][k] = softmax_results[idx];
                    }
                }

                std::vector<std::vector<float>> As_values(K, std::vector<float>(I, 0.0f));
                for (int k = 0; k < K; k++) {
                    std::vector<float> softmax_results(I * J);
                    compute_softmax_3d(
                        Q_acc, R_acc, S_acc,
                        b, h,
                        2, k,  
                        I, J,  
                        0, 1, 2,  
                        scale, softmax_results.data()
                    );
                    
                    for (int i = 0; i < I; i++) {
                        int idx = i * J + j;
                        As_values[k][i] = softmax_results[idx];
                    }
                }

                //scatter update for Y_r_
                for (int d = 0; d < D; d++) {
                    float sum = 0.0f;
                    for (int i = 0; i < I; i++) {
                        for (int k = 0; k < K; k++) {
                            // AqAs[i,j,k] = Aq[i,j,k] * As[i,j,k]
                            float attn = Aq_values[i][k] * As_values[k][i];
                            sum += attn * Vq_2_acc[b][h][i][d] * Vs_2_acc[b][h][k][d];
                        }
                    }
                    Y_r__acc[b][h][j][d] += sum;
                }
            }
        }
    }
}

void compute_Y_scatter_s(
    torch::Tensor& Y_s_,
    const torch::Tensor& Q,
    const torch::Tensor& R,
    const torch::Tensor& S,
    const torch::Tensor& Vq_2,
    const torch::Tensor& Vr_2
) {
    auto Q_acc = Q.accessor<float, 4>();  
    auto R_acc = R.accessor<float, 4>();  
    auto S_acc = S.accessor<float, 4>();  
    auto Vq_2_acc = Vq_2.accessor<float, 4>();  
    auto Vr_2_acc = Vr_2.accessor<float, 4>();  
    auto Y_s__acc = Y_s_.accessor<float, 4>();

    int B = Q.size(0);
    int H = Q.size(1);
    int I = Q.size(2);
    int J = R.size(2);
    int K = S.size(2);
    int D = Q.size(3);
    float scale = 1.0f / std::sqrt(static_cast<float>(D));

    for (int b = 0; b < B; b++) {
        for (int h = 0; h < H; h++) {
            for (int k = 0; k < K; k++) {
                std::vector<std::vector<float>> Aq_values(I, std::vector<float>(J, 0.0f));
                for (int i = 0; i < I; i++) {
                    std::vector<float> softmax_results(J * K);
                    compute_softmax_3d(
                        Q_acc, R_acc, S_acc,
                        b, h,
                        0, i,  
                        J, K,  
                        1, 2, 0,  
                        scale, softmax_results.data()
                    );
                    
                    for (int j = 0; j < J; j++) {
                        int idx = j * K + k;  
                        Aq_values[i][j] = softmax_results[idx];
                    }
                }

                std::vector<std::vector<float>> Ar_values(J, std::vector<float>(I, 0.0f));
                for (int j = 0; j < J; j++) {
                    std::vector<float> softmax_results(I * K);
                    compute_softmax_3d(
                        Q_acc, R_acc, S_acc,
                        b, h,
                        1, j,  
                        I, K,  
                        0, 2, 1,  
                        scale, softmax_results.data()
                    );
                    
                    for (int i = 0; i < I; i++) {
                        int idx = i * K + k;  
                        Ar_values[j][i] = softmax_results[idx];
                    }
                }

                //scatter update for Y_s_
                for (int d = 0; d < D; d++) {
                    float sum = 0.0f;
                    for (int i = 0; i < I; i++) {
                        for (int j = 0; j < J; j++) {
                            // AqAr[i,j,k] = Aq[i,j,k] * Ar[i,j,k]
                            float attn = Aq_values[i][j] * Ar_values[j][i];
                            sum += attn * Vq_2_acc[b][h][i][d] * Vr_2_acc[b][h][j][d];
                        }
                    }
                    Y_s__acc[b][h][k][d] += sum;
                }
            }
        }
    }
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor> forward_cpu(
    torch::Tensor Q,
    torch::Tensor R,
    torch::Tensor S,
    torch::Tensor Vq_1,
    torch::Tensor Vq_2,
    torch::Tensor Vr_1,
    torch::Tensor Vr_2,
    torch::Tensor Vs_1,
    torch::Tensor Vs_2,
    double dropout_rate = 0.0)
{
    auto options = Q.options();
    int B = Q.size(0), H = Q.size(1), I = Q.size(2), D = Q.size(3);
    int J = R.size(2), K = S.size(2);
    auto Y_q = torch::zeros({B, H, I, D}, options);
    auto Y_r = torch::zeros({B, H, J, D}, options);
    auto Y_s = torch::zeros({B, H, K, D}, options);
    auto Y_q_ = torch::zeros({B, H, I, D}, options);
    auto Y_r_ = torch::zeros({B, H, J, D}, options);
    auto Y_s_ = torch::zeros({B, H, K, D}, options);

    compute_Y_gather(Y_q, Q, R, S, Vr_1, Vs_1, 0);
    compute_Y_gather(Y_r, Q, R, S, Vq_1, Vs_1, 1);
    compute_Y_gather(Y_s, Q, R, S, Vq_1, Vr_1, 2);

    compute_Y_scatter_q(Y_q_, Q, R, S, Vr_2, Vs_2);
    compute_Y_scatter_r(Y_r_, Q, R, S, Vq_2, Vs_2);
    compute_Y_scatter_s(Y_s_, Q, R, S, Vq_2, Vr_2);

    // return Y_q + Y_r + Y_s + Y_q_ + Y_r_ + Y_s_;
    return std::make_tuple(Y_q, Y_r, Y_s, Y_q_, Y_r_, Y_s_);
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor> forward(
    torch::Tensor Q, torch::Tensor R, torch::Tensor S,
    torch::Tensor Vq_1, torch::Tensor Vq_2,
    torch::Tensor Vr_1, torch::Tensor Vr_2,
    torch::Tensor Vs_1, torch::Tensor Vs_2,
    double dropout_rate)
{
  if (Q.is_cuda()) {
    return forward_cuda(Q,R,S, Vq_1,Vq_2, Vr_1,Vr_2, Vs_1,Vs_2, dropout_rate);
  } else {
    return forward_cpu(Q,R,S, Vq_1,Vq_2, Vr_1,Vr_2, Vs_1,Vs_2, dropout_rate);
  }
}

// Backward pass

void compute_grad_Vq_1(
    torch::Tensor& grad_Vq_1,        
    const torch::Tensor& grad_output, 
    const torch::Tensor& Q,
    const torch::Tensor& R,
    const torch::Tensor& S,
    const torch::Tensor& Vr_1,
    const torch::Tensor& Vs_1,
    double dropout_rate = 0.0)
{
    auto grad_output_acc = grad_output.accessor<float, 4>();
    auto Q_acc = Q.accessor<float, 4>();
    auto R_acc = R.accessor<float, 4>();
    auto S_acc = S.accessor<float, 4>();
    auto Vr_1_acc = Vr_1.accessor<float, 4>();
    auto Vs_1_acc = Vs_1.accessor<float, 4>();
    auto grad_Vq_1_acc = grad_Vq_1.accessor<float, 4>();

    int B = Q.size(0);
    int H = Q.size(1);
    int I = Q.size(2);
    int J = R.size(2);
    int K = S.size(2);
    int D = Q.size(3);
    float scale = 1.0f / std::sqrt(static_cast<float>(D)); // TODO: pass this in as an argument from the interface

    // Vq_1 contributes to both Y_r and Y_s in the forward pass


    for (int b = 0; b < B; b++) {
        for (int h = 0; h < H; h++) {
            // 1. Contribution from Y_r (gather to position j)
            // Y_r = compute_Y_gather(Q, R, S, Vq_1, Vs_1, 1)
            for (int j = 0; j < J; j++) {
                
                // We need the attention weights Ar_j for fixed j
                // (softmax over i,k dimensions for this fixed j)
                std::vector<float> Ar_j_values(I * K); // Will store the attention weights
                
                compute_softmax_3d(
                    Q_acc, R_acc, S_acc,
                    b, h,
                    1, j,  
                    I, K,  
                    0, 2, 1, 
                    scale, Ar_j_values.data()
                );
                
                // Now propagate the gradient for each dimension
                for (int d = 0; d < D; d++) {
                    // Gradient coming from Y_r at this position
                    float dy_r = grad_output_acc[b][h][j][d];
                    
                    // Propagate to Vq_1
                    for (int i = 0; i < I; i++) {
                        for (int k = 0; k < K; k++) {
                            // Index in the flattened attention array
                            int attn_idx = i * K + k;
                            
                            // dL/dVq_1[b,h,i,d] += dL/dY_r[b,h,j,d] * Ar[b,h,i,j,k] * Vs_1[b,h,k,d]
                            float contribution = dy_r * Ar_j_values[attn_idx] * Vs_1_acc[b][h][k][d];
                            
                            // Add to the grad accumulator
                            grad_Vq_1_acc[b][h][i][d] += contribution;
                        }
                    }
                }
            }
            
            // 2. Contribution from Y_s (gather to position k)
            // Y_s = compute_Y_gather(Q, R, S, Vq_1, Vr_1, 2)
            for (int k = 0; k < K; k++) {
                // (softmax over i,j dimensions for this fixed k)
                std::vector<float> As_k_values(I * J); // Will store the attention weights
                
                // Compute softmax for this (b,h,k) using the same function as forward pass
                compute_softmax_3d(
                    Q_acc, R_acc, S_acc,
                    b, h,
                    2, k,  
                    I, J, 
                    0, 1, 2, 
                    scale, As_k_values.data()
                );
                
                // Now propagate the gradient for each dimension
                for (int d = 0; d < D; d++) {
                    // Gradient coming from Y_s at this position
                    float dy_s = grad_output_acc[b][h][k][d];
                    
                    // Propagate to Vq_1
                    for (int i = 0; i < I; i++) {
                        for (int j = 0; j < J; j++) {
                            // Index in the flattened attention array
                            int attn_idx = i * J + j;
                            
                            // dL/dVq_1[b,h,i,d] += dL/dY_s[b,h,k,d] * As[b,h,i,j,k] * Vr_1[b,h,j,d]
                            float contribution = dy_s * As_k_values[attn_idx] * Vr_1_acc[b][h][j][d];
                            
                            // Add to the gradient accumulator
                            grad_Vq_1_acc[b][h][i][d] += contribution;
                        }
                    }
                }
            }
        }
    }
}

void compute_grad_Vr_1(
    torch::Tensor& grad_Vr_1,        
    const torch::Tensor& grad_output, 
    const torch::Tensor& Q,
    const torch::Tensor& R,
    const torch::Tensor& S,
    const torch::Tensor& Vq_1,
    const torch::Tensor& Vs_1,
    double dropout_rate = 0.0)
{
    auto grad_output_acc = grad_output.accessor<float, 4>();
    auto Q_acc = Q.accessor<float, 4>();
    auto R_acc = R.accessor<float, 4>();
    auto S_acc = S.accessor<float, 4>();
    auto Vq_1_acc = Vq_1.accessor<float, 4>();
    auto Vs_1_acc = Vs_1.accessor<float, 4>();
    auto grad_Vr_1_acc = grad_Vr_1.accessor<float, 4>();

    int B = Q.size(0);
    int H = Q.size(1);
    int I = Q.size(2);
    int J = R.size(2);
    int K = S.size(2);
    int D = Q.size(3);
    float scale = 1.0f / std::sqrt(static_cast<float>(D));

    // Vr_1 contributes to both Y_q and Y_s in the forward pass

    // 1. Contribution from Y_q (gather to position i)
    // Y_q = compute_Y_gather(Q, R, S, Vr_1, Vs_1, 0)
    for (int b = 0; b < B; b++) {
        for (int h = 0; h < H; h++) {
            for (int i = 0; i < I; i++) {
                // (softmax over j,k dimensions for this fixed i)
                std::vector<float> Aq_i_values(J * K);
                
                compute_softmax_3d(
                    Q_acc, R_acc, S_acc,
                    b, h,
                    0, i,  
                    J, K,  
                    1, 2, 0, 
                    scale, Aq_i_values.data()
                );
                
                for (int d = 0; d < D; d++) {
                    float dy_q = grad_output_acc[b][h][i][d];
                    
                    // Propagate to Vr_1
                    for (int j = 0; j < J; j++) {
                        for (int k = 0; k < K; k++) {
                            // Index in the flattened attention array
                            int attn_idx = j * K + k;
                            
                            // dL/dVr_1[b,h,j,d] += dL/dY_q[b,h,i,d] * Aq[b,h,i,j,k] * Vs_1[b,h,k,d]
                            float contribution = dy_q * Aq_i_values[attn_idx] * Vs_1_acc[b][h][k][d];
                            
                            grad_Vr_1_acc[b][h][j][d] += contribution;
                        }
                    }
                }
            }
            
            // 2. Contribution from Y_s (gather to position k)
            // Y_s = compute_Y_gather(Q, R, S, Vq_1, Vr_1, 2)
            for (int k = 0; k < K; k++) {
                // (softmax over i,j dimensions for this fixed k)
                std::vector<float> As_k_values(I * J);
                
                compute_softmax_3d(
                    Q_acc, R_acc, S_acc,
                    b, h,
                    2, k,  
                    I, J,  
                    0, 1, 2, 
                    scale, As_k_values.data()
                );
                
                for (int d = 0; d < D; d++) {
                    float dy_s = grad_output_acc[b][h][k][d];
                    
                    // Propagate to Vr_1
                    for (int i = 0; i < I; i++) {
                        for (int j = 0; j < J; j++) {
                            // Index in the flattened attention array
                            int attn_idx = i * J + j;
                            
                            // dL/dVr_1[b,h,j,d] += dL/dY_s[b,h,k,d] * As[b,h,i,j,k] * Vq_1[b,h,i,d]
                            float contribution = dy_s * As_k_values[attn_idx] * Vq_1_acc[b][h][i][d];
                            
                            grad_Vr_1_acc[b][h][j][d] += contribution;
                        }
                    }
                }
            }
        }
    }
}

void compute_grad_Vs_1(
    torch::Tensor& grad_Vs_1,
    const torch::Tensor& grad_output,
    const torch::Tensor& Q,
    const torch::Tensor& R,
    const torch::Tensor& S,
    const torch::Tensor& Vq_1,
    const torch::Tensor& Vr_1,
    double dropout_rate = 0.0)
{
    auto grad_output_acc = grad_output.accessor<float, 4>();
    auto Q_acc = Q.accessor<float, 4>();
    auto R_acc = R.accessor<float, 4>();
    auto S_acc = S.accessor<float, 4>();
    auto Vq_1_acc = Vq_1.accessor<float, 4>();
    auto Vr_1_acc = Vr_1.accessor<float, 4>();
    auto grad_Vs_1_acc = grad_Vs_1.accessor<float, 4>();

    int B = Q.size(0);
    int H = Q.size(1);
    int I = Q.size(2);
    int J = R.size(2);
    int K = S.size(2);
    int D = Q.size(3);
    float scale = 1.0f / std::sqrt(static_cast<float>(D));

    for (int b = 0; b < B; b++) {
        for (int h = 0; h < H; h++) {
            // 1. Contribution from Y_q (gather to position i)
            for (int i = 0; i < I; i++) {
                std::vector<float> Aq_i_values(J * K);
                
                compute_softmax_3d(
                    Q_acc, R_acc, S_acc,
                    b, h,
                    0, i,  
                    J, K,  
                    1, 2, 0, 
                    scale, Aq_i_values.data()
                );
                
                for (int d = 0; d < D; d++) {
                    float dy_q = grad_output_acc[b][h][i][d];
                    
                    for (int j = 0; j < J; j++) {
                        for (int k = 0; k < K; k++) {
                            int attn_idx = j * K + k;
                            float attn = Aq_i_values[attn_idx];
                            grad_Vs_1_acc[b][h][k][d] += dy_q * attn * Vr_1_acc[b][h][j][d];
                        }
                    }
                }
            }
            
            // 2. Contribution from Y_r (gather to position j)
            for (int j = 0; j < J; j++) {
                std::vector<float> Ar_j_values(I * K);
                
                compute_softmax_3d(
                    Q_acc, R_acc, S_acc,
                    b, h,
                    1, j,  
                    I, K,  
                    0, 2, 1, 
                    scale, Ar_j_values.data()
                );
                
                for (int d = 0; d < D; d++) {
                    float dy_r = grad_output_acc[b][h][j][d];
                    
                    for (int i = 0; i < I; i++) {
                        for (int k = 0; k < K; k++) {
                            int attn_idx = i * K + k;
                            float attn = Ar_j_values[attn_idx];
                            grad_Vs_1_acc[b][h][k][d] += dy_r * attn * Vq_1_acc[b][h][i][d];
                        }
                    }
                }
            }
        }
    }
}

// Helper function to compute a SINGLE attention score (Aq, Ar, or As)
// for specific indices (b,h,i,j,k) by performing softmax over the correct dimensions.
// fixed_dim: 0 for Aq (softmax over j,k), 1 for Ar (softmax over i,k), 2 for As (softmax over i,j)
template <typename T>
inline float compute_single_softmax_attn(
    const T& Q_acc,
    const T& R_acc,
    const T& S_acc,
    int b, int h, int i_target, int j_target, int k_target,
    int I, int J, int K, int D,
    float scale,
    int fixed_dim)
{
    float max_val = -std::numeric_limits<float>::infinity();
    float sum_exp = 0.0f;

    // --- First Pass: Find Max ---
    if (fixed_dim == 0) { 
        for (int j = 0; j < J; ++j) {
            for (int k = 0; k < K; ++k) {
                float dot = compute_dot_product(Q_acc, R_acc, S_acc, b, h, i_target, j, k, D);
                max_val = std::max(max_val, dot * scale);
            }
        }
    } else if (fixed_dim == 1) { 
        for (int i = 0; i < I; ++i) {
            for (int k = 0; k < K; ++k) {
                float dot = compute_dot_product(Q_acc, R_acc, S_acc, b, h, i, j_target, k, D);
                max_val = std::max(max_val, dot * scale);
            }
        }
    } else { 
        for (int i = 0; i < I; ++i) {
            for (int j = 0; j < J; ++j) {
                float dot = compute_dot_product(Q_acc, R_acc, S_acc, b, h, i, j, k_target, D);
                max_val = std::max(max_val, dot * scale);
            }
        }
    }

    // --- Second Pass: Compute Sum Exp ---
     if (fixed_dim == 0) { // Aq
        for (int j = 0; j < J; ++j) {
            for (int k = 0; k < K; ++k) {
                float dot = compute_dot_product(Q_acc, R_acc, S_acc, b, h, i_target, j, k, D);
                sum_exp += std::exp(dot * scale - max_val);
            }
        }
    } else if (fixed_dim == 1) { // Ar
        for (int i = 0; i < I; ++i) {
            for (int k = 0; k < K; ++k) {
                float dot = compute_dot_product(Q_acc, R_acc, S_acc, b, h, i, j_target, k, D);
                sum_exp += std::exp(dot * scale - max_val);
            }
        }
    } else { // As
        for (int i = 0; i < I; ++i) {
            for (int j = 0; j < J; ++j) {
                float dot = compute_dot_product(Q_acc, R_acc, S_acc, b, h, i, j, k_target, D);
                sum_exp += std::exp(dot * scale - max_val);
            }
        }
    }

    float target_dot = compute_dot_product(Q_acc, R_acc, S_acc, b, h, i_target, j_target, k_target, D);
    if (sum_exp == 0.0f) { //prevent division by zero 
        int num_elements = 0;
        if (fixed_dim == 0) num_elements = J * K;
        else if (fixed_dim == 1) num_elements = I * K;
        else num_elements = I * J;
        return 1.0f / static_cast<float>(num_elements);
    }
    return std::exp(target_dot * scale - max_val) / sum_exp;
}

void compute_grad_Vq_2(
    torch::Tensor& grad_Vq_2,        // Output: Gradient w.r.t. Vq_2 [B, H, I, D]
    const torch::Tensor& grad_output, // Input: Gradient w.r.t. final output Y [B, H, max(I,J,K), D]
    const torch::Tensor& Q,           
    const torch::Tensor& R,           
    const torch::Tensor& S,           
    const torch::Tensor& Vr_2,        
    const torch::Tensor& Vs_2,        
    double dropout_rate = 0.0)        
{
    auto grad_Vq_2_acc = grad_Vq_2.accessor<float, 4>();
    auto grad_output_acc = grad_output.accessor<float, 4>();
    auto Q_acc = Q.accessor<float, 4>();
    auto R_acc = R.accessor<float, 4>();
    auto S_acc = S.accessor<float, 4>();
    auto Vr_2_acc = Vr_2.accessor<float, 4>();
    auto Vs_2_acc = Vs_2.accessor<float, 4>();

    const int B = Q.size(0);
    const int H = Q.size(1);
    const int I = Q.size(2);
    const int J = R.size(2);
    const int K = S.size(2);
    const int D = Q.size(3);
    const float scale = 1.0f / std::sqrt(static_cast<float>(D));

    for (int b = 0; b < B; ++b) {
        for (int h = 0; h < H; ++h) {

            // --- Contribution from Y_q_ Path ---
            // dL/dVq_2[i] += sum_{j} (dL/dY_r_[j] * dY_r_[j]/dVq_2[i])
            // dY_r_[j]/dVq_2[i] = sum_{k} (Aq[i,j,k] * As[i,j,k] * Vs_2[k])
            for (int j = 0; j < J; ++j) { 
                for (int d = 0; d < D; ++d) {
                    // Gradient coming from Y_r_ at index j, dimension d
                    // Ensure grad_output access is within bounds if J < max_seq_len
                    if (j >= grad_output.size(2)) continue;
                    const float dy_r = grad_output_acc[b][h][j][d];

                    if (dy_r == 0.0f) continue;

                    for (int i = 0; i < I; ++i) { 
                        for (int k = 0; k < K; ++k) { 
                            float attn_aq = compute_single_softmax_attn(Q_acc, R_acc, S_acc, b, h, i, j, k, I, J, K, D, scale, 0);
                            float attn_as = compute_single_softmax_attn(Q_acc, R_acc, S_acc, b, h, i, j, k, I, J, K, D, scale, 2);

                            float vs2_val = Vs_2_acc[b][h][k][d];

                            // Accumulate gradient: dL/dY_r * Aq * As * Vs_2
                            grad_Vq_2_acc[b][h][i][d] += dy_r * attn_aq * attn_as * vs2_val;
                        }
                    }
                }
            }

            // --- Contribution from Y_s_ Path ---
            // dL/dVq_2[i] += sum_{k} (dL/dY_s_[k] * dY_s_[k]/dVq_2[i])
            // dY_s_[k]/dVq_2[i] = sum_{j} (Aq[i,j,k] * Ar[i,j,k] * Vr_2[j])
             for (int k = 0; k < K; ++k) { // Loop over source gradient index (Y_s_)
                 for (int d = 0; d < D; ++d) {
                     // Gradient coming from Y_s_ at index k, dimension d
                     // Ensure grad_output access is within bounds if K < max_seq_len
                     if (k >= grad_output.size(2)) continue;
                     const float dy_s = grad_output_acc[b][h][k][d];

                     if (dy_s == 0.0f) continue; // Optimization

                     for (int j = 0; j < J; ++j) { 
                         for (int i = 0; i < I; ++i) { 
                             float attn_aq = compute_single_softmax_attn(Q_acc, R_acc, S_acc, b, h, i, j, k, I, J, K, D, scale, 0); // Softmax over j,k
                             float attn_ar = compute_single_softmax_attn(Q_acc, R_acc, S_acc, b, h, i, j, k, I, J, K, D, scale, 1); // Softmax over i,k

                             float vr2_val = Vr_2_acc[b][h][j][d];

                             // Accumulate gradient: dL/dY_s * Aq * Ar * Vr_2
                             grad_Vq_2_acc[b][h][i][d] += dy_s * attn_aq * attn_ar * vr2_val;
                         }
                     }
                 }
            }
        } 
    } 
}

void compute_grad_Vr_2(
    torch::Tensor& grad_Vr_2,        
    const torch::Tensor& grad_output, 
    const torch::Tensor& Q,           
    const torch::Tensor& R,           
    const torch::Tensor& S,           
    const torch::Tensor& Vq_2,        
    const torch::Tensor& Vs_2,        
    double dropout_rate = 0.0)        
{
    auto grad_Vr_2_acc = grad_Vr_2.accessor<float, 4>();
    auto grad_output_acc = grad_output.accessor<float, 4>();
    auto Q_acc = Q.accessor<float, 4>();
    auto R_acc = R.accessor<float, 4>();
    auto S_acc = S.accessor<float, 4>();
    auto Vq_2_acc = Vq_2.accessor<float, 4>();
    auto Vs_2_acc = Vs_2.accessor<float, 4>();

    const int B = Q.size(0);
    const int H = Q.size(1);
    const int I = Q.size(2);
    const int J = R.size(2);
    const int K = S.size(2);
    const int D = Q.size(3);
    const float scale = 1.0f / std::sqrt(static_cast<float>(D));


    for (int b = 0; b < B; ++b) {
        for (int h = 0; h < H; ++h) {

            // --- Contribution from Y_q_ Path ---
            // dL/dVr_2[j] += sum_{i} (dL/dY_q_[i] * dY_q_[i]/dVr_2[j])
            // dY_q_[i]/dVr_2[j] = sum_{k} (Ar[i,j,k] * As[i,j,k] * Vs_2[k])
            for (int i = 0; i < I; ++i) { // Loop over source gradient index (Y_q_)
                for (int d = 0; d < D; ++d) {
                    // Gradient coming from Y_q_ at index i, dimension d
                    // Ensure grad_output access is within bounds if I < max_seq_len
                    if (i >= grad_output.size(2)) continue;
                    const float dy_q = grad_output_acc[b][h][i][d];

                    if (dy_q == 0.0f) continue; // Optimization: skip if gradient is zero

                    for (int j = 0; j < J; ++j) { // Loop over target gradient index (Vr_2)
                        for (int k = 0; k < K; ++k) { // Summation index
                            // Recompute necessary attention scores on the fly
                            float attn_ar = compute_single_softmax_attn(Q_acc, R_acc, S_acc, b, h, i, j, k, I, J, K, D, scale, 1); // Softmax over i,k
                            float attn_as = compute_single_softmax_attn(Q_acc, R_acc, S_acc, b, h, i, j, k, I, J, K, D, scale, 2); // Softmax over i,j

                            float vs2_val = Vs_2_acc[b][h][k][d];

                            // Accumulate gradient: dL/dY_q * Ar * As * Vs_2
                            grad_Vr_2_acc[b][h][j][d] += dy_q * attn_ar * attn_as * vs2_val;
                        }
                    }
                }
            }

            // --- Contribution from Y_s_ Path ---
            // dL/dVr_2[j] += sum_{k} (dL/dY_s_[k] * dY_s_[k]/dVr_2[j])
            // dY_s_[k]/dVr_2[j] = sum_{i} (Aq[i,j,k] * Ar[i,j,k] * Vq_2[i])
            for (int k = 0; k < K; ++k) { // Loop over source gradient index (Y_s_)
                for (int d = 0; d < D; ++d) {
                    // Gradient coming from Y_s_ at index k, dimension d
                    // Ensure grad_output access is within bounds if K < max_seq_len
                    if (k >= grad_output.size(2)) continue;
                    const float dy_s = grad_output_acc[b][h][k][d];

                    if (dy_s == 0.0f) continue; 

                    for (int j = 0; j < J; ++j) { // Loop over target gradient index (Vr_2)
                        for (int i = 0; i < I; ++i) { 
                            // Recompute necessary attention scores on the fly
                            float attn_aq = compute_single_softmax_attn(Q_acc, R_acc, S_acc, b, h, i, j, k, I, J, K, D, scale, 0); // Softmax over j,k
                            float attn_ar = compute_single_softmax_attn(Q_acc, R_acc, S_acc, b, h, i, j, k, I, J, K, D, scale, 1); // Softmax over i,k

                            float vq2_val = Vq_2_acc[b][h][i][d];

                            // Accumulate gradient: dL/dY_s * Aq * Ar * Vq_2
                            grad_Vr_2_acc[b][h][j][d] += dy_s * attn_aq * attn_ar * vq2_val;
                        }
                    }
                }
            }
        } 
    } 
}

void compute_grad_Vs_2(
    torch::Tensor& grad_Vs_2,        
    const torch::Tensor& grad_output, 
    const torch::Tensor& Q,           
    const torch::Tensor& R,           
    const torch::Tensor& S,           
    const torch::Tensor& Vq_2,        
    const torch::Tensor& Vr_2,        
    double dropout_rate = 0.0)        
{
    auto grad_Vs_2_acc = grad_Vs_2.accessor<float, 4>();
    auto grad_output_acc = grad_output.accessor<float, 4>();
    auto Q_acc = Q.accessor<float, 4>();
    auto R_acc = R.accessor<float, 4>();
    auto S_acc = S.accessor<float, 4>();
    auto Vq_2_acc = Vq_2.accessor<float, 4>();
    auto Vr_2_acc = Vr_2.accessor<float, 4>();

    const int B = Q.size(0);
    const int H = Q.size(1);
    const int I = Q.size(2);
    const int J = R.size(2);
    const int K = S.size(2);
    const int D = Q.size(3);
    const float scale = 1.0f / std::sqrt(static_cast<float>(D));

    for (int b = 0; b < B; ++b) {
        for (int h = 0; h < H; ++h) {

            // --- Contribution from Y_q_ Path ---
            // dL/dVs_2[k] += sum_{i} (dL/dY_q_[i] * dY_q_[i]/dVs_2[k])
            // dY_q_[i]/dVs_2[k] = sum_{j} (Ar[i,j,k] * As[i,j,k] * Vr_2[j])
            for (int i = 0; i < I; ++i) { // Loop over source gradient index (Y_q_)
                for (int d = 0; d < D; ++d) {
                    // Gradient coming from Y_q_ at index i, dimension d
                    // Ensure grad_output access is within bounds if I < max_seq_len
                    if (i >= grad_output.size(2)) continue;
                    const float dy_q = grad_output_acc[b][h][i][d];

                    if (dy_q == 0.0f) continue; 

                    for (int k = 0; k < K; ++k) { // Loop over target gradient index (Vs_2)
                        for (int j = 0; j < J; ++j) { 
                            // Recompute necessary attention scores on the fly
                            float attn_ar = compute_single_softmax_attn(Q_acc, R_acc, S_acc, b, h, i, j, k, I, J, K, D, scale, 1); // Softmax over i,k
                            float attn_as = compute_single_softmax_attn(Q_acc, R_acc, S_acc, b, h, i, j, k, I, J, K, D, scale, 2); // Softmax over i,j

                            float vr2_val = Vr_2_acc[b][h][j][d];

                            // Accumulate gradient: dL/dY_q * Ar * As * Vr_2
                            grad_Vs_2_acc[b][h][k][d] += dy_q * attn_ar * attn_as * vr2_val;
                        }
                    }
                }
            }

            // --- Contribution from Y_r_ Path ---
            // dL/dVs_2[k] += sum_{j} (dL/dY_r_[j] * dY_r_[j]/dVs_2[k])
            // dY_r_[j]/dVs_2[k] = sum_{i} (Aq[i,j,k] * As[i,j,k] * Vq_2[i])
            for (int j = 0; j < J; ++j) { // Loop over source gradient index (Y_r_)
                for (int d = 0; d < D; ++d) {
                    // Gradient coming from Y_r_ at index j, dimension d
                    // Ensure grad_output access is within bounds if J < max_seq_len
                    if (j >= grad_output.size(2)) continue;
                    const float dy_r = grad_output_acc[b][h][j][d];

                    if (dy_r == 0.0f) continue; 

                    for (int k = 0; k < K; ++k) { // Loop over target gradient index (Vs_2)
                        for (int i = 0; i < I; ++i) {
                            // Recompute necessary attention scores on the fly
                            float attn_aq = compute_single_softmax_attn(Q_acc, R_acc, S_acc, b, h, i, j, k, I, J, K, D, scale, 0); // Softmax over j,k
                            float attn_as = compute_single_softmax_attn(Q_acc, R_acc, S_acc, b, h, i, j, k, I, J, K, D, scale, 2); // Softmax over i,j

                            float vq2_val = Vq_2_acc[b][h][i][d];

                            // Accumulate gradient: dL/dY_r * Aq * As * Vq_2
                            grad_Vs_2_acc[b][h][k][d] += dy_r * attn_aq * attn_as * vq2_val;
                        }
                    }
                }
            }
        } 
    } 
}


// Computes A slice, Aq, Ar, As for a single batch item and head
std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
compute_attention_tensors_single(
    const torch::Tensor& Q_slice, // Shape [I, D]
    const torch::Tensor& R_slice, // Shape [J, D]
    const torch::Tensor& S_slice, // Shape [K, D]
    float scale                   // Scaling factor 1/sqrt(D)
) {
    const int I = Q_slice.size(0);
    const int J = R_slice.size(0);
    const int K = S_slice.size(0);
    const int D = Q_slice.size(1);
    auto options = Q_slice.options();

    // 1. Compute A_slice [I, J, K] 
    auto A_slice = torch::zeros({I, J, K}, options); 
    auto A_acc = A_slice.accessor<float, 3>();      
    auto Q_acc = Q_slice.accessor<float, 2>();
    auto R_acc = R_slice.accessor<float, 2>();
    auto S_acc = S_slice.accessor<float, 2>();

    auto compute_dot_single = [&](int i, int j, int k) {
        float dot = 0.0f;
        for (int d = 0; d < D; ++d) {
            dot += Q_acc[i][d] * R_acc[j][d] * S_acc[k][d];
        }
        return dot * scale;
    };

    for (int i = 0; i < I; ++i) {
        for (int j = 0; j < J; ++j) {
            for (int k = 0; k < K; ++k) {
                A_acc[i][j][k] = compute_dot_single(i, j, k); // <-- Renamed
            }
        }
    }

    // 2. Create tensors for normalized attention scores
    auto Aq_slice = torch::zeros({I, J, K}, options);
    auto Ar_slice = torch::zeros({I, J, K}, options);
    auto As_slice = torch::zeros({I, J, K}, options);
    
        // Simplified adapter for the compute_softmax_3d function
    auto apply_softmax_3d = [&](torch::Tensor& output_tensor, int fixed_dim) {
        auto output_acc = output_tensor.accessor<float, 3>();
        
        // Configure dimension parameters based on fixed_dim
        int dim1_size, dim2_size;
        int dim1_idx_fn, dim2_idx_fn;
        
        if (fixed_dim == 0) {
            dim1_size = J; dim2_size = K;
            dim1_idx_fn = 1; dim2_idx_fn = 2;
        } else if (fixed_dim == 1) {
            dim1_size = I; dim2_size = K;
            dim1_idx_fn = 0; dim2_idx_fn = 2;
        } else { // fixed_dim == 2
            dim1_size = I; dim2_size = J;
            dim1_idx_fn = 0; dim2_idx_fn = 1;
        }
        
        // Apply softmax for each fixed dimension index
        int fixed_dim_size = fixed_dim == 0 ? I : (fixed_dim == 1 ? J : K);
        
        for (int fixed_idx = 0; fixed_idx < fixed_dim_size; ++fixed_idx) {
            // Create temporary buffer for softmax results
            std::vector<float> softmax_buffer(dim1_size * dim2_size);
            
            // Simplify compute_dot_product to use our existing compute_dot_single
            auto compute_dot_wrapper = [&](int i, int j, int k) {
                return A_acc[i][j][k];  // Use precomputed A_slice values
            };
            
            // Find max for numerical stability
            float max_val = -std::numeric_limits<float>::infinity();
            for (int idx1 = 0; idx1 < dim1_size; idx1++) {
                for (int idx2 = 0; idx2 < dim2_size; idx2++) {
                    int i = fixed_dim == 0 ? fixed_idx : (dim1_idx_fn == 0 ? idx1 : idx2);
                    int j = fixed_dim == 1 ? fixed_idx : (dim1_idx_fn == 1 ? idx1 : idx2);
                    int k = fixed_dim == 2 ? fixed_idx : (dim1_idx_fn == 2 ? idx1 : idx2);
                    
                    float val = compute_dot_wrapper(i, j, k);
                    if (val > max_val) max_val = val;
                }
            }
            
            // Compute sum of exponentials
            float sum_exp = 0.0f;
            for (int idx1 = 0; idx1 < dim1_size; idx1++) {
                for (int idx2 = 0; idx2 < dim2_size; idx2++) {
                    int i = fixed_dim == 0 ? fixed_idx : (dim1_idx_fn == 0 ? idx1 : idx2);
                    int j = fixed_dim == 1 ? fixed_idx : (dim1_idx_fn == 1 ? idx1 : idx2);
                    int k = fixed_dim == 2 ? fixed_idx : (dim1_idx_fn == 2 ? idx1 : idx2);
                    
                    float val = compute_dot_wrapper(i, j, k);
                    sum_exp += std::exp(val - max_val);
                }
            }
            
            // Compute softmax values
            int idx = 0;
            for (int idx1 = 0; idx1 < dim1_size; idx1++) {
                for (int idx2 = 0; idx2 < dim2_size; idx2++) {
                    int i = fixed_dim == 0 ? fixed_idx : (dim1_idx_fn == 0 ? idx1 : idx2);
                    int j = fixed_dim == 1 ? fixed_idx : (dim1_idx_fn == 1 ? idx1 : idx2);
                    int k = fixed_dim == 2 ? fixed_idx : (dim1_idx_fn == 2 ? idx1 : idx2);
                    
                    float val = compute_dot_wrapper(i, j, k);
                    float softmax_val = std::exp(val - max_val) / sum_exp;
                    
                    // Store in output tensor
                    output_acc[i][j][k] = softmax_val;
                    idx++;
                }
            }
        }
    };
    
    // Apply softmax to each dimension
    apply_softmax_3d(Aq_slice, 0);  // Fix i (dim 0), softmax over j,k
    apply_softmax_3d(Ar_slice, 1);  // Fix j (dim 1), softmax over i,k
    apply_softmax_3d(As_slice, 2);  // Fix k (dim 2), softmax over i,j

    // Return A_slice (raw scores) along with normalized scores
    return std::make_tuple(A_slice, Aq_slice, Ar_slice, As_slice); 
}


// Computes grad_A for a single batch item and head 
torch::Tensor compute_grad_A_single( 
    const torch::Tensor& grad_output_slice, // Shape [N, D]
    const torch::Tensor& Q_slice,           // [I, D]
    const torch::Tensor& R_slice,           // [J, D]
    const torch::Tensor& S_slice,           // [K, D]
    const torch::Tensor& Vq_1_slice,        // [I, D]
    const torch::Tensor& Vq_2_slice,        // [I, D]
    const torch::Tensor& Vr_1_slice,        // [J, D]
    const torch::Tensor& Vr_2_slice,        // [J, D]
    const torch::Tensor& Vs_1_slice,        // [K, D]
    const torch::Tensor& Vs_2_slice,        // [K, D]
    const torch::Tensor& A_slice,           // [I, J, K] 
    const torch::Tensor& Aq_slice,          // [I, J, K]
    const torch::Tensor& Ar_slice,          // [I, J, K]
    const torch::Tensor& As_slice,           // [I, J, K]
    int b, int h 
) {
    const int I = Q_slice.size(0);
    const int J = R_slice.size(0);
    const int K = S_slice.size(0);
    const int D = Q_slice.size(1);
    const int N = grad_output_slice.size(0); 
    // auto options = Q_slice.options(); 

    auto grad_output_acc = grad_output_slice.accessor<float, 2>();
    auto Vq_1_acc = Vq_1_slice.accessor<float, 2>();
    auto Vq_2_acc = Vq_2_slice.accessor<float, 2>();
    auto Vr_1_acc = Vr_1_slice.accessor<float, 2>();
    auto Vr_2_acc = Vr_2_slice.accessor<float, 2>();
    auto Vs_1_acc = Vs_1_slice.accessor<float, 2>();
    auto Vs_2_acc = Vs_2_slice.accessor<float, 2>();
    auto Aq_acc = Aq_slice.accessor<float, 3>();
    auto Ar_acc = Ar_slice.accessor<float, 3>();
    auto As_acc = As_slice.accessor<float, 3>();

    // --- Phase 1: Compute grad_A* slices ---
    auto grad_Aq_slice = torch::zeros_like(Aq_slice);
    auto grad_Ar_slice = torch::zeros_like(Ar_slice);
    auto grad_As_slice = torch::zeros_like(As_slice);
    auto grad_Aq_acc = grad_Aq_slice.accessor<float, 3>();
    auto grad_Ar_acc = grad_Ar_slice.accessor<float, 3>();
    auto grad_As_acc = grad_As_slice.accessor<float, 3>();

    // 1.a) grad_Aq from Yq (gather)
    if (I <= N) {
        for (int i = 0; i < I; ++i) { for (int j = 0; j < J; ++j) { for (int k = 0; k < K; ++k) { for (int d = 0; d < D; ++d) {
            grad_Aq_acc[i][j][k] += grad_output_acc[i][d] * Vr_1_acc[j][d] * Vs_1_acc[k][d];
        }}}}
    }
    // 1.b) grad_Aq from Yr' (scatter)
    if (J <= N) {
        for (int i = 0; i < I; ++i) { for (int j = 0; j < J; ++j) { for (int k = 0; k < K; ++k) {
            float grad_x_vals = 0.0f;
            for (int d = 0; d < D; ++d) { grad_x_vals += grad_output_acc[j][d] * Vq_2_acc[i][d] * Vs_2_acc[k][d]; }
            grad_Aq_acc[i][j][k] += grad_x_vals * As_acc[i][j][k];
        }}}
    }
    // 1.c) grad_Aq from Ys' (scatter)
    if (K <= N) {
        for (int i = 0; i < I; ++i) { for (int j = 0; j < J; ++j) { for (int k = 0; k < K; ++k) {
            float grad_x_vals = 0.0f;
            for (int d = 0; d < D; ++d) { grad_x_vals += grad_output_acc[k][d] * Vq_2_acc[i][d] * Vr_2_acc[j][d]; }
            grad_Aq_acc[i][j][k] += grad_x_vals * Ar_acc[i][j][k];
        }}}
    }

    // 2.a) grad_Ar from Yr
    if (J <= N) {
        for (int i = 0; i < I; ++i) { for (int j = 0; j < J; ++j) { for (int k = 0; k < K; ++k) { for (int d = 0; d < D; ++d) {
            grad_Ar_acc[i][j][k] += grad_output_acc[j][d] * Vq_1_acc[i][d] * Vs_1_acc[k][d];
        }}}}
    }
    // 2.b) grad_Ar from Yq'
    if (I <= N) {
        for (int i = 0; i < I; ++i) { for (int j = 0; j < J; ++j) { for (int k = 0; k < K; ++k) {
            float grad_x_vals = 0.0f;
            for (int d = 0; d < D; ++d) { grad_x_vals += grad_output_acc[i][d] * Vr_2_acc[j][d] * Vs_2_acc[k][d]; }
            grad_Ar_acc[i][j][k] += grad_x_vals * As_acc[i][j][k];
        }}}
    }
    // 2.c) grad_Ar from Ys'
    if (K <= N) {
        for (int i = 0; i < I; ++i) { for (int j = 0; j < J; ++j) { for (int k = 0; k < K; ++k) {
            float grad_x_vals = 0.0f;
            for (int d = 0; d < D; ++d) { grad_x_vals += grad_output_acc[k][d] * Vq_2_acc[i][d] * Vr_2_acc[j][d]; }
            grad_Ar_acc[i][j][k] += grad_x_vals * Aq_acc[i][j][k];
        }}}
    }

    // 3.a) grad_As from Ys
    if (K <= N) {
        for (int i = 0; i < I; ++i) { for (int j = 0; j < J; ++j) { for (int k = 0; k < K; ++k) { for (int d = 0; d < D; ++d) {
            grad_As_acc[i][j][k] += grad_output_acc[k][d] * Vq_1_acc[i][d] * Vr_1_acc[j][d];
        }}}}
    }
    // 3.b) grad_As from Yq' 
    if (I <= N) {
        for (int i = 0; i < I; ++i) { for (int j = 0; j < J; ++j) { for (int k = 0; k < K; ++k) {
            float grad_x_vals = 0.0f;
            for (int d = 0; d < D; ++d) { grad_x_vals += grad_output_acc[i][d] * Vr_2_acc[j][d] * Vs_2_acc[k][d]; }
            grad_As_acc[i][j][k] += grad_x_vals * Ar_acc[i][j][k];
        }}}
    }
    // 3.c) grad_As from Yr' 
    if (J <= N) {
        for (int i = 0; i < I; ++i) { for (int j = 0; j < J; ++j) { for (int k = 0; k < K; ++k) {
            float grad_x_vals = 0.0f;
            for (int d = 0; d < D; ++d) { grad_x_vals += grad_output_acc[j][d] * Vq_2_acc[i][d] * Vs_2_acc[k][d]; }
            grad_As_acc[i][j][k] += grad_x_vals * Aq_acc[i][j][k];
        }}}
    }

    // --- Phase 2: Propagate Gradients Back Through Softmax to get grad_A_slice ---
    auto grad_A_slice = torch::zeros_like(A_slice);
    auto grad_A_acc = grad_A_slice.accessor<float, 3>();

    // 2.1 Contribution from Aq (Softmax over j, k)
    for (int i = 0; i < I; ++i) {
        float sum_q = 0.0f;
        for (int j = 0; j < J; ++j) { for (int k = 0; k < K; ++k) { sum_q += grad_Aq_acc[i][j][k] * Aq_acc[i][j][k]; }}
        for (int j = 0; j < J; ++j) { for (int k = 0; k < K; ++k) { grad_A_acc[i][j][k] += (grad_Aq_acc[i][j][k] - sum_q) * Aq_acc[i][j][k]; }}
    }

    // 2.2 Contribution from Ar (Softmax over i, k)
    for (int j = 0; j < J; ++j) {
        float sum_r = 0.0f;
        for (int i = 0; i < I; ++i) { for (int k = 0; k < K; ++k) { sum_r += grad_Ar_acc[i][j][k] * Ar_acc[i][j][k]; }}
        for (int i = 0; i < I; ++i) { for (int k = 0; k < K; ++k) { grad_A_acc[i][j][k] += (grad_Ar_acc[i][j][k] - sum_r) * Ar_acc[i][j][k]; }}
    }

    // 2.3 Contribution from As (Softmax over i, j)
    for (int k = 0; k < K; ++k) {
        float sum_s = 0.0f;
        for (int i = 0; i < I; ++i) { for (int j = 0; j < J; ++j) { sum_s += grad_As_acc[i][j][k] * As_acc[i][j][k]; }}
        for (int i = 0; i < I; ++i) { for (int j = 0; j < J; ++j) { grad_A_acc[i][j][k] += (grad_As_acc[i][j][k] - sum_s) * As_acc[i][j][k]; }}
    }

    return grad_A_slice;
}

// Computes grad_Q for a single batch item and head
torch::Tensor compute_grad_Q_single(
    const torch::Tensor& grad_A_slice, // Input: Gradient w.r.t. A [I, J, K] 
    const torch::Tensor& R_slice,      // Input: R slice [J, D]
    const torch::Tensor& S_slice,      // Input: S slice [K, D]
    float scale                        // Input: Scaling factor 1/sqrt(D)
) {
    const int I = grad_A_slice.size(0);
    const int J = grad_A_slice.size(1);
    const int K = grad_A_slice.size(2);
    const int D = R_slice.size(1); // Get D from R_slice or S_slice
    auto options = R_slice.options();

    auto grad_Q_slice = torch::zeros({I, D}, options); // Output tensor [I, D]
    auto grad_Q_acc = grad_Q_slice.accessor<float, 2>();
    auto grad_A_acc = grad_A_slice.accessor<float, 3>();
    auto R_acc = R_slice.accessor<float, 2>();
    auto S_acc = S_slice.accessor<float, 2>();

    // grad_Q = scale * einsum("ijk,jd,kd->id", grad_A_slice, R_slice, S_slice)
     for (int i = 0; i < I; ++i) {
        for (int d = 0; d < D; ++d) {
            float sum_for_grad_q = 0.0f;
            for (int j = 0; j < J; ++j) {
                for (int k = 0; k < K; ++k) {
                    sum_for_grad_q += grad_A_acc[i][j][k] * R_acc[j][d] * S_acc[k][d];
                }
            }
            grad_Q_acc[i][d] = scale * sum_for_grad_q;
        }
    }
    return grad_Q_slice;
}

// Computes grad_R for a single batch item and head
torch::Tensor compute_grad_R_single(
    const torch::Tensor& grad_A_slice, // Input: Gradient w.r.t. A [I, J, K] 
    const torch::Tensor& Q_slice,      // Input: Q slice [I, D]
    const torch::Tensor& S_slice,      // Input: S slice [K, D]
    float scale                        // Input: Scaling factor 1/sqrt(D)
) {
    const int I = grad_A_slice.size(0);
    const int J = grad_A_slice.size(1);
    const int K = grad_A_slice.size(2);
    const int D = Q_slice.size(1); // Get D from Q_slice or S_slice
    auto options = Q_slice.options();

    auto grad_R_slice = torch::zeros({J, D}, options); // Output tensor [J, D]
    auto grad_R_acc = grad_R_slice.accessor<float, 2>();
    auto grad_A_acc = grad_A_slice.accessor<float, 3>();
    auto Q_acc = Q_slice.accessor<float, 2>();
    auto S_acc = S_slice.accessor<float, 2>();

    // grad_R = scale * einsum("ijk,id,kd->jd", grad_A_slice, Q_slice, S_slice)
    for (int j = 0; j < J; ++j) { // Loop over target dimension J
        for (int d = 0; d < D; ++d) {
            float sum_for_grad_r = 0.0f;
            for (int i = 0; i < I; ++i) { // Sum over other dimensions I, K
                for (int k = 0; k < K; ++k) {
                    sum_for_grad_r += grad_A_acc[i][j][k] * Q_acc[i][d] * S_acc[k][d];
                }
            }
            grad_R_acc[j][d] = scale * sum_for_grad_r;
        }
    }
    return grad_R_slice;
}

// Computes grad_S for a single batch item and head
torch::Tensor compute_grad_S_single(
    const torch::Tensor& grad_A_slice, // Input: Gradient w.r.t. A [I, J, K]
    const torch::Tensor& Q_slice,      // Input: Q slice [I, D]
    const torch::Tensor& R_slice,      // Input: R slice [J, D]
    float scale                        // Input: Scaling factor 1/sqrt(D)
) {
    const int I = grad_A_slice.size(0);
    const int J = grad_A_slice.size(1);
    const int K = grad_A_slice.size(2);
    const int D = Q_slice.size(1); // Get D from Q_slice or R_slice
    auto options = Q_slice.options();

    auto grad_S_slice = torch::zeros({K, D}, options); // Output tensor [K, D]
    auto grad_S_acc = grad_S_slice.accessor<float, 2>();
    auto grad_A_acc = grad_A_slice.accessor<float, 3>();
    auto Q_acc = Q_slice.accessor<float, 2>();
    auto R_acc = R_slice.accessor<float, 2>();

    // grad_S = scale * einsum("ijk,id,jd->kd", grad_A_slice, Q_slice, R_slice)
    for (int k = 0; k < K; ++k) { // Loop over target dimension K
        for (int d = 0; d < D; ++d) {
            float sum_for_grad_s = 0.0f;
            for (int i = 0; i < I; ++i) { // Sum over other dimensions I, J
                for (int j = 0; j < J; ++j) {
                    sum_for_grad_s += grad_A_acc[i][j][k] * Q_acc[i][d] * R_acc[j][d];
                }
            }
            grad_S_acc[k][d] = scale * sum_for_grad_s;
        }
    }
    return grad_S_slice;
}


std::tuple<torch::Tensor, torch::Tensor, torch::Tensor,
          torch::Tensor, torch::Tensor,
          torch::Tensor, torch::Tensor,
          torch::Tensor, torch::Tensor>
backward_cpu(
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
    double dropout_rate = 0.0) // need to remove later
{
    const int B = Q.size(0);
    const int H = Q.size(1);

    // const int I = Q.size(2);
    // const int J = R.size(2);
    // const int K = S.size(2);
    const int D = Q.size(3);
    const float scale = 1.0f / std::sqrt(static_cast<float>(D));

    auto grad_Q = torch::zeros_like(Q);
    auto grad_R = torch::zeros_like(R);
    auto grad_S = torch::zeros_like(S);
    auto grad_Vq_1 = torch::zeros_like(Vq_1);
    auto grad_Vq_2 = torch::zeros_like(Vq_2);
    auto grad_Vr_1 = torch::zeros_like(Vr_1);
    auto grad_Vr_2 = torch::zeros_like(Vr_2);
    auto grad_Vs_1 = torch::zeros_like(Vs_1);
    auto grad_Vs_2 = torch::zeros_like(Vs_2);

    compute_grad_Vq_1(grad_Vq_1, grad_output, Q, R, S, Vr_1, Vs_1, dropout_rate);
    compute_grad_Vr_1(grad_Vr_1, grad_output, Q, R, S, Vq_1, Vs_1, dropout_rate);
    compute_grad_Vs_1(grad_Vs_1, grad_output, Q, R, S, Vq_1, Vr_1, dropout_rate);

    compute_grad_Vq_2(grad_Vq_2, grad_output, Q, R, S, Vr_2, Vs_2, dropout_rate);
    compute_grad_Vr_2(grad_Vr_2, grad_output, Q, R, S, Vq_2, Vs_2, dropout_rate);
    compute_grad_Vs_2(grad_Vs_2, grad_output, Q, R, S, Vq_2, Vr_2, dropout_rate);

    // --- Compute Gradients for Q, R, S iteratively ---
    for (int b = 0; b < B; ++b) {
        for (int h = 0; h < H; ++h) {
            // Get slices for the current batch and head
            auto Q_slice = Q.select(0, b).select(0, h);
            auto R_slice = R.select(0, b).select(0, h);
            auto S_slice = S.select(0, b).select(0, h);
            auto Vq_1_slice = Vq_1.select(0, b).select(0, h);
            auto Vq_2_slice = Vq_2.select(0, b).select(0, h);
            auto Vr_1_slice = Vr_1.select(0, b).select(0, h);
            auto Vr_2_slice = Vr_2.select(0, b).select(0, h);
            auto Vs_1_slice = Vs_1.select(0, b).select(0, h);
            auto Vs_2_slice = Vs_2.select(0, b).select(0, h);

            auto grad_output_slice = grad_output.select(0, b).select(0, h);
            // const int N = grad_output_slice.size(0);

            torch::Tensor A_slice, Aq_slice, Ar_slice, As_slice;
            std::tie(A_slice, Aq_slice, Ar_slice, As_slice) =
                compute_attention_tensors_single(Q_slice, R_slice, S_slice, scale);

            auto grad_A_slice = compute_grad_A_single(
                grad_output_slice, Q_slice, R_slice, S_slice,
                Vq_1_slice, Vq_2_slice, Vr_1_slice, Vr_2_slice, Vs_1_slice, Vs_2_slice,
                A_slice, Aq_slice, Ar_slice, As_slice,
                b, h
            );

            auto grad_Q_slice = compute_grad_Q_single(grad_A_slice, R_slice, S_slice, scale);
            auto grad_R_slice = compute_grad_R_single(grad_A_slice, Q_slice, S_slice, scale);
            auto grad_S_slice = compute_grad_S_single(grad_A_slice, Q_slice, R_slice, scale);

            grad_Q.select(0, b).select(0, h).add_(grad_Q_slice);
            grad_R.select(0, b).select(0, h).add_(grad_R_slice);
            grad_S.select(0, b).select(0, h).add_(grad_S_slice);
        }
    }

    return std::make_tuple(
        grad_Q, grad_R, grad_S,
        grad_Vq_1, grad_Vq_2,
        grad_Vr_1, grad_Vr_2,
        grad_Vs_1, grad_Vs_2
    );
}

auto backward(
    torch::Tensor grad_output,
    torch::Tensor Q, torch::Tensor R, torch::Tensor S,
    torch::Tensor Vq_1, torch::Tensor Vq_2,
    torch::Tensor Vr_1, torch::Tensor Vr_2,
    torch::Tensor Vs_1, torch::Tensor Vs_2,
    double dropout_rate)
{
  if (grad_output.is_cuda()) {
    return backward_cuda(grad_output, Q,R,S, Vq_1,Vq_2, Vr_1,Vr_2, Vs_1,Vs_2, dropout_rate);
  } else {
    return backward_cpu(grad_output, Q,R,S, Vq_1,Vq_2, Vr_1,Vr_2, Vs_1,Vs_2, dropout_rate);
  }
}


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward_cuda,
          "Hypergraph Attention forward (returns Y_q, Y_r, Y_s, Y_q_, Y_r_, Y_s_)",
          py::arg("Q"),
          py::arg("R"),
          py::arg("S"),
          py::arg("Vq_1"),
          py::arg("Vq_2"),
          py::arg("Vr_1"),
          py::arg("Vr_2"),
          py::arg("Vs_1"),
          py::arg("Vs_2"),
          py::arg("dropout_rate") = 0.0);

    m.def("backward", &backward,
          "Hypergraph Attention backward (returns dQ, dR, dS, dVq_1, dVq_2, dVr_1, dVr_2, dVs_1, dVs_2)",
          py::arg("grad_output"),
          py::arg("Q"),
          py::arg("R"),
          py::arg("S"),
          py::arg("Vq_1"),
          py::arg("Vq_2"),
          py::arg("Vr_1"),
          py::arg("Vr_2"),
          py::arg("Vs_1"),
          py::arg("Vs_2"),
          py::arg("dropout_rate") = 0.0);
}

