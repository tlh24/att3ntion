#include <torch/extension.h> 
#include <iostream> 
#include <cmath> 
#include <limits>

// Helper function to compute dot product between three vectors at specific indices
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

// Helper function to compute 3D softmax with configurable dimensions
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
    torch::Tensor& Y_out,      // Output tensor (Y_q, Y_r, or Y_s)
    const torch::Tensor& Q,
    const torch::Tensor& R, 
    const torch::Tensor& S,
    const torch::Tensor& V1,   // First value tensor
    const torch::Tensor& V2,   // Second value tensor
    int fixed_dim              // Which dimension to fix (0=i/q, 1=j/r, 2=k/s)
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

    // Sizes of each dimension
    int sizes[3] = {I, J, K};
    
    // For each batch and head
    for (int b = 0; b < B; b++) {
        for (int h = 0; h < H; h++) {
            // Loop over the fixed dimension
            for (int fixed_idx = 0; fixed_idx < sizes[fixed_dim]; fixed_idx++) {
                // Define variables for the other two dimensions based on the fixed dimension
                int dim1, dim2;
                int dim1_size, dim2_size;
                
                if (fixed_dim == 0) {      // If i is fixed
                    dim1 = 1; dim2 = 2;    // j and k are the other dimensions
                    dim1_size = J; dim2_size = K;
                } else if (fixed_dim == 1) { // If j is fixed
                    dim1 = 0; dim2 = 2;    // i and k are the other dimensions
                    dim1_size = I; dim2_size = K;
                } else {                   // If k is fixed
                    dim1 = 0; dim2 = 1;    // i and j are the other dimensions
                    dim1_size = I; dim2_size = J;
                }
                
                // Memory-efficient implementation: compute softmax without materializing full tensor
                for (int d = 0; d < D; d++) {
                    // Zero out the output for this fixed position and dimension
                    Y_out_acc[b][h][fixed_idx][d] = 0.0f;
                    
                    // First pass: compute max for numerical stability in softmax
                    float max_val = -std::numeric_limits<float>::infinity();
                    for (int idx1 = 0; idx1 < dim1_size; idx1++) {
                        for (int idx2 = 0; idx2 < dim2_size; idx2++) {
                            // Map idx1, idx2 to i, j, k based on which dimensions they represent
                            int i = (fixed_dim == 0) ? fixed_idx : (dim1 == 0) ? idx1 : idx2;
                            int j = (fixed_dim == 1) ? fixed_idx : (dim1 == 1) ? idx1 : idx2;
                            int k = (fixed_dim == 2) ? fixed_idx : (dim1 == 2) ? idx1 : idx2;
                            
                            // Compute dot product
                            float dot_prod = 0.0f;
                            for (int d_dot = 0; d_dot < D; d_dot++) {
                                dot_prod += Q_acc[b][h][i][d_dot] * R_acc[b][h][j][d_dot] * S_acc[b][h][k][d_dot];
                            }
                            dot_prod *= scale;
                            
                            max_val = std::max(max_val, dot_prod);
                        }
                    }
                    
                    // Second pass: compute sum of exp(dot_prod - max_val) for normalization
                    float sum_exp = 0.0f;
                    for (int idx1 = 0; idx1 < dim1_size; idx1++) {
                        for (int idx2 = 0; idx2 < dim2_size; idx2++) {
                            int i = (fixed_dim == 0) ? fixed_idx : (dim1 == 0) ? idx1 : idx2;
                            int j = (fixed_dim == 1) ? fixed_idx : (dim1 == 1) ? idx1 : idx2;
                            int k = (fixed_dim == 2) ? fixed_idx : (dim1 == 2) ? idx1 : idx2;
                            
                            float dot_prod = 0.0f;
                            for (int d_dot = 0; d_dot < D; d_dot++) {
                                dot_prod += Q_acc[b][h][i][d_dot] * R_acc[b][h][j][d_dot] * S_acc[b][h][k][d_dot];
                            }
                            dot_prod *= scale;
                            
                            sum_exp += std::exp(dot_prod - max_val);
                        }
                    }
                    
                    // Third pass: compute weighted sum directly
                    for (int idx1 = 0; idx1 < dim1_size; idx1++) {
                        for (int idx2 = 0; idx2 < dim2_size; idx2++) {
                            int i = (fixed_dim == 0) ? fixed_idx : (dim1 == 0) ? idx1 : idx2;
                            int j = (fixed_dim == 1) ? fixed_idx : (dim1 == 1) ? idx1 : idx2;
                            int k = (fixed_dim == 2) ? fixed_idx : (dim1 == 2) ? idx1 : idx2;
                            
                            // Get indices for value tensors
                            int v1_idx = (dim1 == 0) ? i : (dim1 == 1) ? j : k;
                            int v2_idx = (dim2 == 0) ? i : (dim2 == 1) ? j : k;
                            
                            float dot_prod = 0.0f;
                            for (int d_dot = 0; d_dot < D; d_dot++) {
                                dot_prod += Q_acc[b][h][i][d_dot] * R_acc[b][h][j][d_dot] * S_acc[b][h][k][d_dot];
                            }
                            dot_prod *= scale;
                            
                            // Compute softmax value and multiply by values
                            float attn = std::exp(dot_prod - max_val) / sum_exp;
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
            // For each i position
            for (int i = 0; i < I; i++) {
                // We need to compute the effect of ArAs * Vr_2 * Vs_2
                // But we'll do it without explicitly forming ArAs
                
                // first precompute some values for efficiency
                // For each j, compute attention to js (softmax over i,k)
                std::vector<std::vector<float>> Ar_values(J, std::vector<float>(K, 0.0f));
                for (int j = 0; j < J; j++) {
                    // Use compute_softmax_3d to compute Ar values (softmax over i,k for fixed j)
                    std::vector<float> softmax_results(I * K);
                    compute_softmax_3d(
                        Q_acc, R_acc, S_acc,
                        b, h,
                        1, j,  // fixed_dim = 1 (j), fixed_idx = j
                        I, K,  // dim1_size = I, dim2_size = K
                        0, 2, 1,  // dim1_idx_fn = 0 (i), dim2_idx_fn = 2 (k), fixed_dim_idx_fn = 1 (j)
                        scale, softmax_results.data()
                    );
                    
                    // Extract the attention values for the current i
                    for (int k = 0; k < K; k++) {
                        // Find the index of (i,k) in the softmax_results array
                        int idx = i * K + k;
                        Ar_values[j][k] = softmax_results[idx];
                    }
                }

                // Now compute As values
                std::vector<std::vector<float>> As_values(K, std::vector<float>(J, 0.0f));
                for (int k = 0; k < K; k++) {
                    // Use compute_softmax_3d to compute As values (softmax over i,j for fixed k)
                    std::vector<float> softmax_results(I * J);
                    compute_softmax_3d(
                        Q_acc, R_acc, S_acc,
                        b, h,
                        2, k,  // fixed_dim = 2 (k), fixed_idx = k
                        I, J,  // dim1_size = I, dim2_size = J
                        0, 1, 2,  // dim1_idx_fn = 0 (i), dim2_idx_fn = 1 (j), fixed_dim_idx_fn = 2 (k)
                        scale, softmax_results.data()
                    );
                    
                    // Extract the attention values for the current i
                    for (int j = 0; j < J; j++) {
                        // Find the index of (i,j) in the softmax_results array
                        int idx = i * J + j;
                        As_values[k][j] = softmax_results[idx];
                    }
                }

                // Now compute the actual scatter update for Y_q
                for (int d = 0; d < D; d++) {
                    float sum = 0.0f;
                    for (int j = 0; j < J; j++) {
                        for (int k = 0; k < K; k++) {
                            // ArAs[i,j,k] = Ar[i,j,k] * As[i,j,k]
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
            // For each j position
            for (int j = 0; j < J; j++) {
                // compute the effect of AqAs * Vq_2 * Vs_2 without explicitly forming AqAs
                
                // For each i, compute attention to is (softmax over j,k)
                std::vector<std::vector<float>> Aq_values(I, std::vector<float>(K, 0.0f));
                for (int i = 0; i < I; i++) {
                    // Use compute_softmax_3d to compute Aq values (softmax over j,k for fixed i)
                    std::vector<float> softmax_results(J * K);
                    compute_softmax_3d(
                        Q_acc, R_acc, S_acc,
                        b, h,
                        0, i,  // fixed_dim = 0 (i), fixed_idx = i
                        J, K,  // dim1_size = J, dim2_size = K
                        1, 2, 0,  // dim1_idx_fn = 1 (j), dim2_idx_fn = 2 (k), fixed_dim_idx_fn = 0 (i)
                        scale, softmax_results.data()
                    );
                    
                    // Extract the attention values for the current j
                    for (int k = 0; k < K; k++) {
                        // Find the index of (j,k) in the softmax_results array
                        int idx = j * K + k;
                        Aq_values[i][k] = softmax_results[idx];
                    }
                }

                // Now compute As values
                std::vector<std::vector<float>> As_values(K, std::vector<float>(I, 0.0f));
                for (int k = 0; k < K; k++) {
                    // Use compute_softmax_3d to compute As values (softmax over i,j for fixed k)
                    std::vector<float> softmax_results(I * J);
                    compute_softmax_3d(
                        Q_acc, R_acc, S_acc,
                        b, h,
                        2, k,  // fixed_dim = 2 (k), fixed_idx = k
                        I, J,  // dim1_size = I, dim2_size = J
                        0, 1, 2,  // dim1_idx_fn = 0 (i), dim2_idx_fn = 1 (j), fixed_dim_idx_fn = 2 (k)
                        scale, softmax_results.data()
                    );
                    
                    // Extract the attention values for the current i and j
                    for (int i = 0; i < I; i++) {
                        // Find the index of (i,j) in the softmax_results array
                        int idx = i * J + j;
                        As_values[k][i] = softmax_results[idx];
                    }
                }

                // Now compute the actual scatter update for Y_r_
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
            // For each k position
            for (int k = 0; k < K; k++) {
                //  compute the effect of AqAr * Vq_2 * Vr_2 without explicitly forming AqAr
                // For each i, compute attention to is (softmax over j,k)
                std::vector<std::vector<float>> Aq_values(I, std::vector<float>(J, 0.0f));
                for (int i = 0; i < I; i++) {
                    // Use compute_softmax_3d to compute Aq values (softmax over j,k for fixed i)
                    std::vector<float> softmax_results(J * K);
                    compute_softmax_3d(
                        Q_acc, R_acc, S_acc,
                        b, h,
                        0, i,  // fixed_dim = 0 (i), fixed_idx = i
                        J, K,  // dim1_size = J, dim2_size = K
                        1, 2, 0,  // dim1_idx_fn = 1 (j), dim2_idx_fn = 2 (k), fixed_dim_idx_fn = 0 (i)
                        scale, softmax_results.data()
                    );
                    
                    // Extract the attention values for the current k
                    for (int j = 0; j < J; j++) {
                        // We're interested in the current k, so find the correct index
                        int idx = j * K + k;  // Position for (j,k) in the softmax array
                        Aq_values[i][j] = softmax_results[idx];
                    }
                }

                // Now compute Ar values
                std::vector<std::vector<float>> Ar_values(J, std::vector<float>(I, 0.0f));
                for (int j = 0; j < J; j++) {
                    // Use compute_softmax_3d to compute Ar values (softmax over i,k for fixed j)
                    std::vector<float> softmax_results(I * K);
                    compute_softmax_3d(
                        Q_acc, R_acc, S_acc,
                        b, h,
                        1, j,  // fixed_dim = 1 (j), fixed_idx = j
                        I, K,  // dim1_size = I, dim2_size = K
                        0, 2, 1,  // dim1_idx_fn = 0 (i), dim2_idx_fn = 2 (k), fixed_dim_idx_fn = 1 (j)
                        scale, softmax_results.data()
                    );
                    
                    // Extract the attention values for the current k
                    for (int i = 0; i < I; i++) {
                        // We're interested in the current k, so find the correct index
                        int idx = i * K + k;  // Position for (i,k) in the softmax array
                        Ar_values[j][i] = softmax_results[idx];
                    }
                }

                // Now compute the actual scatter update for Y_s_
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

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor> hyper_attn_forward(
    torch::Tensor Q,       // [B, H, I, D]
    torch::Tensor R,       // [B, H, J, D]
    torch::Tensor S,       // [B, H, K, D]
    torch::Tensor Vq_1,    // [B, H, I, D]
    torch::Tensor Vq_2,    // [B, H, I, D]
    torch::Tensor Vr_1,    // [B, H, J, D]
    torch::Tensor Vr_2,    // [B, H, J, D]
    torch::Tensor Vs_1,    // [B, H, K, D]
    torch::Tensor Vs_2,    // [B, H, K, D]
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

    return std::make_tuple(Y_q, Y_r, Y_s, Y_q_, Y_r_, Y_s_);
}

// Helper function to backpropagate through softmax
void backprop_softmax_3d(
    float* d_dot_products,   // Output: gradient w.r.t dot products
    const float* d_softmax,  // Input: gradient w.r.t softmax outputs
    const float* softmax,    // Input: softmax outputs from forward pass
    int size                 // Number of elements
) {
    // Compute sum of softmax * gradient
    float sum = 0.0f;
    for (int i = 0; i < size; i++) {
        sum += softmax[i] * d_softmax[i];
    }
    
    // Apply softmax derivative: d_softmax[i] * (softmax[i] - softmax[i] * sum)
    for (int i = 0; i < size; i++) {
        d_dot_products[i] = softmax[i] * (d_softmax[i] - sum);
    }
}

// Helper function for backpropagating through gather operations
void backprop_Y_gather(
    torch::Tensor& dQ,       // Gradient for Q
    torch::Tensor& dR,       // Gradient for R
    torch::Tensor& dS,       // Gradient for S
    torch::Tensor& dV1,      // Gradient for V1
    torch::Tensor& dV2,      // Gradient for V2
    const torch::Tensor& grad_Y, // Gradient from upstream
    const torch::Tensor& Q,
    const torch::Tensor& R, 
    const torch::Tensor& S,
    const torch::Tensor& V1,
    const torch::Tensor& V2,
    int fixed_dim              // Which dimension is fixed (0=i/q, 1=j/r, 2=k/s)
) {
    auto Q_acc = Q.accessor<float, 4>();  
    auto R_acc = R.accessor<float, 4>();  
    auto S_acc = S.accessor<float, 4>();  
    auto V1_acc = V1.accessor<float, 4>();  
    auto V2_acc = V2.accessor<float, 4>();  
    auto grad_Y_acc = grad_Y.accessor<float, 4>();
    auto dQ_acc = dQ.accessor<float, 4>();
    auto dR_acc = dR.accessor<float, 4>();
    auto dS_acc = dS.accessor<float, 4>();
    auto dV1_acc = dV1.accessor<float, 4>();
    auto dV2_acc = dV2.accessor<float, 4>();

    int B = Q.size(0);
    int H = Q.size(1);
    int I = Q.size(2);
    int J = R.size(2);
    int K = S.size(2);
    int D = Q.size(3);
    float scale = 1.0f / std::sqrt(static_cast<float>(D));

    // Sizes of each dimension
    int sizes[3] = {I, J, K};
    
    // For each batch and head
    for (int b = 0; b < B; b++) {
        for (int h = 0; h < H; h++) {
            // Loop over the fixed dimension
            for (int fixed_idx = 0; fixed_idx < sizes[fixed_dim]; fixed_idx++) {
                // Define variables for the other two dimensions based on the fixed dimension
                int dim1, dim2;
                int dim1_size, dim2_size;
                
                if (fixed_dim == 0) {      // If i is fixed
                    dim1 = 1; dim2 = 2;    // j and k are the other dimensions
                    dim1_size = J; dim2_size = K;
                } else if (fixed_dim == 1) { // If j is fixed
                    dim1 = 0; dim2 = 2;    // i and k are the other dimensions
                    dim1_size = I; dim2_size = K;
                } else {                   // If k is fixed
                    dim1 = 0; dim2 = 1;    // i and j are the other dimensions
                    dim1_size = I; dim2_size = J;
                }
                
                // Allocate array for softmax attention values and derivatives
                std::vector<float> attn_values(dim1_size * dim2_size);
                std::vector<float> d_attn_values(dim1_size * dim2_size, 0.0f);
                
                // Compute softmax over the two non-fixed dimensions
                compute_softmax_3d(
                    Q_acc, R_acc, S_acc,
                    b, h,
                    fixed_dim, fixed_idx,
                    dim1_size, dim2_size,
                    dim1, dim2, fixed_dim,
                    scale, attn_values.data()
                );
                
                // Backpropagate through the weighted value tensors
                for (int d = 0; d < D; d++) {
                    float grad = grad_Y_acc[b][h][fixed_idx][d];
                    
                    int idx = 0;
                    for (int idx1 = 0; idx1 < dim1_size; idx1++) {
                        for (int idx2 = 0; idx2 < dim2_size; idx2++) {
                            float attn = attn_values[idx++];
                            
                            // Map idx1, idx2 to i, j, k based on which dimensions they represent
                            int i = (fixed_dim == 0) ? fixed_idx : (dim1 == 0) ? idx1 : idx2;
                            int j = (fixed_dim == 1) ? fixed_idx : (dim1 == 1) ? idx1 : idx2;
                            int k = (fixed_dim == 2) ? fixed_idx : (dim1 == 2) ? idx1 : idx2;
                            
                            // Get indices for value tensors
                            int v1_idx = (dim1 == 0) ? i : (dim1 == 1) ? j : k;
                            int v2_idx = (dim2 == 0) ? i : (dim2 == 1) ? j : k;
                            
                            // Backpropagate gradient to values
                            dV1_acc[b][h][v1_idx][d] += grad * attn * V2_acc[b][h][v2_idx][d];
                            dV2_acc[b][h][v2_idx][d] += grad * attn * V1_acc[b][h][v1_idx][d];
                            
                            // Accumulate gradients for attention weights
                            d_attn_values[idx-1] += grad * V1_acc[b][h][v1_idx][d] * V2_acc[b][h][v2_idx][d];
                        }
                    }
                }
                
                // Backpropagate through softmax to get gradients for dot products
                std::vector<float> d_dot_products(dim1_size * dim2_size, 0.0f);
                backprop_softmax_3d(
                    d_dot_products.data(),
                    d_attn_values.data(),
                    attn_values.data(),
                    dim1_size * dim2_size
                );
                
                // Backpropagate to Q, R, S
                int idx = 0;
                for (int idx1 = 0; idx1 < dim1_size; idx1++) {
                    for (int idx2 = 0; idx2 < dim2_size; idx2++) {
                        float d_dot = d_dot_products[idx++] * scale;
                        
                        int i = (fixed_dim == 0) ? fixed_idx : (dim1 == 0) ? idx1 : idx2;
                        int j = (fixed_dim == 1) ? fixed_idx : (dim1 == 1) ? idx1 : idx2;
                        int k = (fixed_dim == 2) ? fixed_idx : (dim1 == 2) ? idx1 : idx2;
                        
                        for (int d = 0; d < D; d++) {
                            dQ_acc[b][h][i][d] += d_dot * R_acc[b][h][j][d] * S_acc[b][h][k][d];
                            dR_acc[b][h][j][d] += d_dot * Q_acc[b][h][i][d] * S_acc[b][h][k][d];
                            dS_acc[b][h][k][d] += d_dot * Q_acc[b][h][i][d] * R_acc[b][h][j][d];
                        }
                    }
                }
            }
        }
    }
}

// Backpropagation through scatter operations for Q
void backprop_Y_scatter_q(
    torch::Tensor& dQ,
    torch::Tensor& dR,
    torch::Tensor& dS,
    torch::Tensor& dVr_2,
    torch::Tensor& dVs_2,
    const torch::Tensor& grad_Y_q_,
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
    auto grad_Y_q__acc = grad_Y_q_.accessor<float, 4>();
    auto dQ_acc = dQ.accessor<float, 4>();
    auto dR_acc = dR.accessor<float, 4>();
    auto dS_acc = dS.accessor<float, 4>();
    auto dVr_2_acc = dVr_2.accessor<float, 4>();
    auto dVs_2_acc = dVs_2.accessor<float, 4>();

    int B = Q.size(0);
    int H = Q.size(1);
    int I = Q.size(2);
    int J = R.size(2);
    int K = S.size(2);
    int D = Q.size(3);
    float scale = 1.0f / std::sqrt(static_cast<float>(D));

    for (int b = 0; b < B; b++) {
        for (int h = 0; h < H; h++) {
            // For each i position
            for (int i = 0; i < I; i++) {
                // Process each dimension independently
                for (int d = 0; d < D; d++) {
                    float grad = grad_Y_q__acc[b][h][i][d];
                    
                    // Precompute Ar values for each (j,k) for fixed i
                    std::vector<std::vector<float>> Ar_values(J, std::vector<float>(K));
                    for (int j = 0; j < J; j++) {
                        std::vector<float> softmax_results(I * K);
                        compute_softmax_3d(
                            Q_acc, R_acc, S_acc,
                            b, h,
                            1, j,  // fixed j
                            I, K,  // dimensions i and k
                            0, 2, 1,  // map to i, k, j
                            scale, softmax_results.data()
                        );
                        
                        // Extract values for current i
                        for (int k = 0; k < K; k++) {
                            int idx = i * K + k;
                            Ar_values[j][k] = softmax_results[idx];
                        }
                    }
                    
                    // Precompute As values for each (k,j) for fixed i
                    std::vector<std::vector<float>> As_values(K, std::vector<float>(J));
                    for (int k = 0; k < K; k++) {
                        std::vector<float> softmax_results(I * J);
                        compute_softmax_3d(
                            Q_acc, R_acc, S_acc,
                            b, h,
                            2, k,  // fixed k
                            I, J,  // dimensions i and j
                            0, 1, 2,  // map to i, j, k
                            scale, softmax_results.data()
                        );
                        
                        // Extract values for current i
                        for (int j = 0; j < J; j++) {
                            int idx = i * J + j;
                            As_values[k][j] = softmax_results[idx];
                        }
                    }
                    
                    // Backpropagate to Vr_2 and Vs_2
                    for (int j = 0; j < J; j++) {
                        for (int k = 0; k < K; k++) {
                            float attn = Ar_values[j][k] * As_values[k][j];
                            dVr_2_acc[b][h][j][d] += grad * attn * Vs_2_acc[b][h][k][d];
                            dVs_2_acc[b][h][k][d] += grad * attn * Vr_2_acc[b][h][j][d];
                            
                            // We need gradients for Ar and As
                            float d_ar_as = grad * Vr_2_acc[b][h][j][d] * Vs_2_acc[b][h][k][d];
                            
                            // Handle backprop through Ar (fixed j, softmax over i,k)
                            float d_ar = d_ar_as * As_values[k][j];
                            std::vector<float> d_ar_softmax(I * K, 0.0f);
                            int ar_idx = i * K + k;
                            d_ar_softmax[ar_idx] = d_ar;
                            
                            std::vector<float> ar_softmax_outputs(I * K);
                            compute_softmax_3d(
                                Q_acc, R_acc, S_acc,
                                b, h,
                                1, j,  // fixed j
                                I, K,  // dimensions i and k
                                0, 2, 1,  // map to i, j, k
                                scale, ar_softmax_outputs.data()
                            );
                            
                            std::vector<float> d_ar_dot(I * K);
                            backprop_softmax_3d(
                                d_ar_dot.data(),
                                d_ar_softmax.data(),
                                ar_softmax_outputs.data(),
                                I * K
                            );
                            
                            // Handle backprop through As (fixed k, softmax over i,j)
                            float d_as = d_ar_as * Ar_values[j][k];
                            std::vector<float> d_as_softmax(I * J, 0.0f);
                            int as_idx = i * J + j;
                            d_as_softmax[as_idx] = d_as;
                            
                            std::vector<float> as_softmax_outputs(I * J);
                            compute_softmax_3d(
                                Q_acc, R_acc, S_acc,
                                b, h,
                                2, k,  // fixed k
                                I, J,  // dimensions i and j
                                0, 1, 2,  // map to i, j, k
                                scale, as_softmax_outputs.data()
                            );
                            
                            std::vector<float> d_as_dot(I * J);
                            backprop_softmax_3d(
                                d_as_dot.data(),
                                d_as_softmax.data(),
                                as_softmax_outputs.data(),
                                I * J
                            );
                            
                            // Backpropagate to Q, R, S from Ar gradients
                            int ar_idx_iter = 0;
                            for (int i_ar = 0; i_ar < I; i_ar++) {
                                for (int k_ar = 0; k_ar < K; k_ar++) {
                                    float d_dot = d_ar_dot[ar_idx_iter++] * scale;
                                    for (int dd = 0; dd < D; dd++) {
                                        dQ_acc[b][h][i_ar][dd] += d_dot * R_acc[b][h][j][dd] * S_acc[b][h][k_ar][dd];
                                        dR_acc[b][h][j][dd] += d_dot * Q_acc[b][h][i_ar][dd] * S_acc[b][h][k_ar][dd];
                                        dS_acc[b][h][k_ar][dd] += d_dot * Q_acc[b][h][i_ar][dd] * R_acc[b][h][j][dd];
                                    }
                                }
                            }
                            
                            // Backpropagate to Q, R, S from As gradients
                            int as_idx_iter = 0;
                            for (int i_as = 0; i_as < I; i_as++) {
                                for (int j_as = 0; j_as < J; j_as++) {
                                    float d_dot = d_as_dot[as_idx_iter++] * scale;
                                    for (int dd = 0; dd < D; dd++) {
                                        dQ_acc[b][h][i_as][dd] += d_dot * R_acc[b][h][j_as][dd] * S_acc[b][h][k][dd];
                                        dR_acc[b][h][j_as][dd] += d_dot * Q_acc[b][h][i_as][dd] * S_acc[b][h][k][dd];
                                        dS_acc[b][h][k][dd] += d_dot * Q_acc[b][h][i_as][dd] * R_acc[b][h][j_as][dd];
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }
}

// Backpropagation through scatter operations for R
void backprop_Y_scatter_r(
    torch::Tensor& dQ,
    torch::Tensor& dR,
    torch::Tensor& dS,
    torch::Tensor& dVq_2,
    torch::Tensor& dVs_2,
    const torch::Tensor& grad_Y_r_,
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
    auto grad_Y_r__acc = grad_Y_r_.accessor<float, 4>();
    auto dQ_acc = dQ.accessor<float, 4>();
    auto dR_acc = dR.accessor<float, 4>();
    auto dS_acc = dS.accessor<float, 4>();
    auto dVq_2_acc = dVq_2.accessor<float, 4>();
    auto dVs_2_acc = dVs_2.accessor<float, 4>();

    int B = Q.size(0);
    int H = Q.size(1);
    int I = Q.size(2);
    int J = R.size(2);
    int K = S.size(2);
    int D = Q.size(3);
    float scale = 1.0f / std::sqrt(static_cast<float>(D));

    for (int b = 0; b < B; b++) {
        for (int h = 0; h < H; h++) {
            // For each j position
            for (int j = 0; j < J; j++) {
                // Process each dimension independently
                for (int d = 0; d < D; d++) {
                    float grad = grad_Y_r__acc[b][h][j][d];
                    
                    // Precompute Aq values for each (i,k) for fixed j
                    std::vector<std::vector<float>> Aq_values(I, std::vector<float>(K));
                    for (int i = 0; i < I; i++) {
                        std::vector<float> softmax_results(J * K);
                        compute_softmax_3d(
                            Q_acc, R_acc, S_acc,
                            b, h,
                            0, i,  // fixed i
                            J, K,  // dimensions j and k
                            1, 2, 0,  // map to j, k, i
                            scale, softmax_results.data()
                        );
                        
                        // Extract values for current j
                        for (int k = 0; k < K; k++) {
                            int idx = j * K + k;
                            Aq_values[i][k] = softmax_results[idx];
                        }
                    }
                    
                    // Precompute As values for each (k,i) for fixed j
                    std::vector<std::vector<float>> As_values(K, std::vector<float>(I));
                    for (int k = 0; k < K; k++) {
                        std::vector<float> softmax_results(I * J);
                        compute_softmax_3d(
                            Q_acc, R_acc, S_acc,
                            b, h,
                            2, k,  // fixed k
                            I, J,  // dimensions i and j
                            0, 1, 2,  // map to i, j, k
                            scale, softmax_results.data()
                        );
                        
                        // Extract values for current j
                        for (int i = 0; i < I; i++) {
                            int idx = i * J + j;
                            As_values[k][i] = softmax_results[idx];
                        }
                    }
                    
                    // Backpropagate to Vq_2 and Vs_2
                    for (int i = 0; i < I; i++) {
                        for (int k = 0; k < K; k++) {
                            float attn = Aq_values[i][k] * As_values[k][i];
                            dVq_2_acc[b][h][i][d] += grad * attn * Vs_2_acc[b][h][k][d];
                            dVs_2_acc[b][h][k][d] += grad * attn * Vq_2_acc[b][h][i][d];
                            
                            // We need gradients for Aq and As
                            float d_aq_as = grad * Vq_2_acc[b][h][i][d] * Vs_2_acc[b][h][k][d];
                            
                            // Handle backprop through Aq (fixed i, softmax over j,k)
                            float d_aq = d_aq_as * As_values[k][i];
                            std::vector<float> d_aq_softmax(J * K, 0.0f);
                            int aq_idx = j * K + k;
                            d_aq_softmax[aq_idx] = d_aq;
                            
                            std::vector<float> aq_softmax_outputs(J * K);
                            compute_softmax_3d(
                                Q_acc, R_acc, S_acc,
                                b, h,
                                0, i,  // fixed i
                                J, K,  // dimensions j and k
                                1, 2, 0,  // map to j, k, i
                                scale, aq_softmax_outputs.data()
                            );
                            
                            std::vector<float> d_aq_dot(J * K);
                            backprop_softmax_3d(
                                d_aq_dot.data(),
                                d_aq_softmax.data(),
                                aq_softmax_outputs.data(),
                                J * K
                            );
                            
                            // Handle backprop through As (fixed k, softmax over i,j)
                            float d_as = d_aq_as * Aq_values[i][k];
                            std::vector<float> d_as_softmax(I * J, 0.0f);
                            int as_idx = i * J + j;
                            d_as_softmax[as_idx] = d_as;
                            
                            std::vector<float> as_softmax_outputs(I * J);
                            compute_softmax_3d(
                                Q_acc, R_acc, S_acc,
                                b, h,
                                2, k,  // fixed k
                                I, J,  // dimensions i and j
                                0, 1, 2,  // map to i, j, k
                                scale, as_softmax_outputs.data()
                            );
                            
                            std::vector<float> d_as_dot(I * J);
                            backprop_softmax_3d(
                                d_as_dot.data(),
                                d_as_softmax.data(),
                                as_softmax_outputs.data(),
                                I * J
                            );
                            
                            // Backpropagate to Q, R, S from Aq gradients
                            int aq_idx_iter = 0;
                            for (int j_aq = 0; j_aq < J; j_aq++) {
                                for (int k_aq = 0; k_aq < K; k_aq++) {
                                    float d_dot = d_aq_dot[aq_idx_iter++] * scale;
                                    for (int dd = 0; dd < D; dd++) {
                                        dQ_acc[b][h][i][dd] += d_dot * R_acc[b][h][j_aq][dd] * S_acc[b][h][k_aq][dd];
                                        dR_acc[b][h][j_aq][dd] += d_dot * Q_acc[b][h][i][dd] * S_acc[b][h][k_aq][dd];
                                        dS_acc[b][h][k_aq][dd] += d_dot * Q_acc[b][h][i][dd] * R_acc[b][h][j_aq][dd];
                                    }
                                }
                            }
                            
                            // Backpropagate to Q, R, S from As gradients
                            int as_idx_iter = 0;
                            for (int i_as = 0; i_as < I; i_as++) {
                                for (int j_as = 0; j_as < J; j_as++) {
                                    float d_dot = d_as_dot[as_idx_iter++] * scale;
                                    for (int dd = 0; dd < D; dd++) {
                                        dQ_acc[b][h][i_as][dd] += d_dot * R_acc[b][h][j_as][dd] * S_acc[b][h][k][dd];
                                        dR_acc[b][h][j_as][dd] += d_dot * Q_acc[b][h][i_as][dd] * S_acc[b][h][k][dd];
                                        dS_acc[b][h][k][dd] += d_dot * Q_acc[b][h][i_as][dd] * R_acc[b][h][j_as][dd];
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }
}

// Backpropagation through scatter operations for S
void backprop_Y_scatter_s(
    torch::Tensor& dQ,
    torch::Tensor& dR,
    torch::Tensor& dS,
    torch::Tensor& dVq_2,
    torch::Tensor& dVr_2,
    const torch::Tensor& grad_Y_s_,
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
    auto grad_Y_s__acc = grad_Y_s_.accessor<float, 4>();
    auto dQ_acc = dQ.accessor<float, 4>();
    auto dR_acc = dR.accessor<float, 4>();
    auto dS_acc = dS.accessor<float, 4>();
    auto dVq_2_acc = dVq_2.accessor<float, 4>();
    auto dVr_2_acc = dVr_2.accessor<float, 4>();

    int B = Q.size(0);
    int H = Q.size(1);
    int I = Q.size(2);
    int J = R.size(2);
    int K = S.size(2);
    int D = Q.size(3);
    float scale = 1.0f / std::sqrt(static_cast<float>(D));

    for (int b = 0; b < B; b++) {
        for (int h = 0; h < H; h++) {
            // For each k position
            for (int k = 0; k < K; k++) {
                // Process each dimension independently
                for (int d = 0; d < D; d++) {
                    float grad = grad_Y_s__acc[b][h][k][d];
                    
                    // Precompute Aq values for each (i,j) for fixed k
                    std::vector<std::vector<float>> Aq_values(I, std::vector<float>(J));
                    for (int i = 0; i < I; i++) {
                        std::vector<float> softmax_results(J * K);
                        compute_softmax_3d(
                            Q_acc, R_acc, S_acc,
                            b, h,
                            0, i,  // fixed i
                            J, K,  // dimensions j and k
                            1, 2, 0,  // map to j, k, i
                            scale, softmax_results.data()
                        );
                        
                        // Extract values for current k
                        for (int j = 0; j < J; j++) {
                            int idx = j * K + k;
                            Aq_values[i][j] = softmax_results[idx];
                        }
                    }
                    
                    // Precompute Ar values for each (j,i) for fixed k
                    std::vector<std::vector<float>> Ar_values(J, std::vector<float>(I));
                    for (int j = 0; j < J; j++) {
                        std::vector<float> softmax_results(I * K);
                        compute_softmax_3d(
                            Q_acc, R_acc, S_acc,
                            b, h,
                            1, j,  // fixed j
                            I, K,  // dimensions i and k
                            0, 2, 1,  // map to i, k, j
                            scale, softmax_results.data()
                        );
                        
                        // Extract values for current k
                        for (int i = 0; i < I; i++) {
                            int idx = i * K + k;
                            Ar_values[j][i] = softmax_results[idx];
                        }
                    }
                    
                    // Backpropagate to Vq_2 and Vr_2
                    for (int i = 0; i < I; i++) {
                        for (int j = 0; j < J; j++) {
                            float attn = Aq_values[i][j] * Ar_values[j][i];
                            dVq_2_acc[b][h][i][d] += grad * attn * Vr_2_acc[b][h][j][d];
                            dVr_2_acc[b][h][j][d] += grad * attn * Vq_2_acc[b][h][i][d];
                            
                            // We need gradients for Aq and Ar
                            float d_aq_ar = grad * Vq_2_acc[b][h][i][d] * Vr_2_acc[b][h][j][d];
                            
                            // Handle backprop through Aq (fixed i, softmax over j,k)
                            float d_aq = d_aq_ar * Ar_values[j][i];
                            std::vector<float> d_aq_softmax(J * K, 0.0f);
                            int aq_idx = j * K + k;
                            d_aq_softmax[aq_idx] = d_aq;
                            
                            std::vector<float> aq_softmax_outputs(J * K);
                            compute_softmax_3d(
                                Q_acc, R_acc, S_acc,
                                b, h,
                                0, i,  // fixed i
                                J, K,  // dimensions j and k
                                1, 2, 0,  // map to j, k, i
                                scale, aq_softmax_outputs.data()
                            );
                            
                            std::vector<float> d_aq_dot(J * K);
                            backprop_softmax_3d(
                                d_aq_dot.data(),
                                d_aq_softmax.data(),
                                aq_softmax_outputs.data(),
                                J * K
                            );
                            
                            // Handle backprop through Ar (fixed j, softmax over i,k)
                            float d_ar = d_aq_ar * Aq_values[i][j];
                            std::vector<float> d_ar_softmax(I * K, 0.0f);
                            int ar_idx = i * K + k;
                            d_ar_softmax[ar_idx] = d_ar;
                            
                            std::vector<float> ar_softmax_outputs(I * K);
                            compute_softmax_3d(
                                Q_acc, R_acc, S_acc,
                                b, h,
                                1, j,  // fixed j
                                I, K,  // dimensions i and k
                                0, 2, 1,  // map to i, k, j
                                scale, ar_softmax_outputs.data()
                            );
                            
                            std::vector<float> d_ar_dot(I * K);
                            backprop_softmax_3d(
                                d_ar_dot.data(),
                                d_ar_softmax.data(),
                                ar_softmax_outputs.data(),
                                I * K
                            );
                            
                            // Backpropagate to Q, R, S from Aq gradients
                            int aq_idx_iter = 0;
                            for (int j_aq = 0; j_aq < J; j_aq++) {
                                for (int k_aq = 0; k_aq < K; k_aq++) {
                                    float d_dot = d_aq_dot[aq_idx_iter++] * scale;
                                    for (int dd = 0; dd < D; dd++) {
                                        dQ_acc[b][h][i][dd] += d_dot * R_acc[b][h][j_aq][dd] * S_acc[b][h][k_aq][dd];
                                        dR_acc[b][h][j_aq][dd] += d_dot * Q_acc[b][h][i][dd] * S_acc[b][h][k_aq][dd];
                                        dS_acc[b][h][k_aq][dd] += d_dot * Q_acc[b][h][i][dd] * R_acc[b][h][j_aq][dd];
                                    }
                                }
                            }
                            
                            // Backpropagate to Q, R, S from Ar gradients
                            int ar_idx_iter = 0;
                            for (int i_ar = 0; i_ar < I; i_ar++) {
                                for (int k_ar = 0; k_ar < K; k_ar++) {
                                    float d_dot = d_ar_dot[ar_idx_iter++] * scale;
                                    for (int dd = 0; dd < D; dd++) {
                                        dQ_acc[b][h][i_ar][dd] += d_dot * R_acc[b][h][j][dd] * S_acc[b][h][k_ar][dd];
                                        dR_acc[b][h][j][dd] += d_dot * Q_acc[b][h][i_ar][dd] * S_acc[b][h][k_ar][dd];
                                        dS_acc[b][h][k_ar][dd] += d_dot * Q_acc[b][h][i_ar][dd] * R_acc[b][h][j][dd];
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }
}


std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, 
           torch::Tensor, torch::Tensor, torch::Tensor,
           torch::Tensor, torch::Tensor, torch::Tensor> hyper_attn_backward(
    const torch::Tensor& grad_Y_q,  // Gradient of loss w.r.t Y_q
    const torch::Tensor& grad_Y_r,  // Gradient of loss w.r.t Y_r
    const torch::Tensor& grad_Y_s,  // Gradient of loss w.r.t Y_s
    const torch::Tensor& grad_Y_q_, // Gradient of loss w.r.t Y_q_
    const torch::Tensor& grad_Y_r_, // Gradient of loss w.r.t Y_r_
    const torch::Tensor& grad_Y_s_, // Gradient of loss w.r.t Y_s_
    const torch::Tensor& Q,         // Forward inputs
    const torch::Tensor& R,
    const torch::Tensor& S,
    const torch::Tensor& Vq_1,
    const torch::Tensor& Vq_2,
    const torch::Tensor& Vr_1,
    const torch::Tensor& Vr_2,
    const torch::Tensor& Vs_1,
    const torch::Tensor& Vs_2,
    double dropout_rate = 0.0) 
{
    // Create gradient tensors initialized with zeros
    auto dQ = torch::zeros_like(Q);
    auto dR = torch::zeros_like(R);
    auto dS = torch::zeros_like(S);
    auto dVq_1 = torch::zeros_like(Vq_1);
    auto dVq_2 = torch::zeros_like(Vq_2);
    auto dVr_1 = torch::zeros_like(Vr_1);
    auto dVr_2 = torch::zeros_like(Vr_2);
    auto dVs_1 = torch::zeros_like(Vs_1);
    auto dVs_2 = torch::zeros_like(Vs_2);
    
    // Backpropagate through gather operations
    backprop_Y_gather(dQ, dR, dS, dVr_1, dVs_1, grad_Y_q, Q, R, S, Vr_1, Vs_1, 0);
    backprop_Y_gather(dQ, dR, dS, dVq_1, dVs_1, grad_Y_r, Q, R, S, Vq_1, Vs_1, 1);
    backprop_Y_gather(dQ, dR, dS, dVq_1, dVr_1, grad_Y_s, Q, R, S, Vq_1, Vr_1, 2);
    
    // Backpropagate through scatter operations
    backprop_Y_scatter_q(dQ, dR, dS, dVr_2, dVs_2, grad_Y_q_, Q, R, S, Vr_2, Vs_2);
    backprop_Y_scatter_r(dQ, dR, dS, dVq_2, dVs_2, grad_Y_r_, Q, R, S, Vq_2, Vs_2);
    backprop_Y_scatter_s(dQ, dR, dS, dVq_2, dVr_2, grad_Y_s_, Q, R, S, Vq_2, Vr_2);
    
    return std::make_tuple(dQ, dR, dS, dVq_1, dVq_2, dVr_1, dVr_2, dVs_1, dVs_2);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &hyper_attn_forward,
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

    m.def("backward", &hyper_attn_backward,
          "Hypergraph Attention backward (returns dQ, dR, dS, dVq_1, dVq_2, dVr_1, dVr_2, dVs_1, dVs_2)",
          py::arg("grad_Y_q"),
          py::arg("grad_Y_r"),
          py::arg("grad_Y_s"),
          py::arg("grad_Y_q_"),
          py::arg("grad_Y_r_"),
          py::arg("grad_Y_s_"),
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


