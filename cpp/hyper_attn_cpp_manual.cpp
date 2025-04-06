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

torch::Tensor forward_pass(
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


    return Y_q + Y_r + Y_s + Y_q_ + Y_r_ + Y_s_;
}

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
    float scale = 1.0f / std::sqrt(static_cast<float>(D));

    // Vq_1 contributes to both Y_r and Y_s in the forward pass
    // Backpropagate through both of these contributions

    // 1. Contribution from Y_r (gather to position j)
    // Y_r = compute_Y_gather(Q, R, S, Vq_1, Vs_1, 1)
    for (int b = 0; b < B; b++) {
        for (int h = 0; h < H; h++) {
            // For each j (fixed dimension in Y_r)
            for (int j = 0; j < J; j++) {
                // We need the attention weights Ar_j for fixed j
                // (softmax over i,k dimensions for this fixed j)
                std::vector<float> Ar_j_values(I * K); // Will store the attention weights
                
                // Compute softmax for this (b,h,j) using the same function as forward pass
                compute_softmax_3d(
                    Q_acc, R_acc, S_acc,
                    b, h,
                    1, j,  // fixed_dim = 1 (j), fixed_idx = j
                    I, K,  // dim1_size = I, dim2_size = K
                    0, 2, 1,  // dim1_idx_fn = 0 (i), dim2_idx_fn = 2 (k), fixed_dim_idx_fn = 1 (j)
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
                            
                            // Add to the gradient accumulator
                            grad_Vq_1_acc[b][h][i][d] += contribution;
                        }
                    }
                }
            }
            
            // 2. Contribution from Y_s (gather to position k)
            // Y_s = compute_Y_gather(Q, R, S, Vq_1, Vr_1, 2)
            for (int k = 0; k < K; k++) {
                // We need the attention weights As_k for fixed k
                // (softmax over i,j dimensions for this fixed k)
                std::vector<float> As_k_values(I * J); // Will store the attention weights
                
                // Compute softmax for this (b,h,k) using the same function as forward pass
                compute_softmax_3d(
                    Q_acc, R_acc, S_acc,
                    b, h,
                    2, k,  // fixed_dim = 2 (k), fixed_idx = k
                    I, J,  // dim1_size = I, dim2_size = J
                    0, 1, 2,  // dim1_idx_fn = 0 (i), dim2_idx_fn = 1 (j), fixed_dim_idx_fn = 2 (k)
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
    torch::Tensor& grad_Vr_1,        // Output gradient tensor for Vr_1
    const torch::Tensor& grad_output, // Incoming gradient from next layer
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
    // Backpropagate through both of these contributions

    // 1. Contribution from Y_q (gather to position i)
    // Y_q = compute_Y_gather(Q, R, S, Vr_1, Vs_1, 0)
    for (int b = 0; b < B; b++) {
        for (int h = 0; h < H; h++) {
            // For each i (fixed dimension in Y_q)
            for (int i = 0; i < I; i++) {
                // We need the attention weights Aq_i for fixed i
                // (softmax over j,k dimensions for this fixed i)
                std::vector<float> Aq_i_values(J * K);
                
                // Compute softmax for this (b,h,i)
                compute_softmax_3d(
                    Q_acc, R_acc, S_acc,
                    b, h,
                    0, i,  // fixed_dim = 0 (i), fixed_idx = i
                    J, K,  // dim1_size = J, dim2_size = K
                    1, 2, 0,  // dim1_idx_fn = 1 (j), dim2_idx_fn = 2 (k), fixed_dim_idx_fn = 0 (i)
                    scale, Aq_i_values.data()
                );
                
                // Propagate gradient for each dimension
                for (int d = 0; d < D; d++) {
                    // Gradient coming from Y_q at this position
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
                // We need the attention weights As_k for fixed k
                // (softmax over i,j dimensions for this fixed k)
                std::vector<float> As_k_values(I * J);
                
                // Compute softmax for this (b,h,k)
                compute_softmax_3d(
                    Q_acc, R_acc, S_acc,
                    b, h,
                    2, k,  // fixed_dim = 2 (k), fixed_idx = k
                    I, J,  // dim1_size = I, dim2_size = J
                    0, 1, 2,  // dim1_idx_fn = 0 (i), dim2_idx_fn = 1 (j), fixed_dim_idx_fn = 2 (k)
                    scale, As_k_values.data()
                );
                
                // Propagate gradient for each dimension
                for (int d = 0; d < D; d++) {
                    // Gradient coming from Y_s at this position
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
                    0, i,  // fixed_dim = 0 (i), fixed_idx = i
                    J, K,  // dim1_size = J, dim2_size = K
                    1, 2, 0,  // dim1_idx_fn = 1 (j), dim2_idx_fn = 2 (k), fixed_dim_idx_fn = 0 (i)
                    scale, Aq_i_values.data()
                );
                
                for (int d = 0; d < D; d++) {
                    float dy_q = grad_output_acc[b][h][i][d];
                    
                    for (int j = 0; j < J; j++) {
                        for (int k = 0; k < K; k++) {
                            int attn_idx = j * K + k;  // Changed index calculation
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
                    1, j,  // fixed_dim = 1 (j), fixed_idx = j
                    I, K,  // dim1_size = I, dim2_size = K
                    0, 2, 1,  // dim1_idx_fn = 0 (i), dim2_idx_fn = 2 (k), fixed_dim_idx_fn = 1 (j)
                    scale, Ar_j_values.data()
                );
                
                for (int d = 0; d < D; d++) {
                    float dy_r = grad_output_acc[b][h][j][d];
                    
                    for (int i = 0; i < I; i++) {
                        for (int k = 0; k < K; k++) {
                            int attn_idx = i * K + k;  // Changed index calculation
                            float attn = Ar_j_values[attn_idx];
                            grad_Vs_1_acc[b][h][k][d] += dy_r * attn * Vq_1_acc[b][h][i][d];
                        }
                    }
                }
            }
        }
    }
}


std::tuple<torch::Tensor, torch::Tensor, torch::Tensor,  
          torch::Tensor, torch::Tensor,                  
          torch::Tensor, torch::Tensor,                  
          torch::Tensor, torch::Tensor>                  
backward_pass(
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
    double dropout_rate = 0.0)
{
    // Create tensors to store gradients
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

    
    return std::make_tuple(
        grad_Q, grad_R, grad_S,
        grad_Vq_1, grad_Vq_2,
        grad_Vr_1, grad_Vr_2,
        grad_Vs_1, grad_Vs_2
    );
}  
    

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward_pass,
          "Hypergraph Attention forward (returns sum of all Y tensors)",
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

    m.def("backward", &backward_pass,
          "Hypergraph Attention backward (returns dQ, dR, dS, dVq_1, dVq_2, dVr_1, dVr_2, dVs_1, dVs_2)",
          py::arg("gradients"),
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


