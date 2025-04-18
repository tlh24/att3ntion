#include <torch/extension.h> 
#include <iostream> 
#include <cmath> 
#include <limits>
#include <vector> 
#include <tuple>  

//helper: compute dot product between three vectors at specific indices
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

//helper: compute 3D softmax with configurable dimensions
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
    int fixed_dim              //dimension to fix (0=i/q, 1=j/r, 2=k/s)
) {
    std::cout << "Entering compute_Y_gather (fixed_dim=" << fixed_dim << ")" << std::endl;
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
    int sizes[3] = {I, J, K}; // dim sizes

    for (int b = 0; b < B; b++) {
        for (int h = 0; h < H; h++) {
            for (int fixed_idx = 0; fixed_idx < sizes[fixed_dim]; fixed_idx++) {
            
                // Define variables for the other two dimensions based on the fixed dimension
                int dim1, dim2;
                int dim1_size, dim2_size;
                
                if (fixed_dim == 0) {      // If i is fixed
                    dim1 = 1; dim2 = 2;    // j and k are the other dimensions
                    dim1_size = J; dim2_size = K;
                } else if (fixed_dim == 1) {    
                    dim1 = 0; dim2 = 2;    
                    dim1_size = I; dim2_size = K;
                } else {                   
                    dim1 = 0; dim2 = 1;    
                    dim1_size = I; dim2_size = J;
                }
                
                //compute softmax without materializing full tensor
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
                            
                            float dot_prod = compute_dot_product(Q_acc, R_acc, S_acc, b, h, i, j, k, D);
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
                            
                            // Compute dot product using the helper function
                            float dot_prod = compute_dot_product(Q_acc, R_acc, S_acc, b, h, i, j, k, D);
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
                            
                            // Compute dot product using the helper function
                            float dot_prod = compute_dot_product(Q_acc, R_acc, S_acc, b, h, i, j, k, D);
                            dot_prod *= scale;
                            
                            // Compute softmax value and multiply by values
                            // Avoid division by zero if sum_exp is zero or very close to it
                            float attn = (sum_exp > 1e-9) ? (std::exp(dot_prod - max_val) / sum_exp) : 0.0f;
                            Y_out_acc[b][h][fixed_idx][d] += attn * V1_acc[b][h][v1_idx][d] * V2_acc[b][h][v2_idx][d];
                        }
                    }
                }
            }
        }
    }
    std::cout << "Exiting compute_Y_gather" << std::endl;
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
                //Compute the effect of ArAs * Vr_2 * Vs_2
                // For each j, compute attention to js (softmax over i,k)
                std::vector<std::vector<float>> Ar_values(J, std::vector<float>(K, 0.0f));
                for (int j = 0; j < J; j++) {
                    //Compute Ar values (softmax over i,k for fixed j)
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
            for (int j = 0; j < J; j++) {
                
                std::vector<std::vector<float>> Aq_values(I, std::vector<float>(K, 0.0f));
                for (int i = 0; i < I; i++) {
                    std::vector<float> softmax_results(J * K);
                    compute_softmax_3d(
                        Q_acc, R_acc, S_acc,
                        b, h,
                        0, i,  
                        J, K,  
                        1, 2, 0,  // dim1_idx_fn = 1 (j), dim2_idx_fn = 2 (k), fixed_dim_idx_fn = 0 (i)
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
    if (fixed_dim == 0) { // Compute Aq[b,h,i_target,j_target,k_target] (softmax over j,k for fixed i_target)
        for (int j = 0; j < J; ++j) {
            for (int k = 0; k < K; ++k) {
                float dot = compute_dot_product(Q_acc, R_acc, S_acc, b, h, i_target, j, k, D);
                max_val = std::max(max_val, dot * scale);
            }
        }
    } else if (fixed_dim == 1) { // Compute Ar[b,h,i_target,j_target,k_target] (softmax over i,k for fixed j_target)
        for (int i = 0; i < I; ++i) {
            for (int k = 0; k < K; ++k) {
                float dot = compute_dot_product(Q_acc, R_acc, S_acc, b, h, i, j_target, k, D);
                max_val = std::max(max_val, dot * scale);
            }
        }
    } else { // fixed_dim == 2: Compute As[b,h,i_target,j_target,k_target] (softmax over i,j for fixed k_target)
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

    // --- Compute final softmax value for the target indices ---
    float target_dot = compute_dot_product(Q_acc, R_acc, S_acc, b, h, i_target, j_target, k_target, D);
    if (sum_exp == 0.0f) { // Avoid division by zero
        // This might happen if all dot products were extremely small or negative infinity
        // Determine the number of elements in the softmax normalization
        int num_elements = 0;
        if (fixed_dim == 0) num_elements = J * K;
        else if (fixed_dim == 1) num_elements = I * K;
        else num_elements = I * J;
        // Return uniform probability if sum_exp is zero
        return 1.0f / static_cast<float>(num_elements);
    }
    return std::exp(target_dot * scale - max_val) / sum_exp;
}

void compute_grad_Vq_2(
    torch::Tensor& grad_Vq_2,        // Output: Gradient w.r.t. Vq_2 [B, H, I, D]
    const torch::Tensor& grad_output, // Input: Gradient w.r.t. final output Y [B, H, max(I,J,K), D]
    const torch::Tensor& Q,           // Input: Q tensor [B, H, I, D]
    const torch::Tensor& R,           // Input: R tensor [B, H, J, D]
    const torch::Tensor& S,           // Input: S tensor [B, H, K, D]
    const torch::Tensor& Vr_2,        // Input: Vr_2 tensor [B, H, J, D]
    const torch::Tensor& Vs_2,        // Input: Vs_2 tensor [B, H, K, D]
    double dropout_rate = 0.0)        // Note: Dropout not handled in this specific grad computation
{
    // Get tensor accessors
    auto grad_Vq_2_acc = grad_Vq_2.accessor<float, 4>();
    auto grad_output_acc = grad_output.accessor<float, 4>();
    auto Q_acc = Q.accessor<float, 4>();
    auto R_acc = R.accessor<float, 4>();
    auto S_acc = S.accessor<float, 4>();
    auto Vr_2_acc = Vr_2.accessor<float, 4>();
    auto Vs_2_acc = Vs_2.accessor<float, 4>();

    // Get dimensions
    const int B = Q.size(0);
    const int H = Q.size(1);
    const int I = Q.size(2);
    const int J = R.size(2);
    const int K = S.size(2);
    const int D = Q.size(3);
    const float scale = 1.0f / std::sqrt(static_cast<float>(D));

    // grad_Vq_2 should be zero-initialized before calling this function

    // Iterate over batch and head
    for (int b = 0; b < B; ++b) {
        for (int h = 0; h < H; ++h) {

            // --- Contribution from Y_r_ Path ---
            // dL/dVq_2[i] += sum_{j} (dL/dY_r_[j] * dY_r_[j]/dVq_2[i])
            // dY_r_[j]/dVq_2[i] = sum_{k} (Aq[i,j,k] * As[i,j,k] * Vs_2[k])
            for (int j = 0; j < J; ++j) { // Loop over source gradient index (Y_r_)
                for (int d = 0; d < D; ++d) {
                    // Gradient coming from Y_r_ at index j, dimension d
                    // Ensure grad_output access is within bounds if J < max_seq_len
                    if (j >= grad_output.size(2)) continue;
                    const float dy_r = grad_output_acc[b][h][j][d];

                    if (dy_r == 0.0f) continue; // Optimization: skip if gradient is zero

                    for (int i = 0; i < I; ++i) { // Loop over target gradient index (Vq_2)
                        for (int k = 0; k < K; ++k) { // Summation index
                            // Recompute necessary attention scores on the fly
                            float attn_aq = compute_single_softmax_attn(Q_acc, R_acc, S_acc, b, h, i, j, k, I, J, K, D, scale, 0); // Softmax over j,k
                            float attn_as = compute_single_softmax_attn(Q_acc, R_acc, S_acc, b, h, i, j, k, I, J, K, D, scale, 2); // Softmax over i,j

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

                     for (int j = 0; j < J; ++j) { // Loop over target gradient index (Vq_2)
                         for (int i = 0; i < I; ++i) { // Summation index
                             // Recompute necessary attention scores on the fly
                             float attn_aq = compute_single_softmax_attn(Q_acc, R_acc, S_acc, b, h, i, j, k, I, J, K, D, scale, 0); // Softmax over j,k
                             float attn_ar = compute_single_softmax_attn(Q_acc, R_acc, S_acc, b, h, i, j, k, I, J, K, D, scale, 1); // Softmax over i,k

                             float vr2_val = Vr_2_acc[b][h][j][d];

                             // Accumulate gradient: dL/dY_s * Aq * Ar * Vr_2
                             grad_Vq_2_acc[b][h][i][d] += dy_s * attn_aq * attn_ar * vr2_val;
                         }
                     }
                 }
            }
        } // end head loop
    } // end batch loop
}

void compute_grad_Vr_2(
    torch::Tensor& grad_Vr_2,        // Output: Gradient w.r.t. Vr_2 [B, H, J, D]
    const torch::Tensor& grad_output, // Input: Gradient w.r.t. final output Y [B, H, max(I,J,K), D]
    const torch::Tensor& Q,           // Input: Q tensor [B, H, I, D]
    const torch::Tensor& R,           // Input: R tensor [B, H, J, D]
    const torch::Tensor& S,           // Input: S tensor [B, H, K, D]
    const torch::Tensor& Vq_2,        // Input: Vq_2 tensor [B, H, I, D]
    const torch::Tensor& Vs_2,        // Input: Vs_2 tensor [B, H, K, D]
    double dropout_rate = 0.0)        // Note: Dropout not handled in this specific grad computation
{
    // Get tensor accessors
    auto grad_Vr_2_acc = grad_Vr_2.accessor<float, 4>();
    auto grad_output_acc = grad_output.accessor<float, 4>();
    auto Q_acc = Q.accessor<float, 4>();
    auto R_acc = R.accessor<float, 4>();
    auto S_acc = S.accessor<float, 4>();
    auto Vq_2_acc = Vq_2.accessor<float, 4>();
    auto Vs_2_acc = Vs_2.accessor<float, 4>();

    // Get dimensions
    const int B = Q.size(0);
    const int H = Q.size(1);
    const int I = Q.size(2);
    const int J = R.size(2);
    const int K = S.size(2);
    const int D = Q.size(3);
    const float scale = 1.0f / std::sqrt(static_cast<float>(D));

    // grad_Vr_2 should be zero-initialized before calling this function

    // Iterate over batch and head
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

                    if (dy_s == 0.0f) continue; // Optimization

                    for (int j = 0; j < J; ++j) { // Loop over target gradient index (Vr_2)
                        for (int i = 0; i < I; ++i) { // Summation index
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
        } // end head loop
    } // end batch loop
}

void compute_grad_Vs_2(
    torch::Tensor& grad_Vs_2,        // Output: Gradient w.r.t. Vs_2 [B, H, K, D]
    const torch::Tensor& grad_output, // Input: Gradient w.r.t. final output Y [B, H, max(I,J,K), D]
    const torch::Tensor& Q,           // Input: Q tensor [B, H, I, D]
    const torch::Tensor& R,           // Input: R tensor [B, H, J, D]
    const torch::Tensor& S,           // Input: S tensor [B, H, K, D]
    const torch::Tensor& Vq_2,        // Input: Vq_2 tensor [B, H, I, D]
    const torch::Tensor& Vr_2,        // Input: Vr_2 tensor [B, H, J, D]
    double dropout_rate = 0.0)        // Note: Dropout not handled in this specific grad computation
{
    // Get tensor accessors
    auto grad_Vs_2_acc = grad_Vs_2.accessor<float, 4>();
    auto grad_output_acc = grad_output.accessor<float, 4>();
    auto Q_acc = Q.accessor<float, 4>();
    auto R_acc = R.accessor<float, 4>();
    auto S_acc = S.accessor<float, 4>();
    auto Vq_2_acc = Vq_2.accessor<float, 4>();
    auto Vr_2_acc = Vr_2.accessor<float, 4>();

    // Get dimensions
    const int B = Q.size(0);
    const int H = Q.size(1);
    const int I = Q.size(2);
    const int J = R.size(2);
    const int K = S.size(2);
    const int D = Q.size(3);
    const float scale = 1.0f / std::sqrt(static_cast<float>(D));

    // grad_Vs_2 should be zero-initialized before calling this function

    // Iterate over batch and head
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

                    if (dy_q == 0.0f) continue; // Optimization: skip if gradient is zero

                    for (int k = 0; k < K; ++k) { // Loop over target gradient index (Vs_2)
                        for (int j = 0; j < J; ++j) { // Summation index
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

                    if (dy_r == .0f) continue; // Optimization

                    for (int k = 0; k < K; ++k) { // Loop over target gradient index (Vs_2)
                        for (int i = 0; i < I; ++i) { // Summation index
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
        } // end head loop
    } // end batch loop
}


// Helper Function to compute full Attention Tensors (Aq, Ar, As)
// Needed for grad_Q, grad_R, grad_S computation (for now)
std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor> compute_attention_tensors(
    const torch::Tensor& Q,
    const torch::Tensor& R,
    const torch::Tensor& S
) {
    int B = Q.size(0);
    int H = Q.size(1);
    int I = Q.size(2);
    int J = R.size(2);
    int K = S.size(2);
    int D = Q.size(3);
    float scale = 1.0f / std::sqrt(static_cast<float>(D));
    auto options = Q.options();

    // 1. Compute the scaled dot-product tensor P (bhijk)
    // Note: This materializes the potentially large P tensor.
    auto P = torch::zeros({B, H, I, J, K}, options);
    auto P_acc = P.accessor<float, 5>();
    auto Q_acc = Q.accessor<float, 4>();
    auto R_acc = R.accessor<float, 4>();
    auto S_acc = S.accessor<float, 4>();

    for (int b = 0; b < B; ++b) {
        for (int h = 0; h < H; ++h) {
            for (int i = 0; i < I; ++i) {
                for (int j = 0; j < J; ++j) {
                    for (int k = 0; k < K; ++k) {
                        P_acc[b][h][i][j][k] = compute_dot_product(Q_acc, R_acc, S_acc, b, h, i, j, k, D) * scale;
                    }
                }
            }
        }
    }

    // 2. Compute Aq (softmax over j, k for fixed i)
    auto Aq = torch::softmax(P.reshape({B, H, I, J * K}), /*dim=*/3).reshape({B, H, I, J, K});

    // 3. Compute Ar (softmax over i, k for fixed j)
    // Permute P: bhijk -> bhjik
    auto P_r_permuted = P.permute({0, 1, 3, 2, 4}).contiguous(); // contiguous() might be needed for reshape/softmax
    auto Ar_permuted = torch::softmax(P_r_permuted.reshape({B, H, J, I * K}), /*dim=*/3).reshape({B, H, J, I, K});
    // Permute Ar back: bhjik -> bhijk
    auto Ar = Ar_permuted.permute({0, 1, 3, 2, 4});

    // 4. Compute As (softmax over i, j for fixed k)
    // Permute P: bhijk -> bhkij
    auto P_s_permuted = P.permute({0, 1, 4, 2, 3}).contiguous();
    auto As_permuted = torch::softmax(P_s_permuted.reshape({B, H, K, I * J}), /*dim=*/3).reshape({B, H, K, I, J});
    // Permute As back: bhkij -> bhijk
    auto As = As_permuted.permute({0, 1, 3, 4, 2});

    return std::make_tuple(P, Aq, Ar, As); // Return P as well, useful for grad_Q
}


// Helper Function to compute Gradient w.r.t. Scaled Dot Product (P)
// This performs Phases 1 and 2 of the grad_Q/R/S calculation plan.
torch::Tensor compute_grad_P(
    const torch::Tensor& grad_output, // Input: Gradient w.r.t. final output Y [B, H, N, D]
    const torch::Tensor& Q,
    const torch::Tensor& R,
    const torch::Tensor& S,
    const torch::Tensor& Vq_1,
    const torch::Tensor& Vq_2,
    const torch::Tensor& Vr_1,
    const torch::Tensor& Vr_2,
    const torch::Tensor& Vs_1,
    const torch::Tensor& Vs_2,
    const torch::Tensor& P,           // Precomputed scaled dot-product tensor
    const torch::Tensor& Aq,          // Precomputed attention Aq
    const torch::Tensor& Ar,          // Precomputed attention Ar
    const torch::Tensor& As           // Precomputed attention As
) {
    // Get dimensions
    const int B = Q.size(0);
    const int H = Q.size(1);
    const int I = Q.size(2);
    const int J = R.size(2);
    const int K = S.size(2);
    const int D = Q.size(3);
    const int N = grad_output.size(2); // Max sequence length from output grad
    auto options = Q.options();

    // Accessors for inputs
    auto grad_output_acc = grad_output.accessor<float, 4>();
    auto Vq_1_acc = Vq_1.accessor<float, 4>();
    auto Vq_2_acc = Vq_2.accessor<float, 4>();
    auto Vr_1_acc = Vr_1.accessor<float, 4>();
    auto Vr_2_acc = Vr_2.accessor<float, 4>();
    auto Vs_1_acc = Vs_1.accessor<float, 4>();
    auto Vs_2_acc = Vs_2.accessor<float, 4>();

    // Accessors for precomputed attention weights
    auto Aq_acc = Aq.accessor<float, 5>();
    auto Ar_acc = Ar.accessor<float, 5>();
    auto As_acc = As.accessor<float, 5>();

    // --- Phase 1: Compute Gradients w.r.t. Attention Weights ---
    auto grad_Aq = torch::zeros_like(Aq);
    auto grad_Ar = torch::zeros_like(Ar);
    auto grad_As = torch::zeros_like(As);

    auto grad_Aq_acc = grad_Aq.accessor<float, 5>();
    auto grad_Ar_acc = grad_Ar.accessor<float, 5>();
    auto grad_As_acc = grad_As.accessor<float, 5>();

    // Using loops to mimic einsum (same as before)
    for (int b = 0; b < B; ++b) {
        for (int h = 0; h < H; ++h) {
            // 1.a) grad_Aq from Yq
            if (I <= N) {
                for (int i = 0; i < I; ++i) { for (int j = 0; j < J; ++j) { for (int k = 0; k < K; ++k) { for (int d = 0; d < D; ++d) {
                    grad_Aq_acc[b][h][i][j][k] += grad_output_acc[b][h][i][d] * Vr_1_acc[b][h][j][d] * Vs_1_acc[b][h][k][d];
                }}}}
            }
            // 1.b) grad_Aq from Yr'
             if (J <= N) {
                for (int i = 0; i < I; ++i) { for (int j = 0; j < J; ++j) { for (int k = 0; k < K; ++k) {
                    float grad_x_vals = 0.0f;
                    for (int d = 0; d < D; ++d) { grad_x_vals += grad_output_acc[b][h][j][d] * Vq_2_acc[b][h][i][d] * Vs_2_acc[b][h][k][d]; }
                    grad_Aq_acc[b][h][i][j][k] += grad_x_vals * As_acc[b][h][i][j][k];
                }}}
            }
            // 1.c) grad_Aq from Ys'
            if (K <= N) {
                for (int i = 0; i < I; ++i) { for (int j = 0; j < J; ++j) { for (int k = 0; k < K; ++k) {
                    float grad_x_vals = 0.0f;
                    for (int d = 0; d < D; ++d) { grad_x_vals += grad_output_acc[b][h][k][d] * Vq_2_acc[b][h][i][d] * Vr_2_acc[b][h][j][d]; }
                    grad_Aq_acc[b][h][i][j][k] += grad_x_vals * Ar_acc[b][h][i][j][k];
                }}}
            }

            // 2.a) grad_Ar from Yr
            if (J <= N) {
                for (int i = 0; i < I; ++i) { for (int j = 0; j < J; ++j) { for (int k = 0; k < K; ++k) { for (int d = 0; d < D; ++d) {
                    grad_Ar_acc[b][h][i][j][k] += grad_output_acc[b][h][j][d] * Vq_1_acc[b][h][i][d] * Vs_1_acc[b][h][k][d];
                }}}}
            }
             // 2.b) grad_Ar from Yq'
            if (I <= N) {
                 for (int i = 0; i < I; ++i) { for (int j = 0; j < J; ++j) { for (int k = 0; k < K; ++k) {
                    float grad_x_vals = 0.0f;
                    for (int d = 0; d < D; ++d) { grad_x_vals += grad_output_acc[b][h][i][d] * Vr_2_acc[b][h][j][d] * Vs_2_acc[b][h][k][d]; }
                    grad_Ar_acc[b][h][i][j][k] += grad_x_vals * As_acc[b][h][i][j][k];
                }}}
            }
             // 2.c) grad_Ar from Ys'
             if (K <= N) {
                 for (int i = 0; i < I; ++i) { for (int j = 0; j < J; ++j) { for (int k = 0; k < K; ++k) {
                    float grad_x_vals = 0.0f;
                    for (int d = 0; d < D; ++d) { grad_x_vals += grad_output_acc[b][h][k][d] * Vq_2_acc[b][h][i][d] * Vr_2_acc[b][h][j][d]; }
                    grad_Ar_acc[b][h][i][j][k] += grad_x_vals * Aq_acc[b][h][i][j][k];
                }}}
            }

            // 3.a) grad_As from Ys
            if (K <= N) {
                for (int i = 0; i < I; ++i) { for (int j = 0; j < J; ++j) { for (int k = 0; k < K; ++k) { for (int d = 0; d < D; ++d) {
                    grad_As_acc[b][h][i][j][k] += grad_output_acc[b][h][k][d] * Vq_1_acc[b][h][i][d] * Vr_1_acc[b][h][j][d];
                }}}}
            }
             // 3.b) grad_As from Yq'
             if (I <= N) {
                for (int i = 0; i < I; ++i) { for (int j = 0; j < J; ++j) { for (int k = 0; k < K; ++k) {
                    float grad_x_vals = 0.0f;
                    for (int d = 0; d < D; ++d) { grad_x_vals += grad_output_acc[b][h][i][d] * Vr_2_acc[b][h][j][d] * Vs_2_acc[b][h][k][d]; }
                    grad_As_acc[b][h][i][j][k] += grad_x_vals * Ar_acc[b][h][i][j][k];
                }}}
             }
             // 3.c) grad_As from Yr'
             if (J <= N) {
                for (int i = 0; i < I; ++i) { for (int j = 0; j < J; ++j) { for (int k = 0; k < K; ++k) {
                    float grad_x_vals = 0.0f;
                    for (int d = 0; d < D; ++d) { grad_x_vals += grad_output_acc[b][h][j][d] * Vq_2_acc[b][h][i][d] * Vs_2_acc[b][h][k][d]; }
                    grad_As_acc[b][h][i][j][k] += grad_x_vals * Aq_acc[b][h][i][j][k];
                }}}
            }
        } // end head
    } // end batch

    // --- Phase 2: Propagate Gradients Back Through Softmax to get grad_P ---
    auto grad_P = torch::zeros_like(P);
    auto grad_P_acc = grad_P.accessor<float, 5>();

    // 2.1 Contribution from Aq (Softmax over j, k)
    for (int b = 0; b < B; ++b) { for (int h = 0; h < H; ++h) { for (int i = 0; i < I; ++i) {
        float sum_q = 0.0f;
        for (int j = 0; j < J; ++j) { for (int k = 0; k < K; ++k) { sum_q += grad_Aq_acc[b][h][i][j][k] * Aq_acc[b][h][i][j][k]; }}
        for (int j = 0; j < J; ++j) { for (int k = 0; k < K; ++k) { grad_P_acc[b][h][i][j][k] += (grad_Aq_acc[b][h][i][j][k] - sum_q) * Aq_acc[b][h][i][j][k]; }}
    }}}

    // 2.2 Contribution from Ar (Softmax over i, k)
    for (int b = 0; b < B; ++b) { for (int h = 0; h < H; ++h) { for (int j = 0; j < J; ++j) {
        float sum_r = 0.0f;
        for (int i = 0; i < I; ++i) { for (int k = 0; k < K; ++k) { sum_r += grad_Ar_acc[b][h][i][j][k] * Ar_acc[b][h][i][j][k]; }}
        for (int i = 0; i < I; ++i) { for (int k = 0; k < K; ++k) { grad_P_acc[b][h][i][j][k] += (grad_Ar_acc[b][h][i][j][k] - sum_r) * Ar_acc[b][h][i][j][k]; }}
    }}}

    // 2.3 Contribution from As (Softmax over i, j)
    for (int b = 0; b < B; ++b) { for (int h = 0; h < H; ++h) { for (int k = 0; k < K; ++k) {
        float sum_s = 0.0f;
        for (int i = 0; i < I; ++i) { for (int j = 0; j < J; ++j) { sum_s += grad_As_acc[b][h][i][j][k] * As_acc[b][h][i][j][k]; }}
        for (int i = 0; i < I; ++i) { for (int j = 0; j < J; ++j) { grad_P_acc[b][h][i][j][k] += (grad_As_acc[b][h][i][j][k] - sum_s) * As_acc[b][h][i][j][k]; }}
    }}}

    return grad_P;
}


void compute_grad_Q(
    torch::Tensor& grad_Q,          // Output: Gradient w.r.t. Q [B, H, I, D]
    const torch::Tensor& grad_P,    // Input: Gradient w.r.t. P [B, H, I, J, K]
    const torch::Tensor& R,
    const torch::Tensor& S,
    float scale                       // Input: Scaling factor 1/sqrt(D)
) {
    const int B = grad_Q.size(0);
    const int H = grad_Q.size(1);
    const int I = grad_Q.size(2);
    const int J = R.size(2);
    const int K = S.size(2);
    const int D = grad_Q.size(3);

    auto grad_Q_acc = grad_Q.accessor<float, 4>();
    auto grad_P_acc = grad_P.accessor<float, 5>();
    auto R_acc = R.accessor<float, 4>();
    auto S_acc = S.accessor<float, 4>();

    // --- Phase 3: Compute grad_Q from grad_P ---
    // grad_Q = scale * einsum("bhijk,bhjd,bhkd->bhid", grad_P, R, S)
    for (int b = 0; b < B; ++b) {
        for (int h = 0; h < H; ++h) {
            for (int i = 0; i < I; ++i) {
                for (int d = 0; d < D; ++d) {
                    float sum_for_grad_q = 0.0f;
                    for (int j = 0; j < J; ++j) {
                        for (int k = 0; k < K; ++k) {
                             sum_for_grad_q += grad_P_acc[b][h][i][j][k] * R_acc[b][h][j][d] * S_acc[b][h][k][d];
                        }
                    }
                    grad_Q_acc[b][h][i][d] = scale * sum_for_grad_q; // Assign result (not accumulate)
                }
            }
        }
    }
}

void compute_grad_R(
    torch::Tensor& grad_R,          // Output: Gradient w.r.t. R [B, H, J, D]
    const torch::Tensor& grad_P,    // Input: Gradient w.r.t. P [B, H, I, J, K]
    const torch::Tensor& Q,
    const torch::Tensor& S,
    float scale                       // Input: Scaling factor 1/sqrt(D)
) {
    const int B = grad_R.size(0);
    const int H = grad_R.size(1);
    const int J = grad_R.size(2); // Target dim J
    const int I = Q.size(2);      // Other dims I, K
    const int K = S.size(2);
    const int D = grad_R.size(3);

    auto grad_R_acc = grad_R.accessor<float, 4>();
    auto grad_P_acc = grad_P.accessor<float, 5>();
    auto Q_acc = Q.accessor<float, 4>();
    auto S_acc = S.accessor<float, 4>();

    // grad_R = scale * einsum("bhijk,bhid,bhkd->bhjd", grad_P, Q, S)
    for (int b = 0; b < B; ++b) {
        for (int h = 0; h < H; ++h) {
            for (int j = 0; j < J; ++j) { // Loop over the target dimension J
                for (int d = 0; d < D; ++d) {
                    float sum_for_grad_r = 0.0f;
                    for (int i = 0; i < I; ++i) { // Sum over other dimensions I, K
                        for (int k = 0; k < K; ++k) {
                             sum_for_grad_r += grad_P_acc[b][h][i][j][k] * Q_acc[b][h][i][d] * S_acc[b][h][k][d];
                        }
                    }
                    grad_R_acc[b][h][j][d] = scale * sum_for_grad_r; // Assign result
                }
            }
        }
    }
}

void compute_grad_S(
    torch::Tensor& grad_S,          // Output: Gradient w.r.t. S [B, H, K, D]
    const torch::Tensor& grad_P,    // Input: Gradient w.r.t. P [B, H, I, J, K]
    const torch::Tensor& Q,
    const torch::Tensor& R,
    float scale                       // Input: Scaling factor 1/sqrt(D)
) {
    const int B = grad_S.size(0);
    const int H = grad_S.size(1);
    const int K = grad_S.size(2); // Target dim K
    const int I = Q.size(2);      // Other dims I, J
    const int J = R.size(2);
    const int D = grad_S.size(3);

    auto grad_S_acc = grad_S.accessor<float, 4>();
    auto grad_P_acc = grad_P.accessor<float, 5>();
    auto Q_acc = Q.accessor<float, 4>();
    auto R_acc = R.accessor<float, 4>();

    // grad_S = scale * einsum("bhijk,bhid,bhjd->bhkd", grad_P, Q, R)
    for (int b = 0; b < B; ++b) {
        for (int h = 0; h < H; ++h) {
            for (int k = 0; k < K; ++k) { // Loop over the target dimension K
                for (int d = 0; d < D; ++d) {
                    float sum_for_grad_s = 0.0f;
                    for (int i = 0; i < I; ++i) { // Sum over other dimensions I, J
                        for (int j = 0; j < J; ++j) {
                             sum_for_grad_s += grad_P_acc[b][h][i][j][k] * Q_acc[b][h][i][d] * R_acc[b][h][j][d];
                        }
                    }
                    grad_S_acc[b][h][k][d] = scale * sum_for_grad_s; // Assign result
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
    double dropout_rate = 0.0) // Dropout not handled in backward yet
{
    // Get scale factor
    const int D = Q.size(3);
    const float scale = 1.0f / std::sqrt(static_cast<float>(D));

    // Create tensors to store gradients, initialized to zero
    auto grad_Q = torch::zeros_like(Q);
    auto grad_R = torch::zeros_like(R);
    auto grad_S = torch::zeros_like(S);
    auto grad_Vq_1 = torch::zeros_like(Vq_1);
    auto grad_Vq_2 = torch::zeros_like(Vq_2);
    auto grad_Vr_1 = torch::zeros_like(Vr_1);
    auto grad_Vr_2 = torch::zeros_like(Vr_2);
    auto grad_Vs_1 = torch::zeros_like(Vs_1);
    auto grad_Vs_2 = torch::zeros_like(Vs_2);

    // --- Precompute Attention Tensors ---
    torch::Tensor P, Aq, Ar, As;
    std::tie(P, Aq, Ar, As) = compute_attention_tensors(Q, R, S);

    // --- Compute Gradient w.r.t. P ---
    // This gradient is needed for dQ, dR, dS
    auto grad_P = compute_grad_P(
        grad_output, Q, R, S, Vq_1, Vq_2, Vr_1, Vr_2, Vs_1, Vs_2,
        P, Aq, Ar, As
    );

    // --- Compute Gradients for Value Tensors (V) ---
    // These don't depend on grad_P
    compute_grad_Vq_1(grad_Vq_1, grad_output, Q, R, S, Vr_1, Vs_1, dropout_rate);
    compute_grad_Vr_1(grad_Vr_1, grad_output, Q, R, S, Vq_1, Vs_1, dropout_rate);
    compute_grad_Vs_1(grad_Vs_1, grad_output, Q, R, S, Vq_1, Vr_1, dropout_rate);

    compute_grad_Vq_2(grad_Vq_2, grad_output, Q, R, S, Vr_2, Vs_2, dropout_rate);
    compute_grad_Vr_2(grad_Vr_2, grad_output, Q, R, S, Vq_2, Vs_2, dropout_rate);
    compute_grad_Vs_2(grad_Vs_2, grad_output, Q, R, S, Vq_2, Vr_2, dropout_rate);

    // --- Compute Gradients for Query/Key/Sensor Tensors (Q, R, S) ---
    // Pass the precomputed grad_P to these functions
    compute_grad_Q(grad_Q, grad_P, R, S, scale);
    compute_grad_R(grad_R, grad_P, Q, S, scale);
    compute_grad_S(grad_S, grad_P, Q, R, scale);


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
          py::arg("grad_output"), // Corrected name
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