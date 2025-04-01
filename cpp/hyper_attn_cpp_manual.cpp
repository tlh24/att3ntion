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

// LATER: Remove redundant code for gather and scatter operations
// separate dot product code from gather and scatter functions 
void compute_Y_gather_q(
    torch::Tensor& Y_q,
    const torch::Tensor& Q,
    const torch::Tensor& R,
    const torch::Tensor& S,
    const torch::Tensor& Vr_1,
    const torch::Tensor& Vs_1
) {
    auto Q_acc = Q.accessor<float, 4>();  
    auto R_acc = R.accessor<float, 4>();  
    auto S_acc = S.accessor<float, 4>();  
    auto Vr_1_acc = Vr_1.accessor<float, 4>();         
    auto Vs_1_acc = Vs_1.accessor<float, 4>();  
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
                // Use compute_softmax_3d instead of direct computation
                // For compute_Y_gather_q, we fix dimension 0 (i) and compute softmax over j,k
                
                // Allocate array for softmax attention values
                std::vector<float> attn_values(J * K);
                
                // Compute softmax over j,k (dimensions 1 and 2) for fixed i (dimension 0)
                compute_softmax_3d(
                    Q_acc, R_acc, S_acc,
                    b, h,
                    0, i,  // fixed_dim = 0 (i), fixed_idx = i
                    J, K,  // dim1_size = J, dim2_size = K
                    1, 2, 0,  // dim1_idx_fn = 1 (j), dim2_idx_fn = 2 (k), fixed_dim_idx_fn = 0 (i)
                    scale, attn_values.data()
                );
                
                // Apply attention values to values and accumulate in Y_q
                int idx = 0;
                for (int j = 0; j < J; j++) {
                    for (int k = 0; k < K; k++) {
                        float attn = attn_values[idx++];
                        for (int d = 0; d < D; d++) {
                            Y_q_acc[b][h][i][d] += attn * Vr_1_acc[b][h][j][d] * Vs_1_acc[b][h][k][d];
                        }
                    }
                }
            }
        }
    }
}

void compute_Y_gather_r(
    torch::Tensor& Y_r,
    const torch::Tensor& Q,
    const torch::Tensor& R,
    const torch::Tensor& S,
    const torch::Tensor& Vq_1,
    const torch::Tensor& Vs_1
) {
    auto Q_acc = Q.accessor<float, 4>();  
    auto R_acc = R.accessor<float, 4>();  
    auto S_acc = S.accessor<float, 4>();  
    auto Vq_1_acc = Vq_1.accessor<float, 4>();  
    auto Vs_1_acc = Vs_1.accessor<float, 4>();  
    auto Y_r_acc = Y_r.accessor<float, 4>();

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
                // Use compute_softmax_3d instead of direct computation
                // For compute_Y_gather_r, we fix dimension 1 (j) and compute softmax over i,k
                
                // Allocate array for softmax attention values
                std::vector<float> attn_values(I * K);
                
                // Compute softmax over i,k (dimensions 0 and 2) for fixed j (dimension 1)
                compute_softmax_3d(
                    Q_acc, R_acc, S_acc,
                    b, h,
                    1, j,  // fixed_dim = 1 (j), fixed_idx = j
                    I, K,  // dim1_size = I, dim2_size = K
                    0, 2, 1,  // dim1_idx_fn = 0 (i), dim2_idx_fn = 2 (k), fixed_dim_idx_fn = 1 (j)
                    scale, attn_values.data()
                );
                
                // Apply attention values to values and accumulate in Y_r
                int idx = 0;
                for (int i = 0; i < I; i++) {
                    for (int k = 0; k < K; k++) {
                        float attn = attn_values[idx++];
                        for (int d = 0; d < D; d++) {
                            Y_r_acc[b][h][j][d] += attn * Vq_1_acc[b][h][i][d] * Vs_1_acc[b][h][k][d];
                        }
                    }
                }
            }
        }
    }
}

void compute_Y_gather_s(
    torch::Tensor& Y_s,
    const torch::Tensor& Q,
    const torch::Tensor& R,
    const torch::Tensor& S,
    const torch::Tensor& Vq_1,
    const torch::Tensor& Vr_1
) {
    auto Q_acc = Q.accessor<float, 4>();  
    auto R_acc = R.accessor<float, 4>();  
    auto S_acc = S.accessor<float, 4>();  
    auto Vq_1_acc = Vq_1.accessor<float, 4>();  
    auto Vr_1_acc = Vr_1.accessor<float, 4>();  
    auto Y_s_acc = Y_s.accessor<float, 4>();

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
                // Use compute_softmax_3d instead of direct computation
                // For compute_Y_gather_s, we fix dimension 2 (k) and compute softmax over i,j
                
                // Allocate array for softmax attention values
                std::vector<float> attn_values(I * J);
                
                // Compute softmax over i,j (dimensions 0 and 1) for fixed k (dimension 2)
                compute_softmax_3d(
                    Q_acc, R_acc, S_acc,
                    b, h,
                    2, k,  // fixed_dim = 2 (k), fixed_idx = k
                    I, J,  // dim1_size = I, dim2_size = J
                    0, 1, 2,  // dim1_idx_fn = 0 (i), dim2_idx_fn = 1 (j), fixed_dim_idx_fn = 2 (k)
                    scale, attn_values.data()
                );
                
                // Apply attention values to values and accumulate in Y_s
                int idx = 0;
                for (int i = 0; i < I; i++) {
                    for (int j = 0; j < J; j++) {
                        float attn = attn_values[idx++];
                        for (int d = 0; d < D; d++) {
                            Y_s_acc[b][h][k][d] += attn * Vq_1_acc[b][h][i][d] * Vr_1_acc[b][h][j][d];
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

    compute_Y_gather_q(Y_q, Q, R, S, Vr_1, Vs_1);
    compute_Y_gather_r(Y_r, Q, R, S, Vq_1, Vs_1);
    compute_Y_gather_s(Y_s, Q, R, S, Vq_1, Vr_1);

    compute_Y_scatter_q(Y_q_, Q, R, S, Vr_2, Vs_2);
    compute_Y_scatter_r(Y_r_, Q, R, S, Vq_2, Vs_2);
    compute_Y_scatter_s(Y_s_, Q, R, S, Vq_2, Vr_2);

    return std::make_tuple(Y_q, Y_r, Y_s, Y_q_, Y_r_, Y_s_);
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
}

