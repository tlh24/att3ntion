#include <torch/extension.h> 
#include <iostream> 
#include <cmath> 
#include <limits>

// Helper to compute output tensor via 3-way softmax attention
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
                // softmax over j,k
                float max_val = -std::numeric_limits<float>::infinity();
                for (int j = 0; j < J; j++) {
                    for (int k = 0; k < K; k++) {
                        float dot = 0.0f;
                        for (int d = 0; d < D; d++)
                            dot += Q_acc[b][h][i][d] * R_acc[b][h][j][d] * S_acc[b][h][k][d];
                        dot *= scale;
                        if (dot > max_val) max_val = dot;
                    }
                }

                float sum_exp = 0.0f;
                for (int j = 0; j < J; j++) {
                    for (int k = 0; k < K; k++) {
                        float dot = 0.0f;
                        for (int d = 0; d < D; d++)
                            dot += Q_acc[b][h][i][d] * R_acc[b][h][j][d] * S_acc[b][h][k][d];
                        dot *= scale;
                        sum_exp += std::exp(dot - max_val);
                    }
                }

                for (int j = 0; j < J; j++) {
                    for (int k = 0; k < K; k++) {
                        float dot = 0.0f;
                        for (int d = 0; d < D; d++)
                            dot += Q_acc[b][h][i][d] * R_acc[b][h][j][d] * S_acc[b][h][k][d];
                        dot *= scale;
                        float attn = std::exp(dot - max_val) / sum_exp;

                        for (int d = 0; d < D; d++)
                            Y_q_acc[b][h][i][d] += attn * Vr_1_acc[b][h][j][d] * Vs_1_acc[b][h][k][d];
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
                // softmax over i,k
                float max_val = -std::numeric_limits<float>::infinity();
                for (int i = 0; i < I; i++) {
                    for (int k = 0; k < K; k++) {
                        float dot = 0.0f;
                        for (int d = 0; d < D; d++)
                            dot += Q_acc[b][h][i][d] * R_acc[b][h][j][d] * S_acc[b][h][k][d];
                        dot *= scale;
                        if (dot > max_val) max_val = dot;
                    }
                }

                float sum_exp = 0.0f;
                for (int i = 0; i < I; i++) {
                    for (int k = 0; k < K; k++) {
                        float dot = 0.0f;
                        for (int d = 0; d < D; d++)
                            dot += Q_acc[b][h][i][d] * R_acc[b][h][j][d] * S_acc[b][h][k][d];
                        dot *= scale;
                        sum_exp += std::exp(dot - max_val);
                    }
                }

                for (int i = 0; i < I; i++) {
                    for (int k = 0; k < K; k++) {
                        float dot = 0.0f;
                        for (int d = 0; d < D; d++)
                            dot += Q_acc[b][h][i][d] * R_acc[b][h][j][d] * S_acc[b][h][k][d];
                        dot *= scale;
                        float attn = std::exp(dot - max_val) / sum_exp;

                        for (int d = 0; d < D; d++)
                            Y_r_acc[b][h][j][d] += attn * Vq_1_acc[b][h][i][d] * Vs_1_acc[b][h][k][d];
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
                // softmax over i,j
                float max_val = -std::numeric_limits<float>::infinity();
                for (int i = 0; i < I; i++) {
                    for (int j = 0; j < J; j++) {
                        float dot = 0.0f;
                        for (int d = 0; d < D; d++)
                            dot += Q_acc[b][h][i][d] * R_acc[b][h][j][d] * S_acc[b][h][k][d];
                        dot *= scale;
                        if (dot > max_val) max_val = dot;
                    }
                }

                float sum_exp = 0.0f;
                for (int i = 0; i < I; i++) {
                    for (int j = 0; j < J; j++) {
                        float dot = 0.0f;
                        for (int d = 0; d < D; d++)
                            dot += Q_acc[b][h][i][d] * R_acc[b][h][j][d] * S_acc[b][h][k][d];
                        dot *= scale;
                        sum_exp += std::exp(dot - max_val);
                    }
                }

                for (int i = 0; i < I; i++) {
                    for (int j = 0; j < J; j++) {
                        float dot = 0.0f;
                        for (int d = 0; d < D; d++)
                            dot += Q_acc[b][h][i][d] * R_acc[b][h][j][d] * S_acc[b][h][k][d];
                        dot *= scale;
                        float attn = std::exp(dot - max_val) / sum_exp;

                        for (int d = 0; d < D; d++)
                            Y_s_acc[b][h][k][d] += attn * Vq_1_acc[b][h][i][d] * Vr_1_acc[b][h][j][d];
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
                    // Compute max for numerical stability for Ar (softmax over i,k)
                    float max_val_j = -std::numeric_limits<float>::infinity();
                    for (int i_idx = 0; i_idx < I; i_idx++) {
                        for (int k = 0; k < K; k++) {
                            float dot = 0.0f;
                            for (int d = 0; d < D; d++)
                                dot += Q_acc[b][h][i_idx][d] * R_acc[b][h][j][d] * S_acc[b][h][k][d];
                            dot *= scale;
                            if (dot > max_val_j) max_val_j = dot;
                        }
                    }

                    float sum_exp_j = 0.0f;
                    for (int i_idx = 0; i_idx < I; i_idx++) {
                        for (int k = 0; k < K; k++) {
                            float dot = 0.0f;
                            for (int d = 0; d < D; d++)
                                dot += Q_acc[b][h][i_idx][d] * R_acc[b][h][j][d] * S_acc[b][h][k][d];
                            dot *= scale;
                            sum_exp_j += std::exp(dot - max_val_j);
                        }
                    }

                    // For each k, compute the partial Ar value (for this specific j)
                    for (int k = 0; k < K; k++) {
                        float dot = 0.0f;
                        for (int d = 0; d < D; d++)
                            dot += Q_acc[b][h][i][d] * R_acc[b][h][j][d] * S_acc[b][h][k][d];
                        dot *= scale;
                        Ar_values[j][k] = std::exp(dot - max_val_j) / sum_exp_j;
                    }
                }

                // Now compute As values
                std::vector<std::vector<float>> As_values(K, std::vector<float>(J, 0.0f));
                for (int k = 0; k < K; k++) {
                    // Compute max for numerical stability for As (softmax over i,j)
                    float max_val_k = -std::numeric_limits<float>::infinity();
                    for (int i_idx = 0; i_idx < I; i_idx++) {
                        for (int j = 0; j < J; j++) {
                            float dot = 0.0f;
                            for (int d = 0; d < D; d++)
                                dot += Q_acc[b][h][i_idx][d] * R_acc[b][h][j][d] * S_acc[b][h][k][d];
                            dot *= scale;
                            if (dot > max_val_k) max_val_k = dot;
                        }
                    }

                    float sum_exp_k = 0.0f;
                    for (int i_idx = 0; i_idx < I; i_idx++) {
                        for (int j = 0; j < J; j++) {
                            float dot = 0.0f;
                            for (int d = 0; d < D; d++)
                                dot += Q_acc[b][h][i_idx][d] * R_acc[b][h][j][d] * S_acc[b][h][k][d];
                            dot *= scale;
                            sum_exp_k += std::exp(dot - max_val_k);
                        }
                    }

                    // For each j, compute the partial As value (for this specific k)
                    for (int j = 0; j < J; j++) {
                        float dot = 0.0f;
                        for (int d = 0; d < D; d++)
                            dot += Q_acc[b][h][i][d] * R_acc[b][h][j][d] * S_acc[b][h][k][d];
                        dot *= scale;
                        As_values[k][j] = std::exp(dot - max_val_k) / sum_exp_k;
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
                    // Compute max for numerical stability for Aq (softmax over j,k)
                    float max_val_i = -std::numeric_limits<float>::infinity();
                    for (int j_idx = 0; j_idx < J; j_idx++) {
                        for (int k = 0; k < K; k++) {
                            float dot = 0.0f;
                            for (int d = 0; d < D; d++)
                                dot += Q_acc[b][h][i][d] * R_acc[b][h][j_idx][d] * S_acc[b][h][k][d];
                            dot *= scale;
                            if (dot > max_val_i) max_val_i = dot;
                        }
                    }

                    float sum_exp_i = 0.0f;
                    for (int j_idx = 0; j_idx < J; j_idx++) {
                        for (int k = 0; k < K; k++) {
                            float dot = 0.0f;
                            for (int d = 0; d < D; d++)
                                dot += Q_acc[b][h][i][d] * R_acc[b][h][j_idx][d] * S_acc[b][h][k][d];
                            dot *= scale;
                            sum_exp_i += std::exp(dot - max_val_i);
                        }
                    }

                    // For each k, compute the partial Aq value (for this specific i)
                    for (int k = 0; k < K; k++) {
                        float dot = 0.0f;
                        for (int d = 0; d < D; d++)
                            dot += Q_acc[b][h][i][d] * R_acc[b][h][j][d] * S_acc[b][h][k][d];
                        dot *= scale;
                        Aq_values[i][k] = std::exp(dot - max_val_i) / sum_exp_i;
                    }
                }

                // Now compute As values
                std::vector<std::vector<float>> As_values(K, std::vector<float>(I, 0.0f));
                for (int k = 0; k < K; k++) {
                    // Compute max for numerical stability for As (softmax over i,j)
                    float max_val_k = -std::numeric_limits<float>::infinity();
                    for (int i = 0; i < I; i++) {
                        for (int j_idx = 0; j_idx < J; j_idx++) {
                            float dot = 0.0f;
                            for (int d = 0; d < D; d++)
                                dot += Q_acc[b][h][i][d] * R_acc[b][h][j_idx][d] * S_acc[b][h][k][d];
                            dot *= scale;
                            if (dot > max_val_k) max_val_k = dot;
                        }
                    }

                    float sum_exp_k = 0.0f;
                    for (int i = 0; i < I; i++) {
                        for (int j_idx = 0; j_idx < J; j_idx++) {
                            float dot = 0.0f;
                            for (int d = 0; d < D; d++)
                                dot += Q_acc[b][h][i][d] * R_acc[b][h][j_idx][d] * S_acc[b][h][k][d];
                            dot *= scale;
                            sum_exp_k += std::exp(dot - max_val_k);
                        }
                    }

                    // For each i, compute the partial As value (for this specific k)
                    for (int i = 0; i < I; i++) {
                        float dot = 0.0f;
                        for (int d = 0; d < D; d++)
                            dot += Q_acc[b][h][i][d] * R_acc[b][h][j][d] * S_acc[b][h][k][d];
                        dot *= scale;
                        As_values[k][i] = std::exp(dot - max_val_k) / sum_exp_k;
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
                    // Compute max for numerical stability for Aq (softmax over j,k)
                    float max_val_i = -std::numeric_limits<float>::infinity();
                    for (int j = 0; j < J; j++) {
                        for (int k_idx = 0; k_idx < K; k_idx++) {
                            float dot = 0.0f;
                            for (int d = 0; d < D; d++)
                                dot += Q_acc[b][h][i][d] * R_acc[b][h][j][d] * S_acc[b][h][k_idx][d];
                            dot *= scale;
                            if (dot > max_val_i) max_val_i = dot;
                        }
                    }

                    float sum_exp_i = 0.0f;
                    for (int j = 0; j < J; j++) {
                        for (int k_idx = 0; k_idx < K; k_idx++) {
                            float dot = 0.0f;
                            for (int d = 0; d < D; d++)
                                dot += Q_acc[b][h][i][d] * R_acc[b][h][j][d] * S_acc[b][h][k_idx][d];
                            dot *= scale;
                            sum_exp_i += std::exp(dot - max_val_i);
                        }
                    }

                    // For each j, compute the partial Aq value (for this specific i)
                    for (int j = 0; j < J; j++) {
                        float dot = 0.0f;
                        for (int d = 0; d < D; d++)
                            dot += Q_acc[b][h][i][d] * R_acc[b][h][j][d] * S_acc[b][h][k][d];
                        dot *= scale;
                        Aq_values[i][j] = std::exp(dot - max_val_i) / sum_exp_i;
                    }
                }

                // Now compute Ar values
                std::vector<std::vector<float>> Ar_values(J, std::vector<float>(I, 0.0f));
                for (int j = 0; j < J; j++) {
                    // Compute max for numerical stability for Ar (softmax over i,k)
                    float max_val_j = -std::numeric_limits<float>::infinity();
                    for (int i = 0; i < I; i++) {
                        for (int k_idx = 0; k_idx < K; k_idx++) {
                            float dot = 0.0f;
                            for (int d = 0; d < D; d++)
                                dot += Q_acc[b][h][i][d] * R_acc[b][h][j][d] * S_acc[b][h][k_idx][d];
                            dot *= scale;
                            if (dot > max_val_j) max_val_j = dot;
                        }
                    }

                    float sum_exp_j = 0.0f;
                    for (int i = 0; i < I; i++) {
                        for (int k_idx = 0; k_idx < K; k_idx++) {
                            float dot = 0.0f;
                            for (int d = 0; d < D; d++)
                                dot += Q_acc[b][h][i][d] * R_acc[b][h][j][d] * S_acc[b][h][k_idx][d];
                            dot *= scale;
                            sum_exp_j += std::exp(dot - max_val_j);
                        }
                    }

                    // For each i, compute the partial Ar value (for this specific j)
                    for (int i = 0; i < I; i++) {
                        float dot = 0.0f;
                        for (int d = 0; d < D; d++)
                            dot += Q_acc[b][h][i][d] * R_acc[b][h][j][d] * S_acc[b][h][k][d];
                        dot *= scale;
                        Ar_values[j][i] = std::exp(dot - max_val_j) / sum_exp_j;
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

