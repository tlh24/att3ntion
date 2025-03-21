#include <torch/extension.h> 
#include <iostream> 
#include <cmath> 
#include <limits>

// Helper to compute output tensor via 3-way softmax attention
void compute_Y(
    torch::Tensor& Y,
    const torch::Tensor& Q,
    const torch::Tensor& R,
    const torch::Tensor& S,
    const torch::Tensor& V1,
    const torch::Tensor& V2,
    const std::string& gather_dim  // "i", "j", or "k"
) {
    auto Q_acc = Q.accessor<float, 4>();  // [B, H, I, D]
    auto R_acc = R.accessor<float, 4>();  // [B, H, J, D]
    auto S_acc = S.accessor<float, 4>();  // [B, H, K, D]
    auto V1_acc = V1.accessor<float, 4>();  // [B, H, ?, D]
    auto V2_acc = V2.accessor<float, 4>();  // [B, H, ?, D]
    auto Y_acc = Y.accessor<float, 4>();

    int B = Q.size(0);
    int H = Q.size(1);
    int I = Q.size(2);
    int J = R.size(2);
    int K = S.size(2);
    int D = Q.size(3);
    float scale = 1.0f / std::sqrt(static_cast<float>(D));

    for (int b = 0; b < B; b++) {
        for (int h = 0; h < H; h++) {
            if (gather_dim == "i") {
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
                                Y_acc[b][h][i][d] += attn * V1_acc[b][h][j][d] * V2_acc[b][h][k][d];
                        }
                    }
                }
            } else if (gather_dim == "j") {
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
                                Y_acc[b][h][j][d] += attn * V1_acc[b][h][i][d] * V2_acc[b][h][k][d];
                        }
                    }
                }
            } else if (gather_dim == "k") {
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
                                Y_acc[b][h][k][d] += attn * V1_acc[b][h][i][d] * V2_acc[b][h][j][d];
                        }
                    }
                }
            }
        }
    }
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> hyper_attn_forward(
    torch::Tensor Q,       // [B, H, I, D]
    torch::Tensor R,       // [B, H, J, D]
    torch::Tensor S,       // [B, H, K, D]
    torch::Tensor Vq_1,    // [B, H, I, D]
    torch::Tensor Vq_2,    // unused
    torch::Tensor Vr_1,    // [B, H, J, D]
    torch::Tensor Vr_2,    // unused
    torch::Tensor Vs_1,    // [B, H, K, D]
    torch::Tensor Vs_2,    // unused
    double dropout_rate = 0.0) 
{
    auto options = Q.options();
    int B = Q.size(0), H = Q.size(1), I = Q.size(2), D = Q.size(3);
    int J = R.size(2), K = S.size(2);

    auto Y_q = torch::zeros({B, H, I, D}, options);
    auto Y_r = torch::zeros({B, H, J, D}, options);
    auto Y_s = torch::zeros({B, H, K, D}, options);

    compute_Y(Y_q, Q, R, S, Vr_1, Vs_1, "i");
    compute_Y(Y_r, Q, R, S, Vq_1, Vs_1, "j");
    compute_Y(Y_s, Q, R, S, Vq_1, Vr_1, "k");

    return std::make_tuple(Y_q, Y_r, Y_s); //Y_q + Y_r + Y_s;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &hyper_attn_forward,
          "Hypergraph Attention forward (returns Y_q, Y_r, Y_s)",
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

