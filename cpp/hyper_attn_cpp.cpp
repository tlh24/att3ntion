#include <torch/extension.h> 
#include <iostream> 
#include <cmath> 

torch::Tensor hyper_attn_forward(
    torch::Tensor Q,       // [batch_size, n_heads, seq_len, head_dim]
    torch::Tensor R,       // [batch_size, n_heads, seq_len, head_dim]
    torch::Tensor S,       // [batch_size, n_heads, seq_len, head_dim]
    torch::Tensor Vq_1,    // [batch_size, n_heads, seq_len, head_dim]
    torch::Tensor Vq_2,    // [batch_size, n_heads, seq_len, head_dim]
    torch::Tensor Vr_1,    // [batch_size, n_heads, seq_len, head_dim]
    torch::Tensor Vr_2,    // [batch_size, n_heads, seq_len, head_dim]
    torch::Tensor Vs_1,    // [batch_size, n_heads, seq_len, head_dim]
    torch::Tensor Vs_2,    // [batch_size, n_heads, seq_len, head_dim]
    double dropout_rate = 0.0) {
    
    auto head_dim = Q.size(3);
    
    // Compute 3-way attention scores - this is the core computation
    auto dot_product = torch::einsum("bhid,bhjd,bhkd->bhijk", {Q, R, S});
    dot_product = dot_product / std::sqrt(static_cast<double>(head_dim));
    
    // Aq - gathering to position i (softmax over j,k)
    auto Aq = torch::softmax(dot_product.flatten(3, 4), -1).reshape_as(dot_product);
    
    // Ar - gathering to position j (softmax over i,k)
    auto dot_product_r = dot_product.permute({0, 1, 3, 2, 4});
    auto Ar = torch::softmax(dot_product_r.flatten(3, 4), -1).reshape_as(dot_product_r);
    Ar = Ar.permute({0, 1, 3, 2, 4});
    
    // As - gathering to position k (softmax over i,j)
    auto dot_product_s = dot_product.permute({0, 1, 4, 2, 3});
    auto As = torch::softmax(dot_product_s.flatten(3, 4), -1).reshape_as(dot_product_s);
    As = As.permute({0, 1, 3, 4, 2});
    
    // Apply dropout if needed
    if (dropout_rate > 0.0) {
        auto dropout = torch::nn::Dropout(torch::nn::DropoutOptions(dropout_rate));
        Aq = dropout->forward(Aq);
        Ar = dropout->forward(Ar);
        As = dropout->forward(As);
    }
    
    // Gather operations
    auto Y_q = torch::einsum("bhijk,bhjd,bhkd->bhid", {Aq, Vr_1, Vs_1});
    auto Y_r = torch::einsum("bhijk,bhid,bhkd->bhjd", {Ar, Vq_1, Vs_1});
    auto Y_s = torch::einsum("bhijk,bhid,bhjd->bhkd", {As, Vq_1, Vr_1});
    
    // Optimize scatter operations with element-wise multiplication
    auto ArAs = Ar * As;
    auto Y_q_ = torch::einsum("bhijk,bhjd,bhkd->bhid", {ArAs, Vr_2, Vs_2});
    
    auto AqAs = Aq * As;
    auto Y_r_ = torch::einsum("bhijk,bhid,bhkd->bhjd", {AqAs, Vq_2, Vs_2});
    
    auto AqAr = Aq * Ar;
    auto Y_s_ = torch::einsum("bhijk,bhid,bhjd->bhkd", {AqAr, Vq_2, Vr_2});
    
    // Combine results
    auto y = Y_q + Y_r + Y_s + Y_q_ + Y_r_ + Y_s_;
    
    return y;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &hyper_attn_forward, "HypergraphAttention forward",
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