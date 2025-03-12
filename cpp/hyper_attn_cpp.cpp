#include <torch/extension.h> 
#include <iostream> 
#include <cmath> 

struct QuickGELUImpl : torch::nn::Module { //activation function
    torch::Tensor forward(torch::Tensor x) {
        return x * torch::sigmoid(1.702 * x);
    }
};
TORCH_MODULE(QuickGELU);


struct HypergraphAttentionImpl : torch::nn::Module {
    HypergraphAttentionImpl(int64_t d_model, int64_t n_heads, double dropout_rate = 0.0)
        : d_model(d_model), n_heads(n_heads), head_dim(d_model) {

            torch::manual_seed(42);
        
        Wq = register_module("Wq", torch::nn::Linear(torch::nn::LinearOptions(d_model, d_model * n_heads).bias(false)));
        Wr = register_module("Wr", torch::nn::Linear(torch::nn::LinearOptions(d_model, d_model * n_heads).bias(false)));
        Ws = register_module("Ws", torch::nn::Linear(torch::nn::LinearOptions(d_model, d_model * n_heads).bias(false)));
        
        Wv_q = register_module("Wv_q", torch::nn::Linear(torch::nn::LinearOptions(d_model, d_model * n_heads * 2).bias(true)));
        Wv_r = register_module("Wv_r", torch::nn::Linear(torch::nn::LinearOptions(d_model, d_model * n_heads * 2).bias(true)));
        Wv_s = register_module("Wv_s", torch::nn::Linear(torch::nn::LinearOptions(d_model, d_model * n_heads * 2).bias(true)));
        
        Wo = register_module("Wo", torch::nn::Linear(torch::nn::LinearOptions(d_model, d_model).bias(true)));
        
        dropout = register_module("dropout", torch::nn::Dropout(dropout_rate));
        gelu = register_module("gelu", QuickGELU());
    }
    
    torch::Tensor forward(torch::Tensor x) {
        auto batch_size = x.size(0);
        auto ntok = x.size(1);
        
        auto Q = Wq->forward(x);
        auto R = Wr->forward(x);
        auto S = Ws->forward(x);
        
        auto Vq = Wv_q->forward(x);
        auto Vr = Wv_r->forward(x);
        auto Vs = Wv_s->forward(x);
        
        Q = Q.reshape({batch_size, ntok, n_heads, head_dim}).permute({0, 2, 1, 3});
        R = R.reshape({batch_size, ntok, n_heads, head_dim}).permute({0, 2, 1, 3});
        S = S.reshape({batch_size, ntok, n_heads, head_dim}).permute({0, 2, 1, 3});
        
        auto Vq_split = Vq.reshape({batch_size, ntok, n_heads, head_dim * 2}).permute({0, 2, 1, 3}).split(head_dim, -1);
        auto Vr_split = Vr.reshape({batch_size, ntok, n_heads, head_dim * 2}).permute({0, 2, 1, 3}).split(head_dim, -1);
        auto Vs_split = Vs.reshape({batch_size, ntok, n_heads, head_dim * 2}).permute({0, 2, 1, 3}).split(head_dim, -1);
        
        auto Vq_1 = Vq_split[0];
        auto Vq_2 = Vq_split[1];
        auto Vr_1 = Vr_split[0];
        auto Vr_2 = Vr_split[1];
        auto Vs_1 = Vs_split[0];
        auto Vs_2 = Vs_split[1];
        
        // Compute 3-way attention scores
        auto dot_product = torch::einsum("bhid,bhjd,bhkd->bhijk", {Q, R, S});
        dot_product = dot_product / std::sqrt(static_cast<double>(head_dim));
        
        // Aq - gathering to position i (softmax over j,k)
        auto dot_product_q = dot_product;
        auto Aq = torch::softmax(dot_product_q.flatten(3, 4), -1).reshape_as(dot_product);
        
        // Ar - gathering to position j (softmax over i,k)
        auto dot_product_r = dot_product.permute({0, 1, 3, 2, 4});
        auto Ar = torch::softmax(dot_product_r.flatten(3, 4), -1).reshape_as(dot_product_r);
        Ar = Ar.permute({0, 1, 3, 2, 4});
        
        // As - gathering to position k (softmax over i,j)
        auto dot_product_s = dot_product.permute({0, 1, 4, 2, 3});
        auto As = torch::softmax(dot_product_s.flatten(3, 4), -1).reshape_as(dot_product_s);
        As = As.permute({0, 1, 3, 4, 2});
        
        Aq = dropout->forward(Aq);
        Ar = dropout->forward(Ar);
        As = dropout->forward(As);
        
        // Gather operations
        auto Y_q = torch::einsum("bhijk,bhjd,bhkd->bhid", {Aq, Vr_1, Vs_1});
        auto Y_r = torch::einsum("bhijk,bhid,bhkd->bhjd", {Ar, Vq_1, Vs_1});
        auto Y_s = torch::einsum("bhijk,bhid,bhjd->bhkd", {As, Vq_1, Vr_1});
        
        // Scatter operations
        auto Y_q_ = torch::einsum("bhijk,bhjd,bhijk,bhkd->bhid", {Ar, Vr_2, As, Vs_2});
        auto Y_r_ = torch::einsum("bhijk,bhid,bhijk,bhkd->bhjd", {Aq, Vq_2, As, Vs_2});
        auto Y_s_ = torch::einsum("bhijk,bhid,bhijk,bhjd->bhkd", {Aq, Vq_2, Ar, Vr_2});
        
        auto y = Y_q + Y_r + Y_s + Y_q_ + Y_r_ + Y_s_;
        
        // Sum along the heads
        y = y.permute({0, 2, 1, 3}).sum(2).squeeze();
        y = gelu->forward(y);
        y = Wo->forward(y);
        
        return y;
    }
    
    torch::nn::Linear Wq{nullptr}, Wr{nullptr}, Ws{nullptr};
    torch::nn::Linear Wv_q{nullptr}, Wv_r{nullptr}, Wv_s{nullptr};
    torch::nn::Linear Wo{nullptr};
    torch::nn::Dropout dropout{nullptr};
    QuickGELU gelu{nullptr};
    
    int64_t d_model;
    int64_t n_heads;
    int64_t head_dim;
};
TORCH_MODULE(HypergraphAttention);

// Python bindings
torch::Tensor hyper_attn_forward(
    torch::Tensor input,
    int64_t d_model,
    int64_t n_heads,
    double dropout_rate = 0.0) {
    
    HypergraphAttention model(d_model, n_heads, dropout_rate);
    model->eval();

    torch::NoGradGuard no_grad;
    auto output = model->forward(input);
    
    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &hyper_attn_forward, "HypergraphAttention forward",
          py::arg("input"),
          py::arg("d_model"),
          py::arg("n_heads"),
          py::arg("dropout_rate") = 0.0);
} 