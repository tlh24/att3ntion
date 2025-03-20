#include <torch/extension.h> 
#include <iostream> 
#include <cmath> 
#include <unordered_map>
#include <mutex>
#include <chrono>

// Helper function to print memory usage of tensor
void print_tensor_memory(const std::string& name, const torch::Tensor& tensor) {
    size_t numel = tensor.numel();
    size_t element_size = 0;
    
    if (tensor.dtype() == torch::kFloat32) {
        element_size = 4;
    } else if (tensor.dtype() == torch::kFloat64) {
        element_size = 8;
    } else if (tensor.dtype() == torch::kInt64) {
        element_size = 8;
    } else {
        element_size = 4;  // Default assumption
    }
    
    size_t memory_bytes = numel * element_size;
    double memory_mb = static_cast<double>(memory_bytes) / (1024 * 1024);
    
}

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
        
        
        // For timing the operations
        auto start_time = std::chrono::high_resolution_clock::now();
        auto end_time = std::chrono::high_resolution_clock::now();
        
        // Gather operations
        start_time = std::chrono::high_resolution_clock::now();
        auto Y_q = torch::einsum("bhijk,bhjd,bhkd->bhid", {Aq, Vr_1, Vs_1});
        end_time = std::chrono::high_resolution_clock::now();
        auto duration_Y_q = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time).count();
        
        start_time = std::chrono::high_resolution_clock::now();
        auto Y_r = torch::einsum("bhijk,bhid,bhkd->bhjd", {Ar, Vq_1, Vs_1});
        end_time = std::chrono::high_resolution_clock::now();
        auto duration_Y_r = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time).count();
        
        start_time = std::chrono::high_resolution_clock::now();
        auto Y_s = torch::einsum("bhijk,bhid,bhjd->bhkd", {As, Vq_1, Vr_1});
        end_time = std::chrono::high_resolution_clock::now();
        auto duration_Y_s = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time).count();
        
        // Optimize scatter operations (Y_q_, Y_r_, Y_s_) while maintaining equivalence
        
        // For Y_x_: First compute element-wise multiplication of Ar and As
        start_time = std::chrono::high_resolution_clock::now();
        
        auto ArAs = Ar * As;
        // Then perform the contraction with value tensors
        auto Y_q_ = torch::einsum("bhijk,bhjd,bhkd->bhid", {ArAs, Vr_2, Vs_2});
        
        end_time = std::chrono::high_resolution_clock::now();
        auto duration_Y_q_ = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time).count();
        
        start_time = std::chrono::high_resolution_clock::now();
        
        auto AqAs = Aq * As;
        auto Y_r_ = torch::einsum("bhijk,bhid,bhkd->bhjd", {AqAs, Vq_2, Vs_2});
        
        end_time = std::chrono::high_resolution_clock::now();
        auto duration_Y_r_ = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time).count();
        
        start_time = std::chrono::high_resolution_clock::now();
        
        auto AqAr = Aq * Ar;
        auto Y_s_ = torch::einsum("bhijk,bhid,bhjd->bhkd", {AqAr, Vq_2, Vr_2});
        
        end_time = std::chrono::high_resolution_clock::now();
        auto duration_Y_s_ = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time).count();
        
        // Total einsum time
        auto total_einsum_time = duration_Y_q + duration_Y_r + duration_Y_s + duration_Y_q_ + duration_Y_r_ + duration_Y_s_;
        std::cout << "Total einsum time: " << total_einsum_time << " microseconds" << std::endl;
        
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
    
    // Use static variables to create the model only once
    static HypergraphAttention model(d_model, n_heads, dropout_rate);
    static bool initialized = false;
    static int64_t cached_d_model = d_model;
    static int64_t cached_n_heads = n_heads;
    static double cached_dropout_rate = dropout_rate;
    
    // If parameters have changed, we need to create a new model
    if (!initialized || 
        cached_d_model != d_model || 
        cached_n_heads != n_heads || 
        cached_dropout_rate != dropout_rate) {
        
        model = HypergraphAttention(d_model, n_heads, dropout_rate);
        model->eval();
        
        // Update cached values
        cached_d_model = d_model;
        cached_n_heads = n_heads;
        cached_dropout_rate = dropout_rate;
        initialized = true;
    }
    
    // Move model to the correct device if needed
    auto params = model->parameters();
    if (!params.empty() && params[0].device() != input.device()) {
        model->to(input.device());
    }

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