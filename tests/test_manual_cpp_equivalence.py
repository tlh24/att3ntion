import torch
import numpy as np
import sys
import os

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from tests.simple_analogy_model import SimpleAnalogyModel, prepare_data
from tests.gen_data import genData

BATCH_SIZE = 2    
HIDDEN_DIM = 4    
NUM_HEADS = 1     
MODULO = 5        
DEVICE = 'cpu'    
SEED = 42

print("--- Setting up test environment ---")
torch.manual_seed(SEED)
np.random.seed(SEED)
torch.set_printoptions(precision=6, sci_mode=False)

print(f"--- Generating data (batch_size={BATCH_SIZE}, modulo={MODULO}) ---")
data_np = genData(BATCH_SIZE, MODULO)
inputs_np = data_np.copy()
inputs_np[:, 1, :4] = 0
inputs_np[:, 5, :4] = 0
inputs_np[:, 7] = 0
inputs_tensor = torch.FloatTensor(inputs_np).to(DEVICE)

# Target extraction (needed for loss calculation)
op_targets = np.zeros((data_np.shape[0], 2), dtype=np.int64)
op_targets[:, 0] = np.argmax(data_np[:, 1, :4], axis=1)
op_targets[:, 1] = np.argmax(data_np[:, 5, :4], axis=1)
value_targets = np.argmax(data_np[:, 7, 4:], axis=1)
op_targets_tensor = torch.LongTensor(op_targets).to(DEVICE)
value_targets_tensor = torch.LongTensor(value_targets).to(DEVICE)


print(f"--- Initializing models (hidden_dim={HIDDEN_DIM}, num_heads={NUM_HEADS}) ---")
model_py = SimpleAnalogyModel(hidden_dim=HIDDEN_DIM, num_heads=NUM_HEADS, attn_impl='pytorch').to(DEVICE)
model_cpp = SimpleAnalogyModel(hidden_dim=HIDDEN_DIM, num_heads=NUM_HEADS, attn_impl='cpp').to(DEVICE)

print("--- Copying weights from PyTorch model to C++ model ---")
model_cpp.load_state_dict(model_py.state_dict())

model_py.train()
model_cpp.train()

print("--- Running forward/backward pass for PyTorch model ---")
model_py.zero_grad()
op_pred1_py, op_pred5_py, value_pred_py = model_py(inputs_tensor)
loss_py = (op_pred1_py.sum() + op_pred5_py.sum() + value_pred_py.sum()) * 0.1 # Scale loss to avoid large gradients initially
print(f"PyTorch Loss: {loss_py.item():.6f}")
loss_py.backward()
print("PyTorch backward pass complete.")

print("--- Running forward/backward pass for C++ model ---")
model_cpp.zero_grad()
op_pred1_cpp, op_pred5_cpp, value_pred_cpp = model_cpp(inputs_tensor)
loss_cpp = (op_pred1_cpp.sum() + op_pred5_cpp.sum() + value_pred_cpp.sum()) * 0.1
print(f"C++ Loss: {loss_cpp.item():.6f}")
loss_cpp.backward()
print("C++ backward pass complete.")

print("--- Comparing Outputs ---")
outputs_match = True
atol = 1e-5
rtol = 1e-4

if torch.allclose(op_pred1_py, op_pred1_cpp, atol=atol, rtol=rtol):
    print("✅ op_pred1 outputs match.")
else:
    print("❌ op_pred1 outputs DO NOT match.")
    print("PY:", op_pred1_py)
    print("CPP:", op_pred1_cpp)
    print("Diff:", torch.abs(op_pred1_py - op_pred1_cpp))
    outputs_match = False

if torch.allclose(op_pred5_py, op_pred5_cpp, atol=atol, rtol=rtol):
    print("✅ op_pred5 outputs match.")
else:
    print("❌ op_pred5 outputs DO NOT match.")
    print("PY:", op_pred5_py)
    print("CPP:", op_pred5_cpp)
    print("Diff:", torch.abs(op_pred5_py - op_pred5_cpp))
    outputs_match = False

if torch.allclose(value_pred_py, value_pred_cpp, atol=atol, rtol=rtol):
    print("✅ value_pred outputs match.")
else:
    print("❌ value_pred outputs DO NOT match.")
    print("PY:", value_pred_py)
    print("CPP:", value_pred_cpp)
    print("Diff:", torch.abs(value_pred_py - value_pred_cpp))
    outputs_match = False

print("--- Comparing Gradients ---")
gradients_match = True
mismatched_params = []

params_py = dict(model_py.named_parameters())
params_cpp = dict(model_cpp.named_parameters())

for name, param_py in params_py.items():
    if name not in params_cpp:
        print(f"❓ Parameter '{name}' exists in PyTorch model but not in C++ model.")
        gradients_match = False
        continue

    param_cpp = params_cpp[name]

    if param_py.grad is None and param_cpp.grad is None:
        # print(f"Parameter '{name}': No gradient in either model (expected for some layers?).")
        continue
    elif param_py.grad is None:
        print(f"❌ Parameter '{name}': No gradient in PyTorch model, but gradient exists in C++ model.")
        print("CPP Grad Norm:", torch.norm(param_cpp.grad).item())
        gradients_match = False
        mismatched_params.append(name)
        continue
    elif param_cpp.grad is None:
        print(f"❌ Parameter '{name}': Gradient exists in PyTorch model, but no gradient in C++ model.")
        print("PY Grad Norm:", torch.norm(param_py.grad).item())
        gradients_match = False
        mismatched_params.append(name)
        continue

    if torch.allclose(param_py.grad, param_cpp.grad, atol=atol, rtol=rtol):
        pass
    else:
        print(f"❌ Parameter '{name}': Gradients DO NOT match.")
        diff = torch.abs(param_py.grad - param_cpp.grad)
        print(f"   Max difference: {torch.max(diff).item():.6g}")
        print(f"   PyTorch Grad Norm: {torch.norm(param_py.grad).item():.6g}")
        print(f"   C++ Grad Norm: {torch.norm(param_cpp.grad).item():.6g}")
        gradients_match = False
        mismatched_params.append(name)

print("--- Summary ---")
if outputs_match:
    print("✅ All forward pass outputs match within tolerance.")
else:
    print("❌ Some forward pass outputs DO NOT match.")

if gradients_match:
    print("✅ All parameter gradients match within tolerance.")
else:
    print(f"❌ Gradients DO NOT match for parameters: {mismatched_params}")

if outputs_match and gradients_match:
    print("🎉 Test Passed: PyTorch and C++ implementations appear equivalent! 🎉")
else:
    print("🔥 Test Failed: Implementations differ. 🔥")
