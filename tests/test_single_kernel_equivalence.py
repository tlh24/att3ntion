"""
Test individual CUDA kernels against the PyTorch reference implementation.

Supports testing forward kernels (Y_q, Y_r, Y_s, Y_q_, Y_r_, Y_s_) and 
backward kernels (grad_Q, grad_R, grad_S, grad_V*) with pattern analysis
for debugging failures.

Usage:
    python tests/test_single_kernel.py --kernel Y_q
    python tests/test_single_kernel.py --kernel all --mode forward
    python tests/test_single_kernel.py --kernel grad_Q --mode backward --verbose
"""
import os
import sys
import argparse
import torch

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

TEST_CASES = [
    # Minimal cases
    {'name': 'minimal_N1',       'B': 1, 'H': 1, 'I': 1,  'J': 1,  'K': 1,  'D': 4},
    {'name': 'minimal_N2',       'B': 1, 'H': 1, 'I': 2,  'J': 2,  'K': 2,  'D': 4},
    {'name': 'minimal_N4',       'B': 1, 'H': 1, 'I': 4,  'J': 4,  'K': 4,  'D': 4},
    {'name': 'minimal_D4',       'B': 1, 'H': 1, 'I': 8,  'J': 8,  'K': 8,  'D': 4},
    {'name': 'minimal_D8',       'B': 1, 'H': 1, 'I': 8,  'J': 8,  'K': 8,  'D': 8},

    # Tile boundary tests (scatter tile = 4, gather tile = 16)
    {'name': 'tile_sp_N3',       'B': 1, 'H': 2, 'I': 3,  'J': 3,  'K': 3,  'D': 32},
    {'name': 'tile_sp_N4',       'B': 1, 'H': 2, 'I': 4,  'J': 4,  'K': 4,  'D': 32},
    {'name': 'tile_sp_N5',       'B': 1, 'H': 2, 'I': 5,  'J': 5,  'K': 5,  'D': 32},
    {'name': 'tile_sp_N7',       'B': 1, 'H': 2, 'I': 7,  'J': 7,  'K': 7,  'D': 32},
    {'name': 'tile_sp_N8',       'B': 1, 'H': 2, 'I': 8,  'J': 8,  'K': 8,  'D': 32},
    {'name': 'tile_sp_N9',       'B': 1, 'H': 2, 'I': 9,  'J': 9,  'K': 9,  'D': 32},
    {'name': 'tile_N15',         'B': 1, 'H': 2, 'I': 15, 'J': 15, 'K': 15, 'D': 32},
    {'name': 'tile_N16',         'B': 1, 'H': 2, 'I': 16, 'J': 16, 'K': 16, 'D': 32},
    {'name': 'tile_N17',         'B': 1, 'H': 2, 'I': 17, 'J': 17, 'K': 17, 'D': 32},
    {'name': 'tile_N31',         'B': 1, 'H': 2, 'I': 31, 'J': 31, 'K': 31, 'D': 32},
    {'name': 'tile_N32',         'B': 1, 'H': 2, 'I': 32, 'J': 32, 'K': 32, 'D': 32},
    {'name': 'tile_N33',         'B': 1, 'H': 2, 'I': 33, 'J': 33, 'K': 33, 'D': 32},
    {'name': 'tile_N47',         'B': 1, 'H': 2, 'I': 47, 'J': 47, 'K': 47, 'D': 32},
    {'name': 'tile_N48',         'B': 1, 'H': 2, 'I': 48, 'J': 48, 'K': 48, 'D': 32},
    {'name': 'tile_N49',         'B': 1, 'H': 2, 'I': 49, 'J': 49, 'K': 49, 'D': 32},
    {'name': 'tile_N63',         'B': 1, 'H': 2, 'I': 63, 'J': 63, 'K': 63, 'D': 32},
    {'name': 'tile_N64',         'B': 1, 'H': 2, 'I': 64, 'J': 64, 'K': 64, 'D': 32},
    {'name': 'tile_N65',         'B': 1, 'H': 2, 'I': 65, 'J': 65, 'K': 65, 'D': 32},

    # Odd N values (partial tile handling)
    {'name': 'odd_N11',          'B': 1, 'H': 2, 'I': 11, 'J': 11, 'K': 11, 'D': 32},
    {'name': 'odd_N13',          'B': 1, 'H': 2, 'I': 13, 'J': 13, 'K': 13, 'D': 32},
    {'name': 'odd_N19',          'B': 1, 'H': 2, 'I': 19, 'J': 19, 'K': 19, 'D': 32},
    {'name': 'odd_N23',          'B': 1, 'H': 2, 'I': 23, 'J': 23, 'K': 23, 'D': 32},
    {'name': 'odd_N29',          'B': 1, 'H': 2, 'I': 29, 'J': 29, 'K': 29, 'D': 32},
    {'name': 'odd_N37',          'B': 1, 'H': 2, 'I': 37, 'J': 37, 'K': 37, 'D': 32},
    {'name': 'odd_N41',          'B': 1, 'H': 2, 'I': 41, 'J': 41, 'K': 41, 'D': 32},
    {'name': 'odd_N43',          'B': 1, 'H': 2, 'I': 43, 'J': 43, 'K': 43, 'D': 32},
    {'name': 'odd_N53',          'B': 1, 'H': 2, 'I': 53, 'J': 53, 'K': 53, 'D': 32},
    {'name': 'odd_N59',          'B': 1, 'H': 2, 'I': 59, 'J': 59, 'K': 59, 'D': 32},
    {'name': 'odd_N61',          'B': 1, 'H': 2, 'I': 61, 'J': 61, 'K': 61, 'D': 32},
    {'name': 'odd_N67',          'B': 1, 'H': 2, 'I': 67, 'J': 67, 'K': 67, 'D': 32},
    {'name': 'odd_N71',          'B': 1, 'H': 2, 'I': 71, 'J': 71, 'K': 71, 'D': 32},
    {'name': 'odd_N73',          'B': 1, 'H': 2, 'I': 73, 'J': 73, 'K': 73, 'D': 32},
    {'name': 'odd_N79',          'B': 1, 'H': 2, 'I': 79, 'J': 79, 'K': 79, 'D': 32},

    # D dimension variations (must be <= 64, multiple of 4)
    {'name': 'D4_N32',           'B': 1, 'H': 2, 'I': 32, 'J': 32, 'K': 32, 'D': 4},
    {'name': 'D8_N32',           'B': 1, 'H': 2, 'I': 32, 'J': 32, 'K': 32, 'D': 8},
    {'name': 'D12_N32',          'B': 1, 'H': 2, 'I': 32, 'J': 32, 'K': 32, 'D': 12},
    {'name': 'D16_N32',          'B': 1, 'H': 2, 'I': 32, 'J': 32, 'K': 32, 'D': 16},
    {'name': 'D20_N32',          'B': 1, 'H': 2, 'I': 32, 'J': 32, 'K': 32, 'D': 20},
    {'name': 'D24_N32',          'B': 1, 'H': 2, 'I': 32, 'J': 32, 'K': 32, 'D': 24},
    {'name': 'D28_N32',          'B': 1, 'H': 2, 'I': 32, 'J': 32, 'K': 32, 'D': 28},
    {'name': 'D32_N32',          'B': 1, 'H': 2, 'I': 32, 'J': 32, 'K': 32, 'D': 32},
    {'name': 'D36_N32',          'B': 1, 'H': 2, 'I': 32, 'J': 32, 'K': 32, 'D': 36},
    {'name': 'D40_N32',          'B': 1, 'H': 2, 'I': 32, 'J': 32, 'K': 32, 'D': 40},
    {'name': 'D44_N32',          'B': 1, 'H': 2, 'I': 32, 'J': 32, 'K': 32, 'D': 44},
    {'name': 'D48_N32',          'B': 1, 'H': 2, 'I': 32, 'J': 32, 'K': 32, 'D': 48},
    {'name': 'D52_N32',          'B': 1, 'H': 2, 'I': 32, 'J': 32, 'K': 32, 'D': 52},
    {'name': 'D56_N32',          'B': 1, 'H': 2, 'I': 32, 'J': 32, 'K': 32, 'D': 56},
    {'name': 'D60_N32',          'B': 1, 'H': 2, 'I': 32, 'J': 32, 'K': 32, 'D': 60},
    {'name': 'D64_N32',          'B': 1, 'H': 2, 'I': 32, 'J': 32, 'K': 32, 'D': 64},

    # Batch size variations
    {'name': 'B1_N32',           'B': 1, 'H': 2, 'I': 32, 'J': 32, 'K': 32, 'D': 32},
    {'name': 'B2_N32',           'B': 2, 'H': 2, 'I': 32, 'J': 32, 'K': 32, 'D': 32},
    {'name': 'B3_N32',           'B': 3, 'H': 2, 'I': 32, 'J': 32, 'K': 32, 'D': 32},
    {'name': 'B4_N32',           'B': 4, 'H': 2, 'I': 32, 'J': 32, 'K': 32, 'D': 32},
    {'name': 'B5_N32',           'B': 5, 'H': 2, 'I': 32, 'J': 32, 'K': 32, 'D': 32},
    {'name': 'B6_N24',           'B': 6, 'H': 2, 'I': 24, 'J': 24, 'K': 24, 'D': 32},
    {'name': 'B7_N24',           'B': 7, 'H': 2, 'I': 24, 'J': 24, 'K': 24, 'D': 32},
    {'name': 'B8_N24',           'B': 8, 'H': 2, 'I': 24, 'J': 24, 'K': 24, 'D': 32},

    # Head variations
    {'name': 'H1_N32',           'B': 1, 'H': 1, 'I': 32, 'J': 32, 'K': 32, 'D': 32},
    {'name': 'H2_N32',           'B': 1, 'H': 2, 'I': 32, 'J': 32, 'K': 32, 'D': 32},
    {'name': 'H3_N32',           'B': 1, 'H': 3, 'I': 32, 'J': 32, 'K': 32, 'D': 32},
    {'name': 'H4_N32',           'B': 1, 'H': 4, 'I': 32, 'J': 32, 'K': 32, 'D': 32},
    {'name': 'H5_N32',           'B': 1, 'H': 5, 'I': 32, 'J': 32, 'K': 32, 'D': 32},
    {'name': 'H6_N32',           'B': 1, 'H': 6, 'I': 32, 'J': 32, 'K': 32, 'D': 32},
    {'name': 'H7_N24',           'B': 1, 'H': 7, 'I': 24, 'J': 24, 'K': 24, 'D': 32},
    {'name': 'H8_N24',           'B': 1, 'H': 8, 'I': 24, 'J': 24, 'K': 24, 'D': 32},
    {'name': 'H12_N16',          'B': 1, 'H': 12,'I': 16, 'J': 16, 'K': 16, 'D': 32},
    {'name': 'H16_N16',          'B': 1, 'H': 16,'I': 16, 'J': 16, 'K': 16, 'D': 32},

    # Combined B*H variations
    {'name': 'B2H4_N32',         'B': 2, 'H': 4, 'I': 32, 'J': 32, 'K': 32, 'D': 32},
    {'name': 'B3H3_N24',         'B': 3, 'H': 3, 'I': 24, 'J': 24, 'K': 24, 'D': 32},
    {'name': 'B4H2_N32',         'B': 4, 'H': 2, 'I': 32, 'J': 32, 'K': 32, 'D': 32},
    {'name': 'B2H8_N24',         'B': 2, 'H': 8, 'I': 24, 'J': 24, 'K': 24, 'D': 32},
    {'name': 'B4H4_N16',         'B': 4, 'H': 4, 'I': 16, 'J': 16, 'K': 16, 'D': 32},
    {'name': 'B8H2_N16',         'B': 8, 'H': 2, 'I': 16, 'J': 16, 'K': 16, 'D': 32},

    # Medium sequence lengths
    {'name': 'med_N40',          'B': 1, 'H': 2, 'I': 40, 'J': 40, 'K': 40, 'D': 32},
    {'name': 'med_N48',          'B': 1, 'H': 2, 'I': 48, 'J': 48, 'K': 48, 'D': 32},
    {'name': 'med_N52',          'B': 1, 'H': 2, 'I': 52, 'J': 52, 'K': 52, 'D': 32},
    {'name': 'med_N54',          'B': 1, 'H': 2, 'I': 54, 'J': 54, 'K': 54, 'D': 32},
    {'name': 'med_N56',          'B': 1, 'H': 2, 'I': 56, 'J': 56, 'K': 56, 'D': 32},
    {'name': 'med_N58',          'B': 1, 'H': 2, 'I': 58, 'J': 58, 'K': 58, 'D': 32},
    {'name': 'med_N60',          'B': 1, 'H': 2, 'I': 60, 'J': 60, 'K': 60, 'D': 32},
    {'name': 'med_N64',          'B': 1, 'H': 4, 'I': 64, 'J': 64, 'K': 64, 'D': 32},
    {'name': 'med_N72',          'B': 1, 'H': 2, 'I': 72, 'J': 72, 'K': 72, 'D': 32},
    {'name': 'med_N80',          'B': 1, 'H': 2, 'I': 80, 'J': 80, 'K': 80, 'D': 32},
    {'name': 'med_N88',          'B': 1, 'H': 2, 'I': 88, 'J': 88, 'K': 88, 'D': 32},
    {'name': 'med_N96',          'B': 1, 'H': 2, 'I': 96, 'J': 96, 'K': 96, 'D': 32},
    {'name': 'med_N100',         'B': 1, 'H': 2, 'I': 100,'J': 100,'K': 100,'D': 32},
    {'name': 'med_N112',         'B': 1, 'H': 2, 'I': 112,'J': 112,'K': 112,'D': 32},
    {'name': 'med_N120',         'B': 1, 'H': 2, 'I': 120,'J': 120,'K': 120,'D': 32},
    {'name': 'med_N128',         'B': 1, 'H': 2, 'I': 128,'J': 128,'K': 128,'D': 32},

    # Large sequence lengths
    {'name': 'large_N128',       'B': 2, 'H': 6, 'I': 128,'J': 128,'K': 128,'D': 64},
    {'name': 'large_N160',       'B': 2, 'H': 4, 'I': 160,'J': 160,'K': 160,'D': 64},
    {'name': 'large_N192',       'B': 2, 'H': 2, 'I': 192,'J': 192,'K': 192,'D': 64},
    {'name': 'large_N224',       'B': 2, 'H': 2, 'I': 224,'J': 224,'K': 224,'D': 32},
    {'name': 'large_N256',       'B': 2, 'H': 2, 'I': 256,'J': 256,'K': 256,'D': 32},
    {'name': 'large_N288',       'B': 2, 'H': 2, 'I': 288,'J': 288,'K': 288,'D': 32},
    {'name': 'large_N320',       'B': 1, 'H': 2, 'I': 320,'J': 320,'K': 320,'D': 32},
    {'name': 'large_N352',       'B': 1, 'H': 2, 'I': 352,'J': 352,'K': 352,'D': 32},
    {'name': 'large_N384',       'B': 1, 'H': 2, 'I': 384,'J': 384,'K': 384,'D': 32},
    {'name': 'large_N416',       'B': 1, 'H': 2, 'I': 416,'J': 416,'K': 416,'D': 32},

    # Cross-product edge cases (D × N interactions)
    {'name': 'D4_N17',           'B': 1, 'H': 2, 'I': 17, 'J': 17, 'K': 17, 'D': 4},
    {'name': 'D8_N33',           'B': 1, 'H': 2, 'I': 33, 'J': 33, 'K': 33, 'D': 8},
    {'name': 'D12_N15',          'B': 1, 'H': 2, 'I': 15, 'J': 15, 'K': 15, 'D': 12},
    {'name': 'D16_N31',          'B': 1, 'H': 2, 'I': 31, 'J': 31, 'K': 31, 'D': 16},
    {'name': 'D20_N47',          'B': 1, 'H': 2, 'I': 47, 'J': 47, 'K': 47, 'D': 20},
    {'name': 'D24_N49',          'B': 1, 'H': 2, 'I': 49, 'J': 49, 'K': 49, 'D': 24},
    {'name': 'D28_N63',          'B': 1, 'H': 2, 'I': 63, 'J': 63, 'K': 63, 'D': 28},
    {'name': 'D36_N65',          'B': 1, 'H': 2, 'I': 65, 'J': 65, 'K': 65, 'D': 36},
    {'name': 'D40_N97',          'B': 1, 'H': 2, 'I': 97, 'J': 97, 'K': 97, 'D': 40},
    {'name': 'D48_N127',         'B': 1, 'H': 1, 'I': 127,'J': 127,'K': 127,'D': 48},
    {'name': 'D52_N129',         'B': 1, 'H': 1, 'I': 129,'J': 129,'K': 129,'D': 52},
    {'name': 'D56_N255',         'B': 1, 'H': 1, 'I': 255,'J': 255,'K': 255,'D': 56},
    {'name': 'D60_N257',         'B': 1, 'H': 1, 'I': 257,'J': 257,'K': 257,'D': 60},
    {'name': 'D64_N127',         'B': 1, 'H': 1, 'I': 127,'J': 127,'K': 127,'D': 64},

    # Stress tests (numerical stability)
    {'name': 'stress_s2_N32',    'B': 1, 'H': 2, 'I': 32, 'J': 32, 'K': 32, 'D': 32, 'input_scale': 2.0},
    {'name': 'stress_s3_N32',    'B': 1, 'H': 2, 'I': 32, 'J': 32, 'K': 32, 'D': 32, 'input_scale': 3.0},
    {'name': 'stress_s5_N16',    'B': 1, 'H': 2, 'I': 16, 'J': 16, 'K': 16, 'D': 32, 'input_scale': 5.0},
    {'name': 'stress_s2_N64',    'B': 1, 'H': 2, 'I': 64, 'J': 64, 'K': 64, 'D': 32, 'input_scale': 2.0},
    {'name': 'stress_s2_D64',    'B': 1, 'H': 2, 'I': 32, 'J': 32, 'K': 32, 'D': 64, 'input_scale': 2.0},
    {'name': 'stress_s3_D64',    'B': 1, 'H': 1, 'I': 24, 'J': 24, 'K': 24, 'D': 64, 'input_scale': 3.0},

    # Production-like configurations
    {'name': 'prod_B4H8_N64',    'B': 4, 'H': 8, 'I': 64, 'J': 64, 'K': 64, 'D': 32},
    {'name': 'prod_B2H12_N48',   'B': 2, 'H': 12,'I': 48, 'J': 48, 'K': 48, 'D': 32},
    {'name': 'prod_B8H4_N32',    'B': 8, 'H': 4, 'I': 32, 'J': 32, 'K': 32, 'D': 64},
    {'name': 'prod_B1H16_N128',  'B': 1, 'H': 16,'I': 128,'J': 128,'K': 128,'D': 32},
]

FORWARD_KERNEL_NAMES = ['Y_q', 'Y_r', 'Y_s', 'Y_q_', 'Y_r_', 'Y_s_']
BACKWARD_KERNEL_NAMES = ['grad_Q', 'grad_R', 'grad_S', 'grad_Vq_1', 'grad_Vq_2', 'grad_Vr_1', 'grad_Vr_2', 'grad_Vs_1', 'grad_Vs_2']

QUICK_TEST_NAMES = [
    'minimal_N4', 'minimal_D4', 'tile_sp_N5', 'tile_N16', 'tile_N17',
    'tile_N32', 'tile_N33', 'odd_N23', 'odd_N37', 'D16_N32', 'D32_N32',
    'D64_N32', 'B2_N32', 'B4_N32', 'H4_N32', 'H8_N24', 'B2H4_N32',
    'med_N64', 'med_N96', 'large_N128', 'stress_s2_N32',
]


def run_test(cuda_ext, ref_ext, config, kernel_idx, device, rtol, atol, verbose, mode='forward'):
    """Run a single kernel test and return (passed, max_diff, error_msg)."""
    B, H, I, J, K, D = config['B'], config['H'], config['I'], config['J'], config['K'], config['D']
    scale = config.get('input_scale', 1.0)

    Q    = torch.randn(B, H, I, D, device=device) * scale
    R    = torch.randn(B, H, J, D, device=device) * scale
    S    = torch.randn(B, H, K, D, device=device) * scale
    Vq_1 = torch.randn(B, H, I, D, device=device) * scale
    Vq_2 = torch.randn(B, H, I, D, device=device) * scale
    Vr_1 = torch.randn(B, H, J, D, device=device) * scale
    Vr_2 = torch.randn(B, H, J, D, device=device) * scale
    Vs_1 = torch.randn(B, H, K, D, device=device) * scale
    Vs_2 = torch.randn(B, H, K, D, device=device) * scale

    try:
        if mode == 'forward':
            cuda_out = cuda_ext.forward(
                Q.clone(), R.clone(), S.clone(),
                Vq_1.clone(), Vq_2.clone(),
                Vr_1.clone(), Vr_2.clone(),
                Vs_1.clone(), Vs_2.clone(), 0.0
            )
            ref_out = ref_ext.forward(
                Q.clone(), R.clone(), S.clone(),
                Vq_1.clone(), Vq_2.clone(),
                Vr_1.clone(), Vr_2.clone(),
                Vs_1.clone(), Vs_2.clone(), 0.0
            )
        else:
            grad_output = torch.randn(B, H, max(I, J, K), D, device=device) * scale
            cuda_out = cuda_ext.backward(
                grad_output.clone(), Q.clone(), R.clone(), S.clone(),
                Vq_1.clone(), Vq_2.clone(),
                Vr_1.clone(), Vr_2.clone(),
                Vs_1.clone(), Vs_2.clone(), 0.0
            )

            tensors = [Q, R, S, Vq_1, Vq_2, Vr_1, Vr_2, Vs_1, Vs_2]
            tensors_ref = [t.detach().clone().requires_grad_(True) for t in tensors]
            Q_ref, R_ref, S_ref, Vq1_ref, Vq2_ref, Vr1_ref, Vr2_ref, Vs1_ref, Vs2_ref = tensors_ref

            outputs_ref = ref_ext.forward(
                Q_ref, R_ref, S_ref, Vq1_ref, Vq2_ref,
                Vr1_ref, Vr2_ref, Vs1_ref, Vs2_ref, 0.0
            )
            Y_q, Y_r, Y_s, Y_q_, Y_r_, Y_s_ = outputs_ref

            T_i = grad_output[:, :, :I, :]
            T_j = grad_output[:, :, :J, :]
            T_k = grad_output[:, :, :K, :]
            loss = ((Y_q * T_i).sum() + (Y_r * T_j).sum() + (Y_s * T_k).sum() +
                    (Y_q_ * T_i).sum() + (Y_r_ * T_j).sum() + (Y_s_ * T_k).sum())
            loss.backward()

            ref_out = (Q_ref.grad, R_ref.grad, S_ref.grad,
                       Vq1_ref.grad, Vq2_ref.grad,
                       Vr1_ref.grad, Vr2_ref.grad,
                       Vs1_ref.grad, Vs2_ref.grad)
    except Exception as e:
        return False, float('nan'), str(e)

    cuda_tensor = cuda_out[kernel_idx]
    ref_tensor = ref_out[kernel_idx]

    if cuda_tensor.shape != ref_tensor.shape:
        return False, float('nan'), f"Shape mismatch: {cuda_tensor.shape} vs {ref_tensor.shape}"

    max_diff = (cuda_tensor - ref_tensor).abs().max().item()
    passed = torch.allclose(cuda_tensor, ref_tensor, rtol=rtol, atol=atol)

    return passed, max_diff, None


def main():
    parser = argparse.ArgumentParser(
        description="Test individual CUDA kernels against PyTorch reference",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument('--mode', type=str, default='forward', choices=['forward', 'backward'],
                        help='Test forward or backward kernels')
    parser.add_argument('--kernel', type=str, required=True,
                        help='Kernel to test or "all" for all kernels in mode')
    parser.add_argument('--rtol', type=float, default=1e-4,
                        help='Relative tolerance (default: 1e-4)')
    parser.add_argument('--atol', type=float, default=1e-5,
                        help='Absolute tolerance (default: 1e-5)')
    parser.add_argument('--seed', type=int, default=0,
                        help='Random seed for reproducibility')
    parser.add_argument('--verbose', '-v', action='store_true',
                        help='Print all test results, not just failures')
    parser.add_argument('--quick', action='store_true',
                        help='Run quick test subset only')
    parser.add_argument('--filter', type=str, default=None,
                        help='Filter test cases by name substring')
    args = parser.parse_args()

    KERNEL_NAMES = FORWARD_KERNEL_NAMES if args.mode == 'forward' else BACKWARD_KERNEL_NAMES

    if args.kernel.lower() == 'all':
        kernels_to_test = KERNEL_NAMES
    else:
        if args.kernel not in KERNEL_NAMES:
            parser.error(f"--kernel must be one of {KERNEL_NAMES} or 'all' for {args.mode} mode")
        kernels_to_test = [args.kernel]

    test_cases = TEST_CASES
    if args.quick:
        test_cases = [c for c in TEST_CASES if c['name'] in QUICK_TEST_NAMES]
    if args.filter:
        test_cases = [c for c in test_cases if args.filter.lower() in c['name'].lower()]

    if not test_cases:
        print("No test cases match the specified criteria.")
        sys.exit(1)

    try:
        import hyper_attn_cpp_manual as cuda_ext
        import hyper_attn_cpp_reference as ref_ext
    except ImportError as e:
        print(f"Import error: {e}\nRun: python setup.py develop")
        sys.exit(1)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    all_kernel_results = {}
    
    for kernel_name in kernels_to_test:
        kernel_idx = KERNEL_NAMES.index(kernel_name)
        kernel_type = ("GATHER" if kernel_idx < 3 else "SCATTER") if args.mode == 'forward' else "BACKWARD"

        print(f"\n{'='*70}")
        print(f"Testing kernel: {kernel_name} ({kernel_type})")
        print(f"{'='*70}")
        print(f"rtol={args.rtol}, atol={args.atol}, seed={args.seed}, tests={len(test_cases)}")
        print()

        results = []
        first_error_shown = False
        fail_count = 0

        for i, config in enumerate(test_cases):
            torch.manual_seed(args.seed + i)
            name = config['name']

            passed, max_diff, error = run_test(
                cuda_ext, ref_ext, config, kernel_idx, device,
                args.rtol, args.atol, args.verbose, args.mode
            )

            results.append({
                'name': name,
                'passed': passed,
                'max_diff': max_diff,
                'config': config,
                'error': error
            })

            if not passed:
                fail_count += 1
                diff_str = f"{max_diff:.2e}" if not error else "ERROR"
                scale_str = f" s={config.get('input_scale', 1.0)}" if config.get('input_scale', 1.0) != 1.0 else ""
                print(f"  [FAIL] {name:<20} | B={config['B']:2} H={config['H']:2} N={config['I']:3} D={config['D']:2}{scale_str} | diff={diff_str}")
                if error and (args.verbose or not first_error_shown):
                    print(f"         Error: {error}")
                    first_error_shown = True
            elif args.verbose:
                diff_str = f"{max_diff:.2e}"
                scale_str = f" s={config.get('input_scale', 1.0)}" if config.get('input_scale', 1.0) != 1.0 else ""
                print(f"  [PASS] {name:<20} | B={config['B']:2} H={config['H']:2} N={config['I']:3} D={config['D']:2}{scale_str} | diff={diff_str}")

        if fail_count == 0:
            print(f"  All {len(test_cases)} tests passed ✓")

        all_kernel_results[kernel_name] = results

    total_passed = 0
    total_failed = 0

    for kernel_name, results in all_kernel_results.items():
        passed_list = [r for r in results if r['passed']]
        failed_list = [r for r in results if not r['passed']]
        total_passed += len(passed_list)
        total_failed += len(failed_list)

        print(f"\n{'='*70}")
        print(f"SUMMARY: {kernel_name}")
        print(f"{'='*70}")
        print(f"  Passed: {len(passed_list)}/{len(results)}")
        print(f"  Failed: {len(failed_list)}/{len(results)}")

        if failed_list:
            print(f"\n  FAILED CASES:")
            for r in failed_list:
                c = r['config']
                scale_str = f" scale={c.get('input_scale', 1.0)}" if c.get('input_scale', 1.0) != 1.0 else ""
                diff_str = f" diff={r['max_diff']:.2e}" if r['max_diff'] == r['max_diff'] else ""
                print(f"    - {r['name']:<20} B={c['B']:2} H={c['H']:2} N={c['I']:3} D={c['D']:2}{scale_str}{diff_str}")

            stress_fails = [r for r in failed_list if 'stress' in r['name'].lower()]
            non_stress_fails = [r for r in failed_list if 'stress' not in r['name'].lower()]

            if not non_stress_fails:
                print(f"\n  Note: All {len(stress_fails)} failure(s) are stress tests (numerical precision expected)")
                print(f"{'='*70}")
                continue

            print(f"\n  PATTERN ANALYSIS:")

            n_values = {}
            for r in results:
                n = r['config']['I']
                if n not in n_values:
                    n_values[n] = {'pass': 0, 'fail': 0}
                n_values[n]['pass' if r['passed'] else 'fail'] += 1

            print(f"    By N (sequence length):")
            for n in sorted(n_values.keys()):
                v = n_values[n]
                status = "✓" if v['fail'] == 0 else "✗" if v['pass'] == 0 else "~"
                print(f"      N={n:3}: {status} pass={v['pass']} fail={v['fail']}")

            d_values = {}
            for r in results:
                d = r['config']['D']
                if d not in d_values:
                    d_values[d] = {'pass': 0, 'fail': 0}
                d_values[d]['pass' if r['passed'] else 'fail'] += 1

            print(f"    By D (dimension):")
            for d in sorted(d_values.keys()):
                v = d_values[d]
                status = "✓" if v['fail'] == 0 else "✗" if v['pass'] == 0 else "~"
                print(f"      D={d:2}: {status} pass={v['pass']} fail={v['fail']}")

            bh_values = {}
            for r in results:
                bh = r['config']['B'] * r['config']['H']
                if bh not in bh_values:
                    bh_values[bh] = {'pass': 0, 'fail': 0}
                bh_values[bh]['pass' if r['passed'] else 'fail'] += 1

            print(f"    By B*H (batch*heads):")
            for bh in sorted(bh_values.keys()):
                v = bh_values[bh]
                status = "✓" if v['fail'] == 0 else "✗" if v['pass'] == 0 else "~"
                print(f"      B*H={bh:2}: {status} pass={v['pass']} fail={v['fail']}")

            tile_fails = len([r for r in failed_list if 'tile' in r['name'].lower()])
            odd_fails = len([r for r in failed_list if 'odd' in r['name'].lower()])
            stress_count = len(stress_fails)

            if tile_fails:
                print(f"\n  Warning: {tile_fails} tile boundary test(s) failed")
            if odd_fails:
                print(f"  Warning: {odd_fails} odd N test(s) failed")
            if stress_count:
                print(f"  Warning: {stress_count} stress test(s) failed")

        print(f"{'='*70}")

    if len(kernels_to_test) > 1:
        print(f"\n{'='*70}")
        print(f"OVERALL SUMMARY")
        print(f"{'='*70}")
        print(f"  Total Passed: {total_passed}/{total_passed + total_failed}")
        print(f"  Total Failed: {total_failed}/{total_passed + total_failed}")
        for kernel_name, results in all_kernel_results.items():
            passed_count = sum(1 for r in results if r['passed'])
            status = "✓" if passed_count == len(results) else "✗"
            print(f"    {kernel_name:<12}: {status} {passed_count}/{len(results)}")
        print(f"{'='*70}")

    sys.exit(0 if total_failed == 0 else 1)


if __name__ == '__main__':
    main()
