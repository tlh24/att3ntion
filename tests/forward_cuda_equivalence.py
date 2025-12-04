import os
import sys
import argparse
from typing import Tuple, Dict, Any
import torch

# Make project root importable
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)


def compare_tensors(name: str, a: torch.Tensor, b: torch.Tensor, rtol: float, atol: float) -> Tuple[bool, float]:
    """Compare two tensors and return (is_close, max_abs_diff)."""
    if a is None or b is None:
        return False, float('nan')
    same = torch.allclose(a, b, rtol=rtol, atol=atol)
    max_abs = (a - b).abs().max().item()
    return same, max_abs


def check_for_nan_inf(tensor: torch.Tensor, name: str, verbose: bool = True) -> Tuple[bool, int, int]:
    """Check tensor for NaN/Inf values. Returns (is_ok, nan_count, inf_count)."""
    if tensor is None:
        return True, 0, 0
    nan_count = torch.isnan(tensor).sum().item()
    inf_count = torch.isinf(tensor).sum().item()
    is_ok = nan_count == 0 and inf_count == 0
    if not is_ok and verbose:
        print(f"  WARNING: {name} has {nan_count} NaNs, {inf_count} Infs")
    return is_ok, nan_count, inf_count


def run_single_forward_test(manual_ext, ref_ext, config: Dict[str, Any], device: torch.device, 
                             rtol: float, atol: float, verbose: bool = True) -> Tuple[bool, Dict]:
    """
    Run a single forward pass equivalence test with the given configuration.
    
    Compares manual CUDA implementation against PyTorch C++ reference.
    
    Returns (all_ok, results_dict)
    """
    B = config['B']
    H = config['H']
    I = config['I']
    J = config['J']
    K = config['K']
    D = config['D']
    input_scale = config.get('input_scale', 1.0)
    dropout_rate = config.get('dropout_rate', 0.0)
    
    # Create inputs with optional scaling for stress tests
    Q    = torch.randn(B, H, I, D, device=device, dtype=torch.float32) * input_scale
    R    = torch.randn(B, H, J, D, device=device, dtype=torch.float32) * input_scale
    S    = torch.randn(B, H, K, D, device=device, dtype=torch.float32) * input_scale
    Vq_1 = torch.randn(B, H, I, D, device=device, dtype=torch.float32) * input_scale
    Vq_2 = torch.randn(B, H, I, D, device=device, dtype=torch.float32) * input_scale
    Vr_1 = torch.randn(B, H, J, D, device=device, dtype=torch.float32) * input_scale
    Vr_2 = torch.randn(B, H, J, D, device=device, dtype=torch.float32) * input_scale
    Vs_1 = torch.randn(B, H, K, D, device=device, dtype=torch.float32) * input_scale
    Vs_2 = torch.randn(B, H, K, D, device=device, dtype=torch.float32) * input_scale

    results = {'config': config, 'outputs': {}, 'nan_check': True}
    
    # --- Run manual CUDA forward ---
    torch.cuda.synchronize() if device.type == 'cuda' else None
    try:
        outputs_manual = manual_ext.forward(
            Q.clone(), R.clone(), S.clone(),
            Vq_1.clone(), Vq_2.clone(),
            Vr_1.clone(), Vr_2.clone(),
            Vs_1.clone(), Vs_2.clone(),
            dropout_rate
        )
        Y_q_m, Y_r_m, Y_s_m, Y_q__m, Y_r__m, Y_s__m = outputs_manual
    except Exception as e:
        if verbose:
            print(f"  ERROR: Manual CUDA forward raised exception: {e}")
        results['error'] = str(e)
        return False, results
    
    # --- Run reference forward ---
    try:
        outputs_ref = ref_ext.forward(
            Q.clone(), R.clone(), S.clone(),
            Vq_1.clone(), Vq_2.clone(),
            Vr_1.clone(), Vr_2.clone(),
            Vs_1.clone(), Vs_2.clone(),
            dropout_rate
        )
        Y_q_r, Y_r_r, Y_s_r, Y_q__r, Y_r__r, Y_s__r = outputs_ref
    except Exception as e:
        if verbose:
            print(f"  ERROR: Reference forward raised exception: {e}")
        results['error'] = str(e)
        return False, results
    
    # Output names and tensors
    output_names = ['Y_q', 'Y_r', 'Y_s', 'Y_q_', 'Y_r_', 'Y_s_']
    manual_outputs = [Y_q_m, Y_r_m, Y_s_m, Y_q__m, Y_r__m, Y_s__m]
    ref_outputs = [Y_q_r, Y_r_r, Y_s_r, Y_q__r, Y_r__r, Y_s__r]
    
    # --- Check for NaN/Inf in manual outputs ---
    for name, tensor in zip(output_names, manual_outputs):
        is_ok, nan_count, inf_count = check_for_nan_inf(tensor, name, verbose)
        if not is_ok:
            results['nan_check'] = False
    
    # --- Compare outputs ---
    all_ok = results['nan_check']
    for name, m_out, r_out in zip(output_names, manual_outputs, ref_outputs):
        # Check shape match first
        if m_out.shape != r_out.shape:
            if verbose:
                print(f"  {name:<8} | FAIL | Shape mismatch: {m_out.shape} vs {r_out.shape}")
            results['outputs'][name] = {'ok': False, 'max_abs_diff': float('nan'), 'shape_mismatch': True}
            all_ok = False
            continue
        
        ok, max_abs = compare_tensors(name, m_out, r_out, rtol, atol)
        results['outputs'][name] = {'ok': ok, 'max_abs_diff': max_abs}
        if verbose:
            status = 'PASS' if ok else 'FAIL'
            print(f"  {name:<8} | {status:<4} | max_abs_diff: {max_abs:.3e}")
        all_ok = all_ok and ok
    
    return all_ok, results


def main():
    parser = argparse.ArgumentParser("CUDA forward equivalence vs PyTorch reference")
    parser.add_argument('--B', type=int, default=1)
    parser.add_argument('--H', type=int, default=2)
    parser.add_argument('--N', type=int, default=8, help='Sequence length I=J=K=N (must be multiple of 8)')
    parser.add_argument('--D', type=int, default=32, help='Dimension (must be <= 64)')
    parser.add_argument('--rtol', type=float, default=1e-4)
    parser.add_argument('--atol', type=float, default=1e-5)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--sweep', action='store_true', 
                        help='Run full test sweep across multiple configurations')
    parser.add_argument('--stress', action='store_true',
                        help='Include numerical stability stress tests (large values)')
    parser.add_argument('--quick', action='store_true',
                        help='Run only small/quick test cases')
    parser.add_argument('--verbose', '-v', action='store_true',
                        help='Show detailed per-test output')
    args = parser.parse_args()

    torch.manual_seed(args.seed)

    # Import extensions
    try:
        import hyper_attn_cpp_manual as manual_ext
    except ImportError as e:
        print(f"Error: Could not import 'hyper_attn_cpp_manual': {e}")
        print("Please compile the extension first: python setup.py develop")
        sys.exit(1)

    try:
        import hyper_attn_cpp_reference as ref_ext
    except ImportError as e:
        print(f"Error: Could not import 'hyper_attn_cpp_reference': {e}")
        print("Please compile the extension first: python setup.py develop")
        sys.exit(1)

    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    if device.type != 'cuda':
        print("WARNING: CUDA not available, running on CPU")
    
    # =========================================================================
    # Test Case Definitions
    # =========================================================================
    # Constraints:
    #   - I = J = K (square only)
    #   - N (I=J=K) must be a multiple of 8
    #   - D <= 64
    
    # Quick tests (small, fast)
    quick_cases = [
        {'name': 'tiny_N8',          'B': 1, 'H': 1, 'I': 8,  'J': 8,  'K': 8,  'D': 16},
        {'name': 'small_N16',        'B': 1, 'H': 2, 'I': 16, 'J': 16, 'K': 16, 'D': 32},
        {'name': 'small_N24',        'B': 1, 'H': 1, 'I': 24, 'J': 24, 'K': 24, 'D': 32},
    ]
    
    # Standard tests (moderate size, covers common cases)
    standard_cases = [
        # Square cases with increasing N
        {'name': 'medium_N32',       'B': 1, 'H': 2, 'I': 32, 'J': 32, 'K': 32, 'D': 32},
        {'name': 'medium_N48',       'B': 1, 'H': 2, 'I': 48, 'J': 48, 'K': 48, 'D': 32},
        {'name': 'larger_N64',       'B': 2, 'H': 4, 'I': 64, 'J': 64, 'K': 64, 'D': 32},
        {'name': 'large_N128',       'B': 1, 'H': 2, 'I': 128,'J': 128,'K': 128,'D': 32},
        
        # Multi-batch configurations
        {'name': 'multi_batch_B4',   'B': 4, 'H': 2, 'I': 32, 'J': 32, 'K': 32, 'D': 32},
        {'name': 'multi_batch_B8',   'B': 8, 'H': 1, 'I': 32, 'J': 32, 'K': 32, 'D': 32},
        
        # Multi-head configurations
        {'name': 'multi_head_H4',    'B': 1, 'H': 4, 'I': 32, 'J': 32, 'K': 32, 'D': 32},
        {'name': 'multi_head_H8',    'B': 1, 'H': 8, 'I': 32, 'J': 32, 'K': 32, 'D': 32},
        {'name': 'multi_head_H12',   'B': 1, 'H': 12,'I': 32, 'J': 32, 'K': 32, 'D': 32},
        
        # Combined batch + head
        {'name': 'multi_B2_H4',      'B': 2, 'H': 4, 'I': 32, 'J': 32, 'K': 32, 'D': 32},
        {'name': 'multi_B4_H8',      'B': 4, 'H': 8, 'I': 32, 'J': 32, 'K': 32, 'D': 32},
        
        # Different D values (D <= 64)
        {'name': 'small_D8',         'B': 1, 'H': 2, 'I': 32, 'J': 32, 'K': 32, 'D': 8},
        {'name': 'small_D16',        'B': 1, 'H': 2, 'I': 32, 'J': 32, 'K': 32, 'D': 16},
        {'name': 'medium_D48',       'B': 1, 'H': 2, 'I': 32, 'J': 32, 'K': 32, 'D': 48},
        {'name': 'max_D64',          'B': 1, 'H': 2, 'I': 32, 'J': 32, 'K': 32, 'D': 64},
    ]
    
    # Stress tests (numerical stability)
    stress_cases = [
        # Large input values -> large logits -> tests overflow protection
        {'name': 'stress_scale3',        'B': 1, 'H': 2, 'I': 32, 'J': 32, 'K': 32, 'D': 32, 'input_scale': 3.0},
        {'name': 'stress_scale5',        'B': 1, 'H': 1, 'I': 16, 'J': 16, 'K': 16, 'D': 32, 'input_scale': 5.0},
        
        # Larger sequences with stress
        {'name': 'stress_N64_scale2',    'B': 1, 'H': 2, 'I': 64, 'J': 64, 'K': 64, 'D': 32, 'input_scale': 2.0},
        {'name': 'stress_N128_scale2',   'B': 1, 'H': 2, 'I': 128,'J': 128,'K': 128,'D': 32, 'input_scale': 2.0},
        
        # Stress with max D
        {'name': 'stress_D64_scale3',    'B': 1, 'H': 2, 'I': 32, 'J': 32, 'K': 32, 'D': 64, 'input_scale': 3.0},
    ]
    
    # =========================================================================
    # Select test cases based on flags
    # =========================================================================
    
    if args.sweep or args.stress or args.quick:
        # Running multiple test cases
        if args.quick:
            test_cases = quick_cases
        elif args.stress:
            test_cases = quick_cases + standard_cases + stress_cases
        else:  # --sweep
            test_cases = quick_cases + standard_cases
        
        passed = 0
        failed = 0
        failed_cases = []
        
        # Track per-output statistics
        output_stats = {name: {'pass': 0, 'fail': 0} for name in ['Y_q', 'Y_r', 'Y_s', 'Y_q_', 'Y_r_', 'Y_s_']}
        
        if not args.verbose:
            print(f"Running {len(test_cases)} tests", end="", flush=True)
        
        for i, config in enumerate(test_cases):
            name = config.get('name', f'test_{i}')
            
            if args.verbose:
                scale_info = f" (input_scale={config.get('input_scale', 1.0)})" if config.get('input_scale', 1.0) != 1.0 else ""
                print(f"\n[{i+1}/{len(test_cases)}] {name}: B={config['B']}, H={config['H']}, "
                      f"I=J=K={config['I']}, D={config['D']}{scale_info}")
                print("-" * 60)
            
            torch.manual_seed(args.seed + i)  # Different seed per test for variety
            
            ok, results = run_single_forward_test(manual_ext, ref_ext, config, device, 
                                                   args.rtol, args.atol, verbose=args.verbose)
            
            # Track per-output stats
            for out_name, out_result in results.get('outputs', {}).items():
                if out_result.get('ok', False):
                    output_stats[out_name]['pass'] += 1
                else:
                    output_stats[out_name]['fail'] += 1
            
            if ok:
                passed += 1
                if args.verbose:
                    print(f"  => PASSED")
                else:
                    print(".", end="", flush=True)
            else:
                failed += 1
                failed_cases.append(name)
                if args.verbose:
                    print(f"  => FAILED")
                else:
                    print("x", end="", flush=True)
        
        if not args.verbose:
            print(" done.\n")
        
        # Summary
        print("\n" + "="*80)
        print("SUMMARY")
        print("="*80)
        
        # Overall results
        total = len(test_cases)
        pass_pct = 100 * passed / total if total > 0 else 0
        print(f"\n{'OVERALL RESULTS':^80}")
        print("-" * 80)
        print(f"  Total Tests: {total}")
        print(f"  Passed:      {passed} ({pass_pct:.0f}%)")
        print(f"  Failed:      {failed} ({100-pass_pct:.0f}%)")
        
        # Per-output breakdown table
        print(f"\n{'PER-OUTPUT BREAKDOWN':^80}")
        print("-" * 80)
        print(f"  {'Output':<8} | {'Pass':<6} | {'Fail':<6} | {'Rate':<8} | {'Status'}")
        print(f"  {'-'*8}-+-{'-'*6}-+-{'-'*6}-+-{'-'*8}-+-{'-'*20}")
        
        gather_pass = 0
        gather_fail = 0
        scatter_pass = 0
        scatter_fail = 0
        
        for out_name in ['Y_q', 'Y_r', 'Y_s', 'Y_q_', 'Y_r_', 'Y_s_']:
            stats = output_stats[out_name]
            total_out = stats['pass'] + stats['fail']
            pct = 100 * stats['pass'] / total_out if total_out > 0 else 0
            if stats['fail'] == 0:
                status = "✓ OK"
            elif stats['pass'] == 0:
                status = "✗ ALL FAIL"
            else:
                status = f"⚠ PARTIAL"
            
            # Track gather vs scatter
            if out_name in ['Y_q', 'Y_r', 'Y_s']:
                gather_pass += stats['pass']
                gather_fail += stats['fail']
            else:
                scatter_pass += stats['pass']
                scatter_fail += stats['fail']
            
            print(f"  {out_name:<8} | {stats['pass']:<6} | {stats['fail']:<6} | {pct:>5.0f}%   | {status}")
        
        print(f"  {'-'*8}-+-{'-'*6}-+-{'-'*6}-+-{'-'*8}-+-{'-'*20}")
        
        # Gather vs Scatter summary
        gather_total = gather_pass + gather_fail
        scatter_total = scatter_pass + scatter_fail
        gather_pct = 100 * gather_pass / gather_total if gather_total > 0 else 0
        scatter_pct = 100 * scatter_pass / scatter_total if scatter_total > 0 else 0
        
        print(f"  {'GATHER':<8} | {gather_pass:<6} | {gather_fail:<6} | {gather_pct:>5.0f}%   | {'✓ OK' if gather_fail == 0 else '⚠ ISSUES'}")
        print(f"  {'SCATTER':<8} | {scatter_pass:<6} | {scatter_fail:<6} | {scatter_pct:>5.0f}%   | {'✓ OK' if scatter_fail == 0 else '⚠ ISSUES'}")
        
        # Diagnostic breakdown
        print(f"\n{'DIAGNOSTIC BREAKDOWN':^80}")
        print("-" * 80)
        print(f"  {'Category':<20} | {'Status':<12} | {'Details'}")
        print(f"  {'-'*20}-+-{'-'*12}-+-{'-'*40}")
        
        # Analyze patterns in failures
        small_n_fails = [c for c in failed_cases if 'N8' in c or 'N16' in c or 'N24' in c]
        medium_n_fails = [c for c in failed_cases if 'N32' in c or 'N48' in c]
        large_n_fails = [c for c in failed_cases if 'N64' in c or 'N128' in c]
        batch_fails = [c for c in failed_cases if 'batch' in c.lower() or c.startswith('multi_B')]
        head_fails = [c for c in failed_cases if 'head' in c.lower() or '_H' in c]
        d_fails = [c for c in failed_cases if '_D' in c]
        stress_fails = [c for c in failed_cases if 'stress' in c.lower()]
        
        def status_icon(count, total_cat):
            if count == 0: return "✓ OK"
            elif count == total_cat: return "✗ ALL FAIL"
            else: return f"⚠ {count} fail"
        
        small_n_total = len([c for c in test_cases if 'N8' in c.get('name','') or 'N16' in c.get('name','') or 'N24' in c.get('name','')])
        medium_n_total = len([c for c in test_cases if 'N32' in c.get('name','') or 'N48' in c.get('name','')])
        large_n_total = len([c for c in test_cases if 'N64' in c.get('name','') or 'N128' in c.get('name','')])
        
        print(f"  {'Small N (8-24)':<20} | {status_icon(len(small_n_fails), small_n_total):<12} | {', '.join(small_n_fails[:3]) if small_n_fails else 'All passing'}")
        print(f"  {'Medium N (32-48)':<20} | {status_icon(len(medium_n_fails), medium_n_total):<12} | {', '.join(medium_n_fails[:3]) if medium_n_fails else 'All passing'}")
        print(f"  {'Large N (64-128)':<20} | {status_icon(len(large_n_fails), large_n_total):<12} | {', '.join(large_n_fails[:3]) if large_n_fails else 'All passing'}")
        print(f"  {'Multi-Batch':<20} | {status_icon(len(batch_fails), 4):<12} | {', '.join(batch_fails[:3]) if batch_fails else 'All passing'}")
        print(f"  {'Multi-Head':<20} | {status_icon(len(head_fails), 5):<12} | {', '.join(head_fails[:3]) if head_fails else 'All passing'}")
        print(f"  {'D Variations':<20} | {status_icon(len(d_fails), 4):<12} | {', '.join(d_fails[:3]) if d_fails else 'All passing'}")
        print(f"  {'Stress Tests':<20} | {status_icon(len(stress_fails), 5):<12} | {', '.join(stress_fails[:3]) if stress_fails else 'All passing'}")
        
        # Root cause analysis
        print(f"\n{'ROOT CAUSE ANALYSIS':^80}")
        print("-" * 80)
        
        if scatter_fail > 0 and gather_fail == 0:
            print("  🔍 SCATTER kernels failing while GATHER kernels pass")
            print("     → Focus on: forward scatter kernel (atomicAdd, accumulation)")
        elif gather_fail > 0 and scatter_fail == 0:
            print("  🔍 GATHER kernels failing while SCATTER kernels pass")
            print("     → Focus on: forward gather kernel (softmax, normalization)")
        elif gather_fail > 0 and scatter_fail > 0:
            print("  🔍 Both GATHER and SCATTER kernels failing")
            print("     → Likely: shared issue (softmax stats m/l, indexing)")
        
        if len(small_n_fails) > 0:
            print("  🔍 Small N tests failing - fundamental logic bug")
            print("     → Debug with N=8 first (single block, no tiling)")
        elif len(medium_n_fails) > 0 and len(small_n_fails) == 0:
            print("  🔍 Medium N fails but Small N passes")
            print("     → Likely: TILING or BLOCK boundary issue (check N > TILE_SIZE)")
        
        if len(batch_fails) > 0 and len([c for c in failed_cases if 'B1' in c or 'batch' not in c.lower()]) == 0:
            print("  🔍 Only multi-batch tests failing")
            print("     → Check: batch stride calculation (bh indexing)")
        
        if passed == total:
            print("  ✓ All tests passing!")
        
        # Failed cases list (compact)
        if failed_cases:
            print(f"\n{'FAILED CASES ({len(failed_cases)} total)':^80}")
            print("-" * 80)
            # Print in columns
            cols = 3
            for i in range(0, len(failed_cases), cols):
                row = failed_cases[i:i+cols]
                print("  " + "  ".join(f"{c:<25}" for c in row))
        
        print("="*80)
        
        sys.exit(0 if failed == 0 else 2)
    
    else:
        # Single test mode (original behavior)
        # Validate constraints: N must be multiple of 8, D <= 64
        if args.N % 8 != 0:
            print(f"Error: N={args.N} must be a multiple of 8")
            sys.exit(1)
        if args.D > 64:
            print(f"Error: D={args.D} must be <= 64")
            sys.exit(1)
        
        I = J = K = args.N
        config = {
            'name': 'single_test',
            'B': args.B, 'H': args.H, 
            'I': I, 'J': J, 'K': K, 
            'D': args.D
        }
        
        print("\nForward Equivalence Check (CUDA vs Reference)")
        print(f"Config: B={args.B}, H={args.H}, I=J=K={I}, D={args.D}")
        print(f"Device: {device}")
        print(f"Tolerances: rtol={args.rtol}, atol={args.atol}")
        print("-" * 80)
        
        ok, results = run_single_forward_test(manual_ext, ref_ext, config, device, 
                                               args.rtol, args.atol, verbose=True)
        
        print("-" * 80)
        if ok:
            print("All forward outputs match within tolerance.")
            sys.exit(0)
        else:
            print("Mismatches detected.")
            sys.exit(2)


if __name__ == '__main__':
    main()

