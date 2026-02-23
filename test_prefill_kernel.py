#!/usr/bin/env python3
"""Test script for the optimized get_block_table_prefill kernel."""

import torch
import time
import sparse_kernel_extension


def test_correctness():
    """Test that the new kernel produces the same output as v2."""
    print("=" * 60)
    print("Testing Correctness")
    print("=" * 60)
    
    # Test parameters
    batch_size = 2
    token_num = 2048
    topk = 96
    seqlen_q_max = 8192
    head_group = 2
    
    # Create test inputs
    torch.manual_seed(42)
    
    # topk_idx: [head_group, token_num, topk]
    topk_idx = torch.randint(0, 100, (head_group, token_num, topk), dtype=torch.int32, device='cuda')
    
    # block_table: [batch_size, seqlen_q_max]
    block_table = torch.arange(batch_size * seqlen_q_max, dtype=torch.int32, device='cuda').reshape(batch_size, seqlen_q_max)
    
    # token_to_bs: [token_num]
    token_to_bs = torch.randint(0, batch_size, (token_num,), dtype=torch.int32, device='cuda')
    
    # token_pos_in_bs: [token_num]
    token_pos_in_bs = torch.randint(0, seqlen_q_max, (token_num,), dtype=torch.int32, device='cuda')
    
    # seqlen_q: [batch_size]
    seqlen_q = torch.full((batch_size,), seqlen_q_max, dtype=torch.int32, device='cuda')
    
    # Run v2 kernel
    out_v2 = sparse_kernel_extension.get_block_table_v2(
        topk_idx, block_table, token_to_bs, token_pos_in_bs, seqlen_q, topk
    )
    
    # Run new prefill kernel
    out_prefill = sparse_kernel_extension.get_block_table_prefill(
        topk_idx, block_table, token_to_bs, token_pos_in_bs, seqlen_q, topk
    )
    
    # Compare outputs
    if torch.equal(out_v2, out_prefill):
        print("✓ Correctness test PASSED: outputs match!")
    else:
        print("✗ Correctness test FAILED: outputs differ!")
        diff = (out_v2 != out_prefill).sum().item()
        print(f"  Number of differing elements: {diff}")
        print(f"  Max difference: {(out_v2 - out_prefill).abs().max().item()}")
        return False
    
    print(f"  Output shape: {out_v2.shape}")
    print(f"  Output dtype: {out_v2.dtype}")
    print()
    return True


def benchmark_kernel(kernel_fn, name, topk_idx, block_table, token_to_bs, token_pos_in_bs, seqlen_q, topk, num_iters=100):
    """Benchmark a kernel function."""
    # Warmup
    for _ in range(10):
        _ = kernel_fn(topk_idx, block_table, token_to_bs, token_pos_in_bs, seqlen_q, topk)
    torch.cuda.synchronize()
    
    # Benchmark
    start = time.perf_counter()
    for _ in range(num_iters):
        _ = kernel_fn(topk_idx, block_table, token_to_bs, token_pos_in_bs, seqlen_q, topk)
    torch.cuda.synchronize()
    end = time.perf_counter()
    
    elapsed_ms = (end - start) * 1000 / num_iters
    return elapsed_ms


def test_performance():
    """Benchmark the performance of v2 vs prefill kernel."""
    print("=" * 60)
    print("Testing Performance")
    print("=" * 60)
    
    batch_size = 4
    seqlen_q_max = 32768
    topk = 96
    head_group = 2
    
    # Test different token numbers
    token_nums = [512, 1024, 2048, 4096, 8192]
    
    print(f"{'Token Num':<12} {'v2 (ms)':<12} {'prefill (ms)':<15} {'Speedup':<10}")
    print("-" * 60)
    
    for token_num in token_nums:
        # Create test inputs
        torch.manual_seed(42)
        
        topk_idx = torch.randint(0, 100, (head_group, token_num, topk), dtype=torch.int32, device='cuda')
        block_table = torch.arange(batch_size * seqlen_q_max, dtype=torch.int32, device='cuda').reshape(batch_size, seqlen_q_max)
        token_to_bs = torch.randint(0, batch_size, (token_num,), dtype=torch.int32, device='cuda')
        token_pos_in_bs = torch.randint(0, seqlen_q_max, (token_num,), dtype=torch.int32, device='cuda')
        seqlen_q = torch.full((batch_size,), seqlen_q_max, dtype=torch.int32, device='cuda')
        
        # Benchmark v2
        time_v2 = benchmark_kernel(
            sparse_kernel_extension.get_block_table_v2,
            "v2",
            topk_idx, block_table, token_to_bs, token_pos_in_bs, seqlen_q, topk,
            num_iters=50 if token_num >= 4096 else 100
        )
        
        # Benchmark prefill
        time_prefill = benchmark_kernel(
            sparse_kernel_extension.get_block_table_prefill,
            "prefill",
            topk_idx, block_table, token_to_bs, token_pos_in_bs, seqlen_q, topk,
            num_iters=50 if token_num >= 4096 else 100
        )
        
        speedup = time_v2 / time_prefill
        print(f"{token_num:<12} {time_v2:<12.3f} {time_prefill:<15.3f} {speedup:<10.2f}x")
    
    print()


if __name__ == "__main__":
    print("Sparse Kernel Prefill Optimization Test")
    print("=" * 60)
    print()
    
    # Check CUDA is available
    if not torch.cuda.is_available():
        print("CUDA is not available!")
        exit(1)
    
    print(f"CUDA device: {torch.cuda.get_device_name(0)}")
    print()
    
    # Run tests
    success = test_correctness()
    if success:
        test_performance()
        print("All tests completed!")
    else:
        print("Tests failed!")
        exit(1)
