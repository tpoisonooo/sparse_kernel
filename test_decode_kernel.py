#!/usr/bin/env python3
"""Test script for the optimized get_block_table_decode kernel."""

import torch
import time
import sparse_kernel_extension


def test_correctness():
    """Test that the new kernel produces the same output as v3."""
    print("=" * 60)
    print("Testing Decode Kernel Correctness")
    print("=" * 60)
    
    # Test parameters for decode (small batch, typical decode scenario)
    batch_size = 8
    token_num = 8  # In decode, token_num = batch_size
    topk = 96
    seqlen_q_max = 32768
    head_group = 2
    
    # Create test inputs
    torch.manual_seed(42)
    
    # topk_idx: [head_group, token_num, topk]
    topk_idx = torch.randint(0, 100, (head_group, token_num, topk), dtype=torch.int32, device='cuda')
    
    # block_table: [batch_size, seqlen_q_max]
    block_table = torch.arange(batch_size * seqlen_q_max, dtype=torch.int32, device='cuda').reshape(batch_size, seqlen_q_max)
    
    # token_to_bs: [token_num] - in decode, each token belongs to one batch
    token_to_bs = torch.arange(token_num, dtype=torch.int32, device='cuda') % batch_size
    
    # token_pos_in_bs: [token_num] - position in the sequence
    token_pos_in_bs = torch.full((token_num,), 1000, dtype=torch.int32, device='cuda')
    
    # seqlen_q: [batch_size]
    seqlen_q = torch.full((batch_size,), seqlen_q_max, dtype=torch.int32, device='cuda')
    
    # Run v3 kernel
    out_v3 = sparse_kernel_extension.get_block_table_v3(
        topk_idx, block_table, token_to_bs, token_pos_in_bs, seqlen_q, topk
    )
    
    # Run new decode kernel
    out_decode = sparse_kernel_extension.get_block_table_decode(
        topk_idx, block_table, token_to_bs, token_pos_in_bs, seqlen_q, topk
    )
    
    # Compare outputs
    if torch.equal(out_v3, out_decode):
        print("✓ Correctness test PASSED: outputs match!")
    else:
        print("✗ Correctness test FAILED: outputs differ!")
        diff = (out_v3 != out_decode).sum().item()
        print(f"  Number of differing elements: {diff}")
        print(f"  Max difference: {(out_v3 - out_decode).abs().max().item()}")
        
        # Find first difference
        diff_idx = (out_v3 != out_decode).nonzero(as_tuple=True)
        if len(diff_idx[0]) > 0:
            idx = diff_idx[0][0].item()
            print(f"  First difference at index: {idx}")
            print(f"  v3 value: {out_v3.flatten()[idx].item()}")
            print(f"  decode value: {out_decode.flatten()[idx].item()}")
        return False
    
    print(f"  Output shape: {out_v3.shape}")
    print(f"  Output dtype: {out_v3.dtype}")
    print()
    return True


def benchmark_kernel(kernel_fn, name, topk_idx, block_table, token_to_bs, token_pos_in_bs, seqlen_q, topk, num_iters=1000):
    """Benchmark a kernel function."""
    # Warmup
    for _ in range(100):
        _ = kernel_fn(topk_idx, block_table, token_to_bs, token_pos_in_bs, seqlen_q, topk)
    torch.cuda.synchronize()
    
    # Benchmark
    start = time.perf_counter()
    for _ in range(num_iters):
        _ = kernel_fn(topk_idx, block_table, token_to_bs, token_pos_in_bs, seqlen_q, topk)
    torch.cuda.synchronize()
    end = time.perf_counter()
    
    elapsed_us = (end - start) * 1e6 / num_iters
    return elapsed_us


def test_performance():
    """Benchmark the performance of v3 vs decode kernel."""
    print("=" * 60)
    print("Testing Decode Kernel Performance")
    print("=" * 60)
    
    seqlen_q_max = 32768
    topk = 96
    head_group = 2
    
    # Test different batch sizes (typical decode scenarios)
    batch_sizes = [1, 4, 8, 16, 32, 64]
    
    print(f"{'Batch Size':<12} {'v3 (us)':<12} {'decode (us)':<15} {'Speedup':<10}")
    print("-" * 60)
    
    for batch_size in batch_sizes:
        token_num = batch_size  # In decode, token_num = batch_size
        
        # Create test inputs
        torch.manual_seed(42)
        
        topk_idx = torch.randint(0, 100, (head_group, token_num, topk), dtype=torch.int32, device='cuda')
        block_table = torch.arange(batch_size * seqlen_q_max, dtype=torch.int32, device='cuda').reshape(batch_size, seqlen_q_max)
        token_to_bs = torch.arange(token_num, dtype=torch.int32, device='cuda') % batch_size
        token_pos_in_bs = torch.full((token_num,), 1000, dtype=torch.int32, device='cuda')
        seqlen_q = torch.full((batch_size,), seqlen_q_max, dtype=torch.int32, device='cuda')
        
        # Benchmark v3
        time_v3 = benchmark_kernel(
            sparse_kernel_extension.get_block_table_v3,
            "v3",
            topk_idx, block_table, token_to_bs, token_pos_in_bs, seqlen_q, topk,
            num_iters=2000
        )
        
        # Benchmark decode
        time_decode = benchmark_kernel(
            sparse_kernel_extension.get_block_table_decode,
            "decode",
            topk_idx, block_table, token_to_bs, token_pos_in_bs, seqlen_q, topk,
            num_iters=2000
        )
        
        speedup = time_v3 / time_decode
        print(f"{batch_size:<12} {time_v3:<12.2f} {time_decode:<15.2f} {speedup:<10.2f}x")
    
    print()


if __name__ == "__main__":
    print("Sparse Kernel Decode Optimization Test")
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
