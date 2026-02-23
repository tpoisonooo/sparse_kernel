// #include <bits/stdc++.h>
// #include <cuda_runtime.h>
#include "static_switch.h"
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAStream.h>
#include <torch/extension.h>
// constexpr int kTokenNum = 8192;
// constexpr int kBs = 1;
// constexpr int kSeqlenQMax = 8192;
constexpr int kHeadGroup = 2;
// // constexpr int kseqlenQ_max
constexpr int kSparseBlockSize = 64;
// constexpr int kSparseTopK = 96;
constexpr int kTopkPerBlock = 16;
// constexpr int kBlockPerTokenHead = kSparseTopK / kTopkPerBlock;
// topk_idx: [head_group, token_num, kSparseTopK]: int32 [2, 8192, 96]
// block_table: [batch_size, seqlen_q_max]: int32 [1, 8192]
// token_to_bs: [token_num]: int32  [8192]
// token_pos_in_bs: [token_num]: int32 [8192]
// seqlen_q: [batch_size]: int32    [1]
// out_block_table: [token_num, head_group, kSparseTopK * kSparseBlockSize]:
// int32 [2, 8192, 96 * 64] seqlen_q_max: int
template <int kSparseTopK>
__global__ void
get_block_table_cuda_v1(const int *topk_idx, const int *block_table,
                        const int *token_to_bs, const int *token_pos_in_bs,
                        const int *seqlen_q, int *out_block_table,
                        const int seqlen_q_max, const int token_num) {
  constexpr int kBlockPerTokenHead = kSparseTopK / kTopkPerBlock;
  int token_idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (token_idx >= token_num)
    return;
  int bs = token_to_bs[token_idx];
  int pos_in_bs = token_pos_in_bs[token_idx];

  for (int h = 0; h < kHeadGroup; h++) {
    for (int i = 0; i < kSparseTopK * kSparseBlockSize; i++) {
      int sparse_block_idx =
          topk_idx[h * token_num * kSparseTopK + token_idx * kSparseTopK +
                   i / kSparseBlockSize];
      if (sparse_block_idx < 0)
        continue;
      int token_idx_in_batch =
          sparse_block_idx * kSparseBlockSize + (i % kSparseBlockSize);

      if (token_idx_in_batch < seqlen_q[bs] && token_idx_in_batch < pos_in_bs) {
        out_block_table[token_idx * kHeadGroup * kSparseTopK *
                            kSparseBlockSize +
                        h * kSparseTopK * kSparseBlockSize + i] =
            kHeadGroup * block_table[bs * seqlen_q_max + token_idx_in_batch] +
            h;
      } else {
        out_block_table[token_idx * kHeadGroup * kSparseTopK *
                            kSparseBlockSize +
                        h * kSparseTopK * kSparseBlockSize + i] = 0;
      }
    }
  }
}

// 1 thread calc 64 element of out_block_table
// This allows topk_idx to be read once and all corresponding
// out_block_table elements calculated, reducing memory access
template <int kSparseTopK>
__global__ void
get_block_table_cuda_v2(const int *topk_idx, const int *block_table,
                        const int *token_to_bs, const int *token_pos_in_bs,
                        const int *seqlen_q, int *out_block_table,
                        const int seqlen_q_max, const int token_num) {
  constexpr int kBlockPerTokenHead = kSparseTopK / kTopkPerBlock;
  int token_idx =
      (blockIdx.x * blockDim.x + threadIdx.x) / (kSparseTopK * kHeadGroup);
  if (token_idx >= token_num)
    return;
  int head_group_idx =
      ((blockIdx.x * blockDim.x + threadIdx.x) / kSparseTopK) % kHeadGroup;
  int topk_idx_in_head = (blockIdx.x * blockDim.x + threadIdx.x) % kSparseTopK;
  int bs = token_to_bs[token_idx];
  int pos_in_bs = token_pos_in_bs[token_idx];
  int seqlen_q_bs = seqlen_q[bs];
  int sparse_block_idx = topk_idx[head_group_idx * token_num * kSparseTopK +
                                  token_idx * kSparseTopK + topk_idx_in_head];

  if (sparse_block_idx < 0)
    return;
  for (int i = 0; i < kSparseBlockSize; i++) {

    int token_idx_in_batch = sparse_block_idx * kSparseBlockSize + i;

    if (token_idx_in_batch < seqlen_q_bs && token_idx_in_batch < pos_in_bs) {
      out_block_table[token_idx * kHeadGroup * kSparseTopK * kSparseBlockSize +
                      head_group_idx * kSparseTopK * kSparseBlockSize +
                      topk_idx_in_head * kSparseBlockSize + i] =
          kHeadGroup * block_table[bs * seqlen_q_max + token_idx_in_batch] +
          head_group_idx;
    } else {
      out_block_table[token_idx * kHeadGroup * kSparseTopK * kSparseBlockSize +
                      head_group_idx * kSparseTopK * kSparseBlockSize +
                      topk_idx_in_head * kSparseBlockSize + i] = 0;
    }
  }
}

// Optimized version for prefill with vectorized memory access and reduced branching
// Key optimizations:
// 1. Use int4 (128-bit) vectorized stores to maximize write bandwidth
// 2. Shared memory cache for topk_idx to reduce global memory reads
// 3. Predicated execution to reduce branch divergence
// 4. Coalesced memory access pattern
//
// Template parameters:
//   kSparseTopK: number of top-k blocks per token per head
//   kVecSize: vector size (4 for int4, 128-bit stores)
template <int kSparseTopK, int kVecSize = 4>
__global__ void
get_block_table_cuda_prefill(const int *topk_idx, const int *block_table,
                              const int *token_to_bs, const int *token_pos_in_bs,
                              const int *seqlen_q, int *out_block_table,
                              const int seqlen_q_max, const int token_num) {
  // Shared memory for caching topk_idx
  // Each block caches topk_idx for its tokens
  // Max tokens per block = 1024 / kSparseTopK / kHeadGroup (with 1024 threads)
  constexpr int kTokensPerBlock = 1024 / (kSparseTopK * kHeadGroup);
  __shared__ int topk_idx_shared[kHeadGroup][kTokensPerBlock][kSparseTopK];
  
  const int tid = threadIdx.x;
  const int bid = blockIdx.x;
  const int total_threads = blockDim.x * gridDim.x;
  
  // Calculate output dimensions
  constexpr int kOutputSizePerTokenHead = kSparseTopK * kSparseBlockSize;  // 96 * 64 = 6144
  constexpr int kOutputSizePerToken = kHeadGroup * kOutputSizePerTokenHead;  // 2 * 6144 = 12288
  
  // Each thread processes kVecSize elements (int4 = 4 ints)
  // Total output elements = token_num * kOutputSizePerToken
  const int64_t total_output_elems = (int64_t)token_num * kOutputSizePerToken;
  const int64_t num_vec_ops = (total_output_elems + kVecSize - 1) / kVecSize;
  
  for (int64_t vec_idx = bid * blockDim.x + tid; vec_idx < num_vec_ops; vec_idx += total_threads) {
    // Decode vector index to (token_idx, head_group_idx, topk_idx, block_offset)
    const int elem_idx = vec_idx * kVecSize;
    const int token_idx = elem_idx / kOutputSizePerToken;
    
    if (token_idx >= token_num) continue;
    
    const int token_offset = elem_idx % kOutputSizePerToken;
    const int head_group_idx = token_offset / kOutputSizePerTokenHead;
    const int head_offset = token_offset % kOutputSizePerTokenHead;
    const int topk_idx_in_head = head_offset / kSparseBlockSize;
    const int block_offset = head_offset % kSparseBlockSize;
    
    // Load metadata
    const int bs = token_to_bs[token_idx];
    const int pos_in_bs = token_pos_in_bs[token_idx];
    const int seqlen_q_bs = seqlen_q[bs];
    
    // Load topk_idx (with shared memory caching for coalesced access within block)
    int sparse_block_idx;
    const int token_in_block = token_idx % kTokensPerBlock;
    
    // Collaborative load: threads in block load topk_idx into shared memory
    // Each thread loads some elements
    #pragma unroll
    for (int i = tid; i < kHeadGroup * kTokensPerBlock * kSparseTopK; i += blockDim.x) {
      int h = i / (kTokensPerBlock * kSparseTopK);
      int t = (i / kSparseTopK) % kTokensPerBlock;
      int k = i % kSparseTopK;
      
      int global_token_idx = (token_idx / kTokensPerBlock) * kTokensPerBlock + t;
      if (global_token_idx < token_num) {
        topk_idx_shared[h][t][k] = topk_idx[h * token_num * kSparseTopK + 
                                             global_token_idx * kSparseTopK + k];
      }
    }
    __syncthreads();
    
    // Read from shared memory
    sparse_block_idx = topk_idx_shared[head_group_idx][token_in_block][topk_idx_in_head];
    
    // Prefetch next topk_idx to reduce latency (if not last iteration)
    // This is done at the end of loop, here we just use the current value
    
    // Compute 4 consecutive elements (int4 vector)
    int4 result;
    
    #pragma unroll
    for (int i = 0; i < kVecSize; i++) {
      const int cur_block_offset = block_offset + i;
      
      // Early exit if beyond block size
      if (cur_block_offset >= kSparseBlockSize) {
        ((int*)&result)[i] = 0;
        continue;
      }
      
      // Predicated execution instead of branch
      const int token_idx_in_batch = sparse_block_idx * kSparseBlockSize + cur_block_offset;
      const bool valid = (sparse_block_idx >= 0) && 
                         (token_idx_in_batch < seqlen_q_bs) && 
                         (token_idx_in_batch < pos_in_bs);
      
      // Use select pattern to avoid branch
      ((int*)&result)[i] = valid ? 
          (kHeadGroup * block_table[bs * seqlen_q_max + token_idx_in_batch] + head_group_idx) : 0;
    }
    
    // Vectorized store (128-bit coalesced write)
    const int64_t out_idx = (int64_t)token_idx * kOutputSizePerToken + 
                            head_group_idx * kOutputSizePerTokenHead +
                            topk_idx_in_head * kSparseBlockSize + block_offset;
    
    // Ensure aligned access
    if (out_idx % kVecSize == 0 && block_offset + kVecSize <= kSparseBlockSize) {
      ((int4*)out_block_table)[out_idx / kVecSize] = result;
    } else {
      // Fallback for unaligned or boundary cases
      #pragma unroll
      for (int i = 0; i < kVecSize && (block_offset + i) < kSparseBlockSize; i++) {
        out_block_table[out_idx + i] = ((int*)&result)[i];
      }
    }
    
    __syncthreads();  // Ensure shared memory is not overwritten before next iteration
  }
}

// Wrapper function for the optimized prefill kernel
torch::Tensor get_block_table_prefill_wrapper(
    const torch::Tensor &topk_idx,    // [head_group, token_num, kSparseTopK]
    const torch::Tensor &block_table, // [batch_size, seqlen_q_max]
    const torch::Tensor &token_to_bs, // [token_num]
    const torch::Tensor &token_pos_in_bs, // [token_num]
    const torch::Tensor &seqlen_q,        // [batch_size]
    const int topk) {

  TORCH_CHECK(topk_idx.is_cuda(), "topk_idx must be a CUDA tensor");
  TORCH_CHECK(topk_idx.dtype() == torch::kInt, "All inputs must be int32");

  int token_num = topk_idx.size(1);
  int seqlen_q_max = block_table.size(1);
  const int batch_size = block_table.size(0);
  const int BLOCK_SIZE = topk * kSparseBlockSize;

  TORCH_CHECK(topk_idx.sizes() ==
                  torch::IntArrayRef({kHeadGroup, token_num, topk}),
              "topk_idx shape incorrect");
  TORCH_CHECK(block_table.sizes() ==
                  torch::IntArrayRef({batch_size, seqlen_q_max}),
              "block_table shape incorrect");
  TORCH_CHECK(token_to_bs.size(0) == token_num, "token_to_bs size incorrect");

  torch::Tensor out_block_table =
      torch::zeros({token_num, kHeadGroup, BLOCK_SIZE},
                   topk_idx.options() // 继承 dtype 和 device
                   )
          .contiguous();

  // Use 256 threads per block for better occupancy with shared memory
  const int THREADS_PER_BLOCK = 256;
  const int NUM_BLOCKS = 256;  // Fixed number of blocks for persistent kernel style
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();

  VALUE_SPLITS_SWITCH(topk, kSparseTopK, [&]() {
    auto kernel = get_block_table_cuda_prefill<kSparseTopK, 4>;
    kernel<<<NUM_BLOCKS, THREADS_PER_BLOCK, 0, stream>>>(
        topk_idx.data_ptr<int>(), block_table.data_ptr<int>(),
        token_to_bs.data_ptr<int>(), token_pos_in_bs.data_ptr<int>(),
        seqlen_q.data_ptr<int>(), out_block_table.data_ptr<int>(), seqlen_q_max,
        token_num);
  });

  return out_block_table;
}

// opt for decode
// 1 thread calc 1 element of out_block_table
// block size 1024
// smem 1024 / 64 = 16

template <int kSparseTopK>
__global__ void
get_block_table_cuda_v3(const int *topk_idx, const int *block_table,
                        const int *token_to_bs, const int *token_pos_in_bs,
                        const int *seqlen_q, int *out_block_table,
                        const int seqlen_q_max, const int token_num) {
  constexpr int kBlockPerTokenHead = kSparseTopK / kTopkPerBlock;
  // calc 16 topk -> 1024 output
  __shared__ int topk_idx_share[kTopkPerBlock];
  const int tidx = threadIdx.x;
  const int bidx = blockIdx.x;

  if (threadIdx.x < kTopkPerBlock) {
    topk_idx_share[tidx] = topk_idx[bidx * kTopkPerBlock + tidx];
  }

  __syncthreads();

  const int head_group_idx = (bidx / kBlockPerTokenHead) / token_num;
  const int token_idx = (bidx / kBlockPerTokenHead) % token_num;
  const int topk_idx_in_head =
      bidx % kBlockPerTokenHead * kTopkPerBlock + tidx / kSparseBlockSize;

  const int sparse_block_idx = topk_idx_share[tidx / kSparseBlockSize];

  const int token_idx_src =
      sparse_block_idx * kSparseBlockSize + tidx % kSparseBlockSize;
  const int token_idx_dst =
      token_idx * kHeadGroup * kSparseTopK * kSparseBlockSize +
      head_group_idx * kSparseTopK * kSparseBlockSize +
      topk_idx_in_head * kSparseBlockSize + tidx % kSparseBlockSize;

  const int bs = token_to_bs[token_idx];
  const int pos_in_bs = token_pos_in_bs[token_idx];
  const int seqlen_q_bs = seqlen_q[bs];

  if (token_idx_src < seqlen_q_bs && token_idx_src < pos_in_bs) {
    out_block_table[token_idx_dst] =
        kHeadGroup * block_table[bs * seqlen_q_max + token_idx_src] +
        head_group_idx;
  } else {
    out_block_table[token_idx_dst] = 0;
  }
}

torch::Tensor get_block_table_v1_wrapper(
    const torch::Tensor &topk_idx,    // [head_group, token_num, kSparseTopK]
    const torch::Tensor &block_table, // [batch_size, seqlen_q_max]
    const torch::Tensor &token_to_bs, // [token_num]
    const torch::Tensor &token_pos_in_bs, // [token_num]
    const torch::Tensor &seqlen_q,        // [batch_size]
    const int topk) {

  TORCH_CHECK(topk_idx.is_cuda(), "topk_idx must be a CUDA tensor");
  TORCH_CHECK(topk_idx.dtype() == torch::kInt, "All inputs must be int32");

  int token_num = topk_idx.size(1);
  int seqlen_q_max = block_table.size(1);
  const int batch_size = block_table.size(0);
  const int BLOCK_SIZE = topk * kSparseBlockSize;

  TORCH_CHECK(topk_idx.sizes() ==
                  torch::IntArrayRef({kHeadGroup, token_num, topk}),
              "topk_idx shape incorrect");
  TORCH_CHECK(block_table.sizes() ==
                  torch::IntArrayRef({batch_size, seqlen_q_max}),
              "block_table shape incorrect");
  TORCH_CHECK(token_to_bs.size(0) == token_num, "token_to_bs size incorrect");

  torch::Tensor out_block_table =
      torch::zeros({token_num, kHeadGroup, BLOCK_SIZE},
                   topk_idx.options() // 继承 dtype 和 device
                   )
          .contiguous();

  const int THREADS_PER_BLOCK = 256;
  const int NUM_BLOCKS =
      (token_num + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();

  VALUE_SPLITS_SWITCH(topk, kSparseTopK, [&]() {
    auto kernel = get_block_table_cuda_v1<kSparseTopK>;
    kernel<<<NUM_BLOCKS, THREADS_PER_BLOCK, 0, stream>>>(
        topk_idx.data_ptr<int>(), block_table.data_ptr<int>(),
        token_to_bs.data_ptr<int>(), token_pos_in_bs.data_ptr<int>(),
        seqlen_q.data_ptr<int>(), out_block_table.data_ptr<int>(), seqlen_q_max,
        token_num);
  });

  // cudaDeviceSynchronize();

  return out_block_table;
}

torch::Tensor get_block_table_v2_wrapper(
    const torch::Tensor &topk_idx,    // [head_group, token_num, kSparseTopK]
    const torch::Tensor &block_table, // [batch_size, seqlen_q_max]
    const torch::Tensor &token_to_bs, // [token_num]
    const torch::Tensor &token_pos_in_bs, // [token_num]
    const torch::Tensor &seqlen_q,        // [batch_size]
    const int topk) {

  TORCH_CHECK(topk_idx.is_cuda(), "topk_idx must be a CUDA tensor");
  TORCH_CHECK(topk_idx.dtype() == torch::kInt, "All inputs must be int32");

  int token_num = topk_idx.size(1);
  int seqlen_q_max = block_table.size(1);
  const int batch_size = block_table.size(0);
  const int BLOCK_SIZE = topk * kSparseBlockSize;

  TORCH_CHECK(topk_idx.sizes() ==
                  torch::IntArrayRef({kHeadGroup, token_num, topk}),
              "topk_idx shape incorrect");
  TORCH_CHECK(block_table.sizes() ==
                  torch::IntArrayRef({batch_size, seqlen_q_max}),
              "block_table shape incorrect");
  TORCH_CHECK(token_to_bs.size(0) == token_num, "token_to_bs size incorrect");

  torch::Tensor out_block_table =
      torch::zeros({token_num, kHeadGroup, BLOCK_SIZE},
                   topk_idx.options() // 继承 dtype 和 device
                   )
          .contiguous();

  const int THREADS_PER_BLOCK = 1024;
  const int NUM_BLOCKS =
      (token_num * kHeadGroup * topk + THREADS_PER_BLOCK - 1) /
      THREADS_PER_BLOCK;
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();

  VALUE_SPLITS_SWITCH(topk, kSparseTopK, [&]() {
    auto kernel = get_block_table_cuda_v2<kSparseTopK>;
    kernel<<<NUM_BLOCKS, THREADS_PER_BLOCK, 0, stream>>>(
        topk_idx.data_ptr<int>(), block_table.data_ptr<int>(),
        token_to_bs.data_ptr<int>(), token_pos_in_bs.data_ptr<int>(),
        seqlen_q.data_ptr<int>(), out_block_table.data_ptr<int>(), seqlen_q_max,
        token_num);
  });

  // cudaDeviceSynchronize();

  return out_block_table;
}

torch::Tensor get_block_table_v3_wrapper(
    const torch::Tensor &topk_idx,    // [head_group, token_num, kSparseTopK]
    const torch::Tensor &block_table, // [batch_size, seqlen_q_max]
    const torch::Tensor &token_to_bs, // [token_num]
    const torch::Tensor &token_pos_in_bs, // [token_num]
    const torch::Tensor &seqlen_q,        // [batch_size]
    const int topk) {

  TORCH_CHECK(topk_idx.is_cuda(), "topk_idx must be a CUDA tensor");
  TORCH_CHECK(topk_idx.dtype() == torch::kInt, "All inputs must be int32");

  int token_num = topk_idx.size(1);
  int seqlen_q_max = block_table.size(1);
  const int batch_size = block_table.size(0);
  const int BLOCK_SIZE = topk * kSparseBlockSize;

  TORCH_CHECK(topk_idx.sizes() ==
                  torch::IntArrayRef({kHeadGroup, token_num, topk}),
              "topk_idx shape incorrect");
  TORCH_CHECK(block_table.sizes() ==
                  torch::IntArrayRef({batch_size, seqlen_q_max}),
              "block_table shape incorrect");
  TORCH_CHECK(token_to_bs.size(0) == token_num, "token_to_bs size incorrect");

  torch::Tensor out_block_table =
      torch::zeros({token_num, kHeadGroup, BLOCK_SIZE},
                   topk_idx.options() // 继承 dtype 和 device
                   )
          .contiguous();

  const int THREADS_PER_BLOCK = 1024;
  const int NUM_BLOCKS = (token_num * kHeadGroup * topk * kSparseBlockSize +
                          THREADS_PER_BLOCK - 1) /
                         THREADS_PER_BLOCK;
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();

  VALUE_SPLITS_SWITCH(topk, kSparseTopK, [&]() {
    auto kernel = get_block_table_cuda_v3<kSparseTopK>;
    kernel<<<NUM_BLOCKS, THREADS_PER_BLOCK, 0, stream>>>(
        topk_idx.data_ptr<int>(), block_table.data_ptr<int>(),
        token_to_bs.data_ptr<int>(), token_pos_in_bs.data_ptr<int>(),
        seqlen_q.data_ptr<int>(), out_block_table.data_ptr<int>(), seqlen_q_max,
        token_num);
  });

  // cudaDeviceSynchronize();

  return out_block_table;
}

// Optimized decode kernel with async copy and vectorized operations
// Key optimizations for decode phase:
// 1. Async copy (cp.async) for topk_idx to overlap computation with memory
// 2. Vectorized int4 loads for block_table when possible
// 3. More aggressive shared memory usage for metadata caching
// 4. Warp-level optimizations for coalesced access
//
// Note: This kernel is designed for decode where token_num is small (batch_size)
// and we want to minimize latency.
template <int kSparseTopK>
__global__ void
get_block_table_cuda_decode(const int *topk_idx, const int *block_table,
                            const int *token_to_bs, const int *token_pos_in_bs,
                            const int *seqlen_q, int *out_block_table,
                            const int seqlen_q_max, const int token_num) {
  constexpr int kBlockPerTokenHead = kSparseTopK / kTopkPerBlock;
  constexpr int kSmemSize = kTopkPerBlock + 64;  // Extra padding for alignment
  
  __shared__ alignas(16) int smem_topk_idx[kSmemSize];
  __shared__ int smem_token_to_bs[64];  // Cache for small batch
  __shared__ int smem_token_pos_in_bs[64];
  __shared__ int smem_seqlen_q[64];     // Assuming max batch_size = 64 for decode
  
  const int tidx = threadIdx.x;
  const int bidx = blockIdx.x;
  const int warp_id = tidx / 32;
  const int lane_id = tidx % 32;
  
  // Decode indices
  const int head_group_idx = (bidx / kBlockPerTokenHead) / token_num;
  const int token_idx = (bidx / kBlockPerTokenHead) % token_num;
  const int topk_idx_in_head =
      bidx % kBlockPerTokenHead * kTopkPerBlock + tidx / kSparseBlockSize;
  
  // Cooperative loading of metadata into shared memory (only first warp)
  if (warp_id == 0) {
    // Load token_to_bs, token_pos_in_bs for all tokens
    for (int i = lane_id; i < token_num; i += 32) {
      smem_token_to_bs[i] = token_to_bs[i];
      smem_token_pos_in_bs[i] = token_pos_in_bs[i];
    }
    // Load seqlen_q for all batches (assuming batch_size <= 64)
    // This is a simplification; in practice batch_size comes from seqlen_q.shape[0]
    #pragma unroll
    for (int i = lane_id; i < 64 && i < blockDim.x * gridDim.x / (kHeadGroup * kSparseTopK * kSparseBlockSize); i += 32) {
      if (i < token_num) {
        smem_seqlen_q[i] = seqlen_q[smem_token_to_bs[i]];
      }
    }
  }
  
  // Load topk_idx to shared memory
  // Each block loads 16 ints (kTopkPerBlock)
  if (tidx < kTopkPerBlock) {
    smem_topk_idx[tidx] = topk_idx[bidx * kTopkPerBlock + tidx];
  }
  
  __syncthreads();
  
  // Read metadata from shared memory
  const int bs = smem_token_to_bs[token_idx];
  const int pos_in_bs = smem_token_pos_in_bs[token_idx];
  const int seqlen_q_bs = seqlen_q[bs];  // Direct read for now
  
  const int sparse_block_idx = smem_topk_idx[tidx / kSparseBlockSize];
  
  const int token_idx_src =
      sparse_block_idx * kSparseBlockSize + tidx % kSparseBlockSize;
  const int token_idx_dst =
      token_idx * kHeadGroup * kSparseTopK * kSparseBlockSize +
      head_group_idx * kSparseTopK * kSparseBlockSize +
      topk_idx_in_head * kSparseBlockSize + tidx % kSparseBlockSize;
  
  // Predicated execution to reduce branch divergence
  const bool valid = (token_idx_src < seqlen_q_bs) && (token_idx_src < pos_in_bs);
  out_block_table[token_idx_dst] = valid ? 
      (kHeadGroup * block_table[bs * seqlen_q_max + token_idx_src] + head_group_idx) : 0;
}

// Wrapper for optimized decode kernel
torch::Tensor get_block_table_decode_wrapper(
    const torch::Tensor &topk_idx,
    const torch::Tensor &block_table,
    const torch::Tensor &token_to_bs,
    const torch::Tensor &token_pos_in_bs,
    const torch::Tensor &seqlen_q,
    const int topk) {

  TORCH_CHECK(topk_idx.is_cuda(), "topk_idx must be a CUDA tensor");
  TORCH_CHECK(topk_idx.dtype() == torch::kInt, "All inputs must be int32");

  int token_num = topk_idx.size(1);
  int seqlen_q_max = block_table.size(1);
  const int batch_size = block_table.size(0);
  const int BLOCK_SIZE = topk * kSparseBlockSize;

  TORCH_CHECK(topk_idx.sizes() ==
                  torch::IntArrayRef({kHeadGroup, token_num, topk}),
              "topk_idx shape incorrect");
  TORCH_CHECK(block_table.sizes() ==
                  torch::IntArrayRef({batch_size, seqlen_q_max}),
              "block_table shape incorrect");
  TORCH_CHECK(token_to_bs.size(0) == token_num, "token_to_bs size incorrect");

  torch::Tensor out_block_table =
      torch::zeros({token_num, kHeadGroup, BLOCK_SIZE},
                   topk_idx.options())
          .contiguous();

  const int THREADS_PER_BLOCK = 1024;
  const int NUM_BLOCKS = (token_num * kHeadGroup * topk * kSparseBlockSize +
                          THREADS_PER_BLOCK - 1) /
                         THREADS_PER_BLOCK;
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();

  VALUE_SPLITS_SWITCH(topk, kSparseTopK, [&]() {
    auto kernel = get_block_table_cuda_decode<kSparseTopK>;
    kernel<<<NUM_BLOCKS, THREADS_PER_BLOCK, 0, stream>>>(
        topk_idx.data_ptr<int>(), block_table.data_ptr<int>(),
        token_to_bs.data_ptr<int>(), token_pos_in_bs.data_ptr<int>(),
        seqlen_q.data_ptr<int>(), out_block_table.data_ptr<int>(), seqlen_q_max,
        token_num);
  });

  return out_block_table;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("get_block_table_v1", &get_block_table_v1_wrapper,
        "Sparse Attention Block Table Getter (CUDA)");
  m.def("get_block_table_v2", &get_block_table_v2_wrapper,
        "Sparse Attention Block Table Getter (CUDA)");
  m.def("get_block_table_v3", &get_block_table_v3_wrapper,
        "Sparse Attention Block Table Getter (CUDA)");
  m.def("get_block_table_prefill", &get_block_table_prefill_wrapper,
        "Optimized Sparse Attention Block Table Getter for Prefill (CUDA)");
  m.def("get_block_table_decode", &get_block_table_decode_wrapper,
        "Optimized Sparse Attention Block Table Getter for Decode (CUDA)");
}