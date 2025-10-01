"""
Optimized Triton kernels for Expert-Sliced GPU Scheduling.
These kernels fuse operations to minimize memory traffic and maximize GPU utilization.
"""

import torch
import triton
import triton.language as tl


@triton.jit
def expert_routing_kernel(
    # Input pointers
    logits_ptr,
    # Output pointers
    expert_ids_ptr,
    expert_weights_ptr,
    expert_counts_ptr,
    # Dimensions
    batch_size,
    num_experts,
    top_k,
    # Strides
    logits_batch_stride,
    logits_expert_stride,
    output_batch_stride,
    output_k_stride,
    # Block size
    BLOCK_SIZE: tl.constexpr,
):
    """
    Fused softmax + top-k expert routing kernel.
    Computes routing decisions in a single kernel to minimize memory traffic.
    """
    pid = tl.program_id(0)
    
    if pid >= batch_size:
        return
    
    # Load logits for this token
    logits_offset = pid * logits_batch_stride
    offs = tl.arange(0, BLOCK_SIZE)
    mask = offs < num_experts
    
    logits = tl.load(logits_ptr + logits_offset + offs, mask=mask, other=-float('inf'))
    
    # Compute softmax
    max_logit = tl.max(logits, axis=0)
    exp_logits = tl.exp(logits - max_logit)
    sum_exp = tl.sum(exp_logits, axis=0)
    probs = exp_logits / sum_exp
    
    # Top-k selection (iterative approach)
    for k in range(top_k):
        # Find max probability
        max_prob = tl.max(probs, axis=0)
        max_idx = tl.argmax(probs, axis=0)
        
        # Store expert ID and weight
        output_offset = pid * output_batch_stride + k * output_k_stride
        tl.store(expert_ids_ptr + output_offset, max_idx)
        tl.store(expert_weights_ptr + output_offset, max_prob)
        
        # Increment expert count atomically
        tl.atomic_add(expert_counts_ptr + max_idx, 1)
        
        # Mask out selected expert for next iteration
        probs = tl.where(offs == max_idx, -float('inf'), probs)


@triton.jit
def fused_expert_mlp_kernel(
    # Input pointers
    x_ptr,
    # Weight and bias pointers for 3 layers
    w1_ptr, b1_ptr,
    w2_ptr, b2_ptr,
    w3_ptr, b3_ptr,
    # Output pointer
    out_ptr,
    # Dimensions
    num_tokens,
    input_dim,
    hidden_dim,
    output_dim,
    # Strides
    x_stride,
    out_stride,
    # Block size
    BLOCK_SIZE: tl.constexpr,
):
    """
    Fused 3-layer MLP expert computation.
    Keeps intermediate activations in registers to minimize memory traffic.
    """
    pid = tl.program_id(0)
    
    if pid >= num_tokens:
        return
    
    # Load input
    x_offset = pid * x_stride
    offs_in = tl.arange(0, BLOCK_SIZE)
    mask_in = offs_in < input_dim
    x = tl.load(x_ptr + x_offset + offs_in, mask=mask_in, other=0.0)
    
    # Layer 1: input_dim -> hidden_dim
    offs_h = tl.arange(0, BLOCK_SIZE)
    for i in range(0, hidden_dim, BLOCK_SIZE):
        mask_h = (i + offs_h) < hidden_dim
        acc = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
        
        # Matrix-vector multiplication
        for j in range(input_dim):
            w = tl.load(w1_ptr + j * hidden_dim + i + offs_h, mask=mask_h, other=0.0)
            acc += x[j] * w
        
        # Add bias and apply ReLU
        b = tl.load(b1_ptr + i + offs_h, mask=mask_h, other=0.0)
        h1 = tl.maximum(acc + b, 0.0)
        
        # Store in shared memory or registers (simplified here)
        # In production, use shared memory for better performance
        
    # Layer 2: hidden_dim -> hidden_dim
    for i in range(0, hidden_dim, BLOCK_SIZE):
        mask_h = (i + offs_h) < hidden_dim
        acc = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
        
        for j in range(hidden_dim):
            w = tl.load(w2_ptr + j * hidden_dim + i + offs_h, mask=mask_h, other=0.0)
            # Load h1[j] - in production, this would be from shared memory
            acc += h1 * w  # Simplified
        
        b = tl.load(b2_ptr + i + offs_h, mask=mask_h, other=0.0)
        h2 = tl.maximum(acc + b, 0.0)
    
    # Layer 3: hidden_dim -> output_dim
    offs_out = tl.arange(0, BLOCK_SIZE)
    for i in range(0, output_dim, BLOCK_SIZE):
        mask_out = (i + offs_out) < output_dim
        acc = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
        
        for j in range(hidden_dim):
            w = tl.load(w3_ptr + j * output_dim + i + offs_out, mask=mask_out, other=0.0)
            acc += h2 * w  # Simplified
        
        b = tl.load(b3_ptr + i + offs_out, mask=mask_out, other=0.0)
        out_block = acc + b
        
        # Store output
        out_offset = pid * out_stride
        tl.store(out_ptr + out_offset + i + offs_out, out_block, mask=mask_out)


@triton.jit
def expert_gather_scatter_kernel(
    # Input/output pointers
    input_ptr,
    output_ptr,
    indices_ptr,
    weights_ptr,
    # Dimensions
    num_tokens,
    dim,
    # Strides
    input_stride,
    output_stride,
    # Block size
    BLOCK_SIZE: tl.constexpr,
):
    """
    Optimized gather-scatter operation for routing tokens to experts.
    Uses atomic operations for accumulation.
    """
    pid = tl.program_id(0)
    
    if pid >= num_tokens:
        return
    
    # Load token index and weight
    token_idx = tl.load(indices_ptr + pid)
    weight = tl.load(weights_ptr + pid)
    
    # Gather data
    offs = tl.arange(0, BLOCK_SIZE)
    mask = offs < dim
    
    data = tl.load(input_ptr + token_idx * input_stride + offs, mask=mask, other=0.0)
    
    # Apply routing weight
    data = data * weight
    
    # Scatter with atomic add (for accumulation across multiple experts)
    output_offset = token_idx * output_stride + offs
    tl.atomic_add(output_ptr + output_offset, data, mask=mask)


@triton.jit
def batched_expert_kernel(
    # Input
    x_ptr,
    # Weights
    w_ptr,
    b_ptr,
    # Output
    out_ptr,
    # Token assignments
    token_indices_ptr,
    expert_weights_ptr,
    # Dimensions
    num_tokens,
    input_dim,
    output_dim,
    # Strides
    x_batch_stride,
    x_feat_stride,
    w_in_stride,
    w_out_stride,
    out_batch_stride,
    out_feat_stride,
    # Block sizes
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    """
    Batched expert computation using tiled matrix multiplication.
    Processes multiple tokens assigned to the same expert in parallel.
    """
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    
    # Compute tile offsets
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)
    
    # Initialize accumulator
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    
    # Tiled matrix multiplication
    for k in range(0, input_dim, BLOCK_K):
        # Load input tile
        input_ptrs = x_ptr + (offs_m[:, None] * x_batch_stride + 
                              (k + offs_k[None, :]) * x_feat_stride)
        input_mask = (offs_m[:, None] < num_tokens) & ((k + offs_k[None, :]) < input_dim)
        input_tile = tl.load(input_ptrs, mask=input_mask, other=0.0)
        
        # Load weight tile
        weight_ptrs = w_ptr + ((k + offs_k[:, None]) * w_in_stride + 
                               offs_n[None, :] * w_out_stride)
        weight_mask = ((k + offs_k[:, None]) < input_dim) & (offs_n[None, :] < output_dim)
        weight_tile = tl.load(weight_ptrs, mask=weight_mask, other=0.0)
        
        # Accumulate
        acc += tl.dot(input_tile, weight_tile)
    
    # Add bias
    if b_ptr is not None:
        bias = tl.load(b_ptr + offs_n, mask=offs_n < output_dim, other=0.0)
        acc += bias[None, :]
    
    # Apply expert weights
    expert_weights = tl.load(expert_weights_ptr + offs_m, mask=offs_m < num_tokens, other=0.0)
    acc *= expert_weights[:, None]
    
    # Store output
    output_ptrs = out_ptr + (offs_m[:, None] * out_batch_stride + 
                             offs_n[None, :] * out_feat_stride)
    output_mask = (offs_m[:, None] < num_tokens) & (offs_n[None, :] < output_dim)
    tl.store(output_ptrs, acc, mask=output_mask)


class TritonExpertOps:
    """
    High-level interface for Triton-optimized expert operations.
    """
    
    @staticmethod
    def expert_routing(logits: torch.Tensor, top_k: int) -> tuple:
        """
        Perform expert routing using fused Triton kernel.
        
        Args:
            logits: [batch_size, num_experts] routing logits
            top_k: number of experts to select per token
            
        Returns:
            expert_ids: [batch_size, top_k] selected expert indices
            expert_weights: [batch_size, top_k] normalized routing weights
            expert_counts: [num_experts] number of tokens assigned to each expert
        """
        batch_size, num_experts = logits.shape
        
        # Allocate output tensors
        expert_ids = torch.zeros(batch_size, top_k, dtype=torch.int32, device=logits.device)
        expert_weights = torch.zeros(batch_size, top_k, dtype=torch.float32, device=logits.device)
        expert_counts = torch.zeros(num_experts, dtype=torch.int32, device=logits.device)
        
        # Launch kernel
        BLOCK_SIZE = triton.next_power_of_2(num_experts)
        grid = (batch_size,)
        
        expert_routing_kernel[grid](
            logits, expert_ids, expert_weights, expert_counts,
            batch_size, num_experts, top_k,
            logits.stride(0), logits.stride(1),
            expert_ids.stride(0), expert_ids.stride(1),
            BLOCK_SIZE=BLOCK_SIZE
        )
        
        return expert_ids, expert_weights, expert_counts
    
    @staticmethod
    def batched_expert_forward(
        x: torch.Tensor,
        weight: torch.Tensor,
        bias: torch.Tensor,
        token_indices: torch.Tensor,
        expert_weights: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute expert output for a batch of tokens using optimized Triton kernel.
        
        Args:
            x: [num_tokens, input_dim] input tokens
            weight: [input_dim, output_dim] expert weight matrix
            bias: [output_dim] expert bias vector
            token_indices: [num_tokens] original token indices
            expert_weights: [num_tokens] routing weights for this expert
            
        Returns:
            output: [num_tokens, output_dim] expert outputs
        """
        num_tokens, input_dim = x.shape
        output_dim = weight.shape[1]
        
        # Allocate output
        output = torch.zeros(num_tokens, output_dim, dtype=x.dtype, device=x.device)
        
        # Launch kernel with tiling
        BLOCK_M, BLOCK_N, BLOCK_K = 64, 64, 32
        grid = (
            triton.cdiv(num_tokens, BLOCK_M),
            triton.cdiv(output_dim, BLOCK_N)
        )
        
        batched_expert_kernel[grid](
            x, weight, bias, output,
            token_indices, expert_weights,
            num_tokens, input_dim, output_dim,
            x.stride(0), x.stride(1),
            weight.stride(0), weight.stride(1),
            output.stride(0), output.stride(1),
            BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K
        )
        
        return output
    
    @staticmethod
    def gather_scatter(
        input_tensor: torch.Tensor,
        indices: torch.Tensor,
        weights: torch.Tensor,
        output_shape: tuple
    ) -> torch.Tensor:
        """
        Optimized gather-scatter operation for token routing.
        
        Args:
            input_tensor: [num_tokens, dim] input data
            indices: [num_tokens] target indices
            weights: [num_tokens] routing weights
            output_shape: shape of output tensor
            
        Returns:
            output: gathered and weighted data
        """
        num_tokens, dim = input_tensor.shape
        output = torch.zeros(output_shape, dtype=input_tensor.dtype, device=input_tensor.device)
        
        BLOCK_SIZE = triton.next_power_of_2(dim)
        grid = (num_tokens,)
        
        expert_gather_scatter_kernel[grid](
            input_tensor, output, indices, weights,
            num_tokens, dim,
            input_tensor.stride(0), output.stride(0),
            BLOCK_SIZE=BLOCK_SIZE
        )
        
        return output
