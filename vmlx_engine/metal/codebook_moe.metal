//
// codebook_moe.metal
// Fused codebook lookup + matvec kernel for MoE expert selection
//
// Key insight: We NEVER materialize the full weight matrix.
// Instead, for each selected expert, we:
//   1. Look up codebook entries using the expert's indices
//   2. Compute dot product with input x
//   3. All in a single GPU pass
//
// This avoids loading 512 experts × 3 projections × 60 layers
// which would be ~720GB of intermediate data.
//
// Data layout:
//   x:             [batch, in_dim]     where in_dim = n_groups * group_size
//   codebook:      [n_codes, group_size] float16
//   indices:       [n_experts, out_dim, n_groups] uint16
//   expert_ids:    [k] uint16 (selected expert indices, top-8 of 512)
//   output:        [batch, k, out_dim] float16 (accumulated)
//
// Algorithm (per thread):
//   1. Get (batch_idx, expert_idx) from thread position
//   2. Look up the actual expert ID from expert_ids[expert_idx]
//   3. For each output dimension:
//      a. Load n_groups indices
//      b. Look up n_groups codebook entries
//      c. Compute partial dot product with x[batch_idx]
//   4. Accumulate result

#include <metal_stdlib>
using namespace metal;

#define MAX_GROUP_SIZE 32
#define MAX_K 16

kernel void codebook_moe_matvec(
    // Buffer 0: x [batch, in_dim] half
    device const half* x [[buffer(0)]],
    // Buffer 1: codebook [n_codes, group_size] half
    device const half* codebook [[buffer(1)]],
    // Buffer 2: indices [n_experts, out_dim, n_groups] uint16
    device const uint16* indices [[buffer(2)]],
    // Buffer 3: selected expert IDs [k] uint16
    device const uint16* expert_ids [[buffer(3)]],
    // Buffer 4: output [batch, k, out_dim] half
    device half* output [[buffer(4)]],
    // Buffer 5: metadata
    constant uint32_t& batch_size [[buffer(5)]],
    constant uint32_t& k [[buffer(6)]],              // num selected experts
    constant uint32_t& out_dim [[buffer(7)]],
    constant uint32_t& n_groups [[buffer(8)]],
    constant uint32_t& group_size [[buffer(9)]],
    constant uint32_t& in_dim [[buffer(10)]],
    // Thread positioning
    uint tid [[thread_position_in_grid]],
    uint bid [[threadgroup_position_in_grid]]
) {
    // Each thread handles one (batch, expert, out_dim) element
    // Grid: batch_size * k * out_dim threads
    
    uint total_threads = batch_size * k * out_dim;
    if (tid >= total_threads) return;
    
    // Unpack thread index
    uint b = tid / (k * out_dim);
    uint expert_local = (tid % (k * out_dim)) / out_dim;
    uint out_d = tid % out_dim;
    
    if (b >= batch_size || expert_local >= k || out_d >= out_dim) return;
    
    // Get actual expert ID from routing
    uint expert_id = expert_ids[expert_local];
    
    // Pointer to this expert's indices: indices[expert_id, out_d, :]
    uint expert_base = expert_id * out_dim * n_groups;
    uint row_base = expert_base + out_d * n_groups;
    
    // Compute dot product: x[b, :] · W_expert[out_d, :]
    // W is reconstructed as: for each group g, lookup codebook[indices[expert_id, out_d, g]]]
    half sum = 0.0h;
    
    for (uint g = 0; g < n_groups; g++) {
        // Get codebook index for this group
        uint idx = indices[row_base + g];
        
        // Codebook entry base: idx * group_size
        uint cb_base = idx * group_size;
        
        // Dot product of x[b, g*group_size : (g+1)*group_size] with codebook[cb_base : cb_base+group_size]
        uint x_base = b * in_dim + g * group_size;
        
        for (uint i = 0; i < group_size; i++) {
            sum += x[x_base + i] * codebook[cb_base + i];
        }
    }
    
    // Store result
    uint out_idx = (b * k + expert_local) * out_dim + out_d;
    output[out_idx] = sum;
}

//
// Batched version with threadgroup cooperation for better memory coalescing
// Each threadgroup handles one batch element's k experts
//
kernel void codebook_moe_batched(
    device const half* x [[buffer(0)]],
    device const half* codebook [[buffer(1)]],
    device const uint16* indices [[buffer(2)]],
    device const uint16* expert_ids [[buffer(3)]],
    device half* output [[buffer(4)]],
    constant uint32_t& batch_size [[buffer(5)]],
    constant uint32_t& k [[buffer(6)]],
    constant uint32_t& out_dim [[buffer(7)]],
    constant uint32_t& n_groups [[buffer(8)]],
    constant uint32_t& group_size [[buffer(9)]],
    constant uint32_t& in_dim [[buffer(10)]],
    uint tid [[thread_position_in_grid]],
    uint gid [[threadgroup_position_in_grid]]
) {
    // Grid: batch_size * k threadgroups
    // Each threadgroup: out_dim threads
    
    uint expert_threads = k * batch_size;
    uint b = gid / k;
    uint expert_local = gid % k;
    
    if (b >= batch_size || expert_local >= k) return;
    
    uint expert_id = expert_ids[expert_local];
    uint x_base = b * in_dim;
    
    // Threadgroup shared memory for codebook entries
    threadgroup half shared_cb[MAX_K * MAX_GROUP_SIZE];
    
    // Each thread handles one output dimension
    uint out_d = tid;
    if (out_d >= out_dim) return;
    
    // Pre-load this expert's indices into threadgroup memory (cooperative)
    uint expert_base = expert_id * out_dim * n_groups;
    uint row_base = expert_base + out_d * n_groups;
    
    for (uint g = 0; g < n_groups; g++) {
        uint idx = indices[row_base + g];
        uint cb_offset = idx * group_size;
        
        // Load codebook entry
        uint shared_offset = expert_local * MAX_GROUP_SIZE;
        for (uint i = 0; i < group_size; i++) {
            shared_cb[shared_offset + i] = codebook[cb_offset + i];
        }
        
        // Compute dot product
        half sum = 0.0h;
        uint x_idx = x_base + g * group_size;
        
        for (uint i = 0; i < group_size; i++) {
            sum += x[x_idx + i] * shared_cb[shared_offset + i];
        }
        
        // Store result
        uint out_idx = (b * k + expert_local) * out_dim + out_d;
        output[out_idx] = sum;
    }
}

//
// Reference 2D kernel (for non-MoE codebook models)
// Preserved for compatibility with simple codebook VQ
//
kernel void codebook_matvec_2d(
    device const half* x [[buffer(0)]],
    device const half* codebook [[buffer(1)]],
    device const uint16* indices [[buffer(2)]],
    device half* output [[buffer(3)]],
    constant uint32_t& batch_size [[buffer(4)]],
    constant uint32_t& out_dim [[buffer(5)]],
    constant uint32_t& n_groups [[buffer(6)]],
    constant uint32_t& group_size [[buffer(7)]],
    constant uint32_t& in_dim [[buffer(8)]],
    uint tid [[thread_position_in_grid]]
) {
    // Each thread handles one (batch, out_dim) element
    uint total = batch_size * out_dim;
    if (tid >= total) return;
    
    uint b = tid / out_dim;
    uint od = tid % out_dim;
    
    // Index into this output row: od * n_groups
    uint idx_base = od * n_groups;
    
    half sum = 0.0h;
    
    for (uint g = 0; g < n_groups; g++) {
        uint idx = indices[idx_base + g];
        uint cb_base = idx * group_size;
        
        uint x_idx = b * in_dim + g * group_size;
        
        for (uint i = 0; i < group_size; i++) {
            sum += x[x_idx + i] * codebook[cb_base + i];
        }
    }
    
    output[b * out_dim + od] = sum;
}