// ============================================================================
// KTIR: Level 4c — MatMul MLP Projection
//
// Granite 3.3-8B gate_proj or up_proj.
//   Prefill: [1, 512, 4096] x [4096, 12800] -> [1, 512, 12800]   (M=512, K=4096, N=12800)
//   Decode:  [4, 1, 4096]   x [4096, 12800] -> [4, 1, 12800]     (M=1, K=4096, N=12800)
//
// Stick layout:
//   Activation: M rows x 64 sticks
//   Weight:     64 x 200 sticks (K x N)
//   Output:     M rows x 200 sticks
//
// Core division:
//   Prefill — split M -> 16 rows/core:
//     Weight: 100 MB -- much larger than 2 MB scratchpad.
//     Must be tiled along K and N, streamed from off-chip in tiles.
//   Decode — batch(4) x groups(8) = 32:
//     Each core: N_slice = 12800/8 = 1600 = 25 sticks
//     Weight partition: [K=64, N_slice=25] sticks per core
// ============================================================================

module {

// ── Prefill ─────────────────────────────────────────────────────────────────

func.func @matmul_mlp_prefill() {
  %tile_id = ktdp.get_compute_tile_id : index
  %c16 = arith.constant 16 : index
  %row_start = arith.muli %tile_id, %c16 : index

  %view_act = ktdp.construct_memory_view @activation
      [0 : 1, %row_start : 16, 0 : 64]
      {memory_space = #ktdp.memory_space<off_chip>,
       stick = vector<64xf16>}
      : ktdp.memory_view<16x64xvector<64xf16>>

  %view_weight = ktdp.construct_memory_view @weight
      [0 : 64, 0 : 200]
      {memory_space = #ktdp.memory_space<off_chip>,
       stick = vector<64xf16>}
      : ktdp.memory_view<64x200xvector<64xf16>>

  %view_output = ktdp.construct_memory_view @output
      [0 : 1, %row_start : 16, 0 : 200]
      {memory_space = #ktdp.memory_space<off_chip>,
       stick = vector<64xf16>}
      : ktdp.memory_view<16x200xvector<64xf16>>

  %act_data = ktdp.load %view_act : tensor<16x64xvector<64xf16>>
  %wt_data  = ktdp.load %view_weight : tensor<64x200xvector<64xf16>>

  // [16×64] × [64×200] → [16×200] sticks
  %result = linalg.matmul %act_data, %wt_data
      : tensor<16x64xvector<64xf16>>, tensor<64x200xvector<64xf16>>
        -> tensor<16x200xvector<64xf16>>

  ktdp.store %result, %view_output : tensor<16x200xvector<64xf16>>

  return
}

// ── Decode ──────────────────────────────────────────────────────────────────

func.func @matmul_mlp_decode() {
  %tile_id = ktdp.get_compute_tile_id : index
  %c8  = arith.constant 8 : index
  %c25 = arith.constant 25 : index

  %batch_idx = arith.divui %tile_id, %c8 : index
  %group_idx = arith.remui %tile_id, %c8 : index
  %n_start   = arith.muli %group_idx, %c25 : index

  %view_act = ktdp.construct_memory_view @activation
      [%batch_idx : 1, 0 : 1, 0 : 64]
      {memory_space = #ktdp.memory_space<off_chip>,
       stick = vector<64xf16>}
      : ktdp.memory_view<1x64xvector<64xf16>>

  %view_weight = ktdp.construct_memory_view @weight
      [0 : 64, %n_start : 25]
      {memory_space = #ktdp.memory_space<off_chip>,
       stick = vector<64xf16>}
      : ktdp.memory_view<64x25xvector<64xf16>>

  %view_output = ktdp.construct_memory_view @output
      [%batch_idx : 1, 0 : 1, %n_start : 25]
      {memory_space = #ktdp.memory_space<off_chip>,
       stick = vector<64xf16>}
      : ktdp.memory_view<1x25xvector<64xf16>>

  %act_data = ktdp.load %view_act : tensor<1x64xvector<64xf16>>
  %wt_data  = ktdp.load %view_weight : tensor<64x25xvector<64xf16>>

  %result = linalg.matmul %act_data, %wt_data
      : tensor<1x64xvector<64xf16>>, tensor<64x25xvector<64xf16>>
        -> tensor<1x25xvector<64xf16>>

  ktdp.store %result, %view_output : tensor<1x25xvector<64xf16>>

  return
}

} // module
