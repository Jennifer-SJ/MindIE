# Profiling Comparison: baseline vs compile_no_fusion

## 1. Overview

| Metric | baseline | compile_no_fusion | Delta |
|---|---|---|---|
| Total kernel count | 88091 | 97035 | +8944 |
| Unique operator count | 64 | 66 | +2 |
| Total kernel duration | 304928.7 ms | 392279.7 ms | +87351.0 ms |
| Data source | kernel_details.csv | kernel_details.csv | |

## 2. New Operators (Target Only)

| Operator | Count | Total Dur (ms) | Max Dur (us) |
|---|---|---|---|
| aclnnInplaceCopy_TensorMoveAiCore_TensorMove | 5120 | 2917.44 | 622 |
| aclnnInplaceCopy_StridedSliceAiCore_StridedSlice | 2560 | 1943.08 | 789 |

## 3. Removed Operators (Baseline Only)

*（无）*

## 4. Operator Duration Changes (Common Ops)

| Operator | BC Count | BC Dur(ms) | TC Count | TC Dur(ms) | Count Delta | Dur Delta(ms) | Dur % |
|---|---|---|---|---|---|---|---|
| aclnnInplaceCopy_ViewCopyAiCore_ViewCopy | 2560 | 84702.40 | 5120 | 169369.96 | +2560 | +84667.55 | +100% | **REGRESSION** |
| aclnnCast_CastAiCore_Cast | 7470 | 4564.66 | 6190 | 3479.78 | -1280 | -1084.88 | -24% | *improvement* |
| aclnnFlashAttentionScore_FlashAttentionScore_FlashAttentionS | 1280 | 125318.64 | 1280 | 124292.44 | +0 | -1026.20 | -1% | *improvement* |
| aclnnPowTensorScalar_PowsAiCore_Pows | 2706 | 2321.26 | 2706 | 2247.09 | +0 | -74.16 | -3% | *improvement* |
| aclnnMean_ReduceMeanAiCore_ReduceMean | 2658 | 1426.18 | 2658 | 1498.70 | +0 | +72.52 | +5% | **REGRESSION** |
| aclnnAddmm_MatMulV3Common_MatMulV3 | 5120 | 53566.15 | 5120 | 53611.46 | +0 | +45.31 | +0% | **REGRESSION** |
| aclnnFlashAttentionScore_TransposeAiCore_Transpose | 3840 | 1883.74 | 3840 | 1856.03 | +0 | -27.71 | -1% | *improvement* |
| aclnnLayerNormWithImplMode_LayerNormV3WithImplMode_LayerNorm | 1936 | 2867.74 | 1936 | 2840.51 | +0 | -27.23 | -1% | *improvement* |
| aclnnCast_TransposeAiCore_Transpose | 32 | 34.38 | 16 | 16.91 | -16 | -17.48 | -51% | *improvement* |
| aclnnMul_MulAiCore_Mul | 13224 | 8606.20 | 13224 | 8591.79 | +0 | -14.41 | -0% | *improvement* |
| aclnnSub_SubAiCore_Sub | 1462 | 1167.10 | 1462 | 1154.93 | +0 | -12.17 | -1% | *improvement* |
| aclnnAdd_AddAiCore_Add | 5454 | 5468.82 | 5454 | 5458.23 | +0 | -10.59 | -0% | *improvement* |
| aclnnInplaceCopy_TransposeAiCore_Transpose | 1356 | 1327.42 | 1356 | 1322.69 | +0 | -4.72 | -0% | *improvement* |
| aclnnInplaceCopy_CastAiCore_Cast | 8400 | 5886.00 | 8400 | 5889.79 | +0 | +3.79 | +0% | **REGRESSION** |
| aclnnMul_StridedSliceAiCore_StridedSlice | 10240 | 4106.20 | 10240 | 4107.24 | +0 | +1.04 | +0% | **REGRESSION** |
| aclnnAddmm_MatMulCommon_MatMulV2 | 1376 | 187.10 | 1376 | 186.39 | +0 | -0.71 | -0% | |
| aclnnGelu_Gelu_Gelu | 656 | 1332.31 | 656 | 1331.78 | +0 | -0.53 | -0% | |
| aclnnAdds_AddAiCore_Add | 4099 | 17.19 | 4099 | 17.53 | +0 | +0.33 | +2% | |
| aclnnGeScalar_GreaterEqual_GreaterEqual | 8 | 0.45 | 8 | 0.66 | +0 | +0.21 | +47% | |
| aclnnAddmm_CastAiCore_Cast | 6464 | 12.48 | 6464 | 12.29 | +0 | -0.19 | -1% | |
| aclnnGtScalar_GreaterAiCore_Greater | 48 | 0.90 | 48 | 1.05 | +0 | +0.15 | +17% | |
| aclnnArange_ArangeAiCore_Range | 66 | 0.53 | 66 | 0.67 | +0 | +0.13 | +25% | |
| aclnnRsqrt_RsqrtAiCore_Rsqrt | 2658 | 12.15 | 2658 | 12.27 | +0 | +0.13 | +1% | |
| aclnnInplaceZero_ZerosLikeAiCore_ZerosLike | 1298 | 2.36 | 1298 | 2.47 | +0 | +0.11 | +5% | |
| aclnnGeScalar_CastAiCpu_Cast | 8 | 1.29 | 8 | 1.21 | +0 | -0.08 | -6% | |
| aclnnInplaceOne_OnesLikeAiCore_OnesLike | 1319 | 2.36 | 1319 | 2.44 | +0 | +0.08 | +3% | |
| aclnnMatmul_TransposeAiCore_Transpose | 144 | 2.63 | 144 | 2.71 | +0 | +0.08 | +3% | |
| aclnnMuls_Muls_Muls | 154 | 1.51 | 154 | 1.57 | +0 | +0.05 | +4% | |
| aclnnTanh_Tanh_Tanh | 48 | 3.27 | 48 | 3.23 | +0 | -0.04 | -1% | |
| aclnnSWhere_SelectV2AiCore_SelectV2 | 48 | 0.84 | 48 | 0.87 | +0 | +0.03 | +4% | |
| aclnnCat_BroadcastToAiCore_BroadcastTo | 96 | 1.07 | 96 | 1.10 | +0 | +0.03 | +3% | |
| aclnnConvolution_Conv3dv2WithFlag_Conv3DV2 | 16 | 5.57 | 16 | 5.55 | +0 | -0.02 | -0% | |
| aclnnMatmul_MatMulCommon_MatMulV2 | 336 | 59.26 | 336 | 59.28 | +0 | +0.02 | +0% | |
| aclnnMinimum_MinimumAiCore_Minimum | 48 | 1.26 | 48 | 1.28 | +0 | +0.02 | +1% | |
| aclnnInplaceFillScalar_FillAiCore_Fill | 48 | 0.95 | 48 | 0.93 | +0 | -0.02 | -2% | |
| aclnnAbs_AbsAiCore_Abs | 48 | 2.45 | 48 | 2.46 | +0 | +0.01 | +1% | |
| aclnnConvolution_TransData_TransData | 48 | 10.89 | 48 | 10.88 | +0 | -0.01 | -0% | |
| aclnnEmbedding_GatherV2AiCore_GatherV2 | 50 | 2.65 | 50 | 2.64 | +0 | -0.01 | -0% | |
| aclnnSoftmax_SoftmaxAiCore_SoftmaxV2 | 48 | 6.85 | 48 | 6.84 | +0 | -0.01 | -0% | |
| aclnnLog_LogAiCore_Log | 132 | 0.44 | 132 | 0.45 | +0 | +0.01 | +2% | |
| aclnnCat_ConcatD_ConcatD | 66 | 1.17 | 66 | 1.16 | +0 | -0.01 | -1% | |
| aclnnMuls_MulAiCore_Mul | 164 | 3.11 | 164 | 3.12 | +0 | +0.01 | +0% | |
| aclnnDivs_RealDivAiCore_RealDiv | 112 | 0.63 | 112 | 0.62 | +0 | -0.01 | -1% | |
| aclnnMatmul_BatchMatMulNd_BatchMatMulV2 | 96 | 2.06 | 96 | 2.07 | +0 | +0.01 | +0% | |
| aclnnLtScalar_LessAiCore_Less | 48 | 0.94 | 48 | 0.93 | +0 | -0.00 | -0% | |
| aclnnConvolution_TransData_MemSet | 16 | 0.14 | 16 | 0.14 | +0 | +0.00 | +3% | |
| aclnnCat_SliceAiCore_Slice | 96 | 0.42 | 96 | 0.42 | +0 | -0.00 | -1% | |
| aclnnAdd_TransposeAiCore_Transpose | 48 | 3.63 | 48 | 3.63 | +0 | +0.00 | +0% | |
| aclnnRsubs_SubAiCore_Sub | 44 | 0.06 | 44 | 0.06 | +0 | -0.00 | -3% | |
| aclnnDiv_RealDivAiCore_RealDiv | 108 | 0.22 | 108 | 0.22 | +0 | -0.00 | -1% | |
| aclnnSilu_SiluAiCore_Swish | 32 | 0.06 | 32 | 0.05 | +0 | -0.00 | -3% | |
| aclnnSub_CastAiCore_Cast | 8 | 0.07 | 8 | 0.07 | +0 | +0.00 | +2% | |
| aclnnStack_PackAiCore_Pack | 59 | 0.18 | 59 | 0.18 | +0 | +0.00 | +1% | |
| aclnnConvolution_CastAiCore_Cast | 16 | 0.03 | 16 | 0.03 | +0 | +0.00 | +2% | |
| aclnnSin_SinAiCore_Sin | 16 | 0.03 | 16 | 0.03 | +0 | -0.00 | -2% | |
| aclnnAny_ReduceAny_ReduceAny | 6 | 0.01 | 6 | 0.01 | +0 | +0.00 | +6% | |
| aclnnExp_ExpAiCore_Exp | 16 | 0.02 | 16 | 0.02 | +0 | -0.00 | -3% | |
| aclnnRepeat_TileAiCore_Tile | 2 | 0.02 | 2 | 0.02 | +0 | +0.00 | +2% | |
| aclnnExpm1_Expm1AiCore_Expm1 | 30 | 0.03 | 30 | 0.03 | +0 | -0.00 | -1% | |
| aclnnSubs_SubAiCore_Sub | 42 | 0.05 | 42 | 0.05 | +0 | -0.00 | -0% | |
| aclnnCos_CosAiCore_Cos | 16 | 0.03 | 16 | 0.03 | +0 | +0.00 | +1% | |
| aclnnAny_CastAiCore_Cast | 6 | 0.01 | 6 | 0.01 | +0 | +0.00 | +2% | |
| aclnnPowTensorScalar_PowAiCore_Pow | 27 | 0.09 | 27 | 0.09 | +0 | -0.00 | -0% | |
| aclnnNeg_NegAiCore_Neg | 15 | 0.02 | 15 | 0.02 | +0 | +0.00 | +0% | |

## 5. Summary

- Total kernel duration changed by: **+87351.0 ms (+28.6%)**
- Kernel count changed by: **+8944**
- New operators: 2, Removed operators: 0
- New operators total duration: **4860.5 ms**

## 6. Auto-Verdict

**Verdict: **FAIL** -- compile introduces negative performance impact**

- Total kernel duration: +28.6%
  - => REGRESSION: timed inference slowed by 28.6%
- Kernel count: +10.2% (+8944)
  - => WARNING: kernel count inflated >=10% (functionalization overhead)
- Copy operators: 96022.0ms -> 185550.2ms (+93.2%)
  - => CRITICAL: copy operator overhead >=50%
- Net new operator cost: +4860.5ms (new=4860.5ms, gone=0.0ms)
  - => WARNING: net new operator cost >10ms
