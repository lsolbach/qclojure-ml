# VQC Training Performance Analysis

## Executive Summary

VQC training performance is dominated by quantum circuit execution time. With the current simulator:
- **Single circuit execution**: ~100ms (regardless of shot count in 64-1024 range)
- **Derivative-free optimizers** (CMAES, Nelder-Mead): ~10 cost evaluations per iteration
- **Each cost evaluation**: batch_size × 100ms

## Performance Profile (Iris Binary Classification)

### Baseline Configuration (Original)
- Optimizer: Gradient Descent
- Batch size: 120 (full batch)
- Shots: 1024
- **Result**: ~600 seconds for 50 iterations
- **Problem**: 2N+1 evaluations for numerical gradients (24 params → ~49 evals/iteration)

### CMAES Configuration (First Optimization)
- Optimizer: CMAES
- Batch size: 32
- Shots: 256
- Iterations: 100
- **Cost evaluations**: 100 × 10.2 = 1,020
- **Time per evaluation**: 32 circuits × 100ms = 3.2s
- **Total time**: 1,020 × 3.2s = **3,264s (54 minutes)**

### Aggressive Configuration (Recommended)
- Optimizer: CMAES or Nelder-Mead
- Batch size: 8
- Shots: 64
- Iterations: 50
- **Cost evaluations**: 50 × 10 = 500
- **Time per evaluation**: 8 circuits × 100ms = 0.8s
- **Total time**: 500 × 0.8s = **400s (6.7 minutes)**

### Ultra-Fast Configuration (For Experimentation)
- Optimizer: CMAES
- Batch size: 4
- Shots: 32
- Iterations: 30
- **Cost evaluations**: 30 × 10 = 300
- **Time per evaluation**: 4 circuits × 100ms = 0.4s
- **Total time**: 300 × 0.4s = **120s (2 minutes)**

## Key Findings

1. **Circuit execution is the bottleneck**: ~100ms per circuit is constant
2. **Derivative-free optimizers need ~10 cost evaluations per iteration**
3. **Shot count has minimal impact on simulator performance** (64 vs 1024 shots → same 100ms)
4. **Batch size is the main performance lever**: Reducing from 32 → 8 gives 4x speedup

## Recommendations

### For Development/Experimentation
```clojure
{:optimization-method :cmaes
 :max-iterations 30
 :batch-size 4
 :shots 64
 :tolerance 1e-4}
```
**Expected time**: 2-3 minutes

### For Production/Best Accuracy
```clojure
{:optimization-method :cmaes
 :max-iterations 100
 :batch-size 16
 :shots 256
 :tolerance 1e-6}
```
**Expected time**: 15-20 minutes

### Alternative: Gradient-Based with Parameter-Shift
If parameter-shift gradients are implemented:
- 2 evaluations per parameter (24 params → 48 evals)
- With batch-size=8: 48 × 0.8s = 38s per iteration
- For 50 iterations: **31 minutes**
- **Advantage**: Exact gradients, better convergence

## Future Optimizations

1. **Parallel circuit execution**: Execute multiple circuits simultaneously
2. **Circuit caching**: Reuse circuit results for same parameters
3. **Adaptive shot allocation**: Start with low shots, increase for convergence
4. **Hardware acceleration**: Use GPU-accelerated simulators
5. **Circuit optimization**: Reduce circuit depth/gates before execution

## Practical Guidelines

1. **Start small**: Use ultra-fast config for initial development
2. **Validate convergence**: Check if optimizer is actually improving cost
3. **Monitor progress**: Track cost per iteration to detect convergence
4. **Early stopping**: Stop if cost hasn't improved for N iterations
5. **Warm starts**: Reuse parameters from previous runs

## Simulator Performance Notes

The ideal simulator shows consistent ~100ms per circuit execution regardless of:
- Shot count (64-1024): Same performance
- Circuit depth: Hardware-efficient ansatz with 2 layers
- Number of qubits: 4 qubits

This suggests the simulator overhead is dominated by circuit construction and state vector manipulation rather than measurement sampling.
