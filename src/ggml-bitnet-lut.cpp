#include <stdio.h>  // For printf
#include <string.h> // For memset (if used in future stubs)
#include "ggml-bitnet.h" // For declarations and ggml_tensor struct

// --- Unconditional stub for ggml_bitnet_init ---
void ggml_bitnet_init(void) {
    printf("STUB: ggml_bitnet_init called (unconditional stub)\n");
}

// --- Unconditional stub for ggml_bitnet_free ---
void ggml_bitnet_free(void) {
    printf("STUB: ggml_bitnet_free called (unconditional stub)\n");
}

// --- Unconditional stub for ggml_bitnet_mul_mat_task_compute ---
void ggml_bitnet_mul_mat_task_compute(
    const void * src0, const void * scales, const void * qlut, 
    const void * lut_scales, const void * lut_biases, void * dst, 
    int n, int k, int m, int bits) {
    
    (void)src0; (void)scales; (void)qlut; (void)lut_scales; (void)lut_biases; 
    (void)n; (void)k; (void)m; (void)bits;

    if (dst && m > 0 && n > 0) { // Assuming n is columns, m is rows for dst
        // This is a guess for dst structure. A real implementation needs precise details.
        // For a stub, zeroing out is a safe operation if dst is writable.
        // Example: if dst is float*, (m*n) elements.
        // float* out_dst = (float*)dst;
        // for(int i=0; i < m*n; ++i) out_dst[i] = 0.0f;
    }
    printf("STUB: ggml_bitnet_mul_mat_task_compute called (unconditional stub)\n");
}

// --- Unconditional stub for ggml_bitnet_transform_tensor ---
void ggml_bitnet_transform_tensor(struct ggml_tensor * tensor) {
    (void)tensor; // Mark as unused
    printf("STUB: ggml_bitnet_transform_tensor called (unconditional stub)\n");
}
