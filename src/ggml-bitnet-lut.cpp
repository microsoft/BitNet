#include <stdio.h>  // For printf
#include <string.h> // For memset (if used in future stubs)
#include "ggml-bitnet.h" // For declarations and ggml_tensor struct

extern "C" { // Explicitly wrap definitions

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
    printf("STUB: ggml_bitnet_mul_mat_task_compute called (unconditional stub)\n");
}

// --- Unconditional stub for ggml_bitnet_transform_tensor ---
void ggml_bitnet_transform_tensor(struct ggml_tensor * tensor) {
    (void)tensor;
    printf("STUB: ggml_bitnet_transform_tensor called (unconditional stub)\n");
}

} // end extern "C"
