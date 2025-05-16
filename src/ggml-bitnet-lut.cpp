#include <stdio.h>
#include <string.h>
#include "ggml-bitnet.h"
#include <emscripten.h> // For EMSCRIPTEN_KEEPALIVE

extern "C" {

// No keepalive for this one yet, focus on ggml_bitnet_free
void ggml_bitnet_init(void) {
    printf("STUB: ggml_bitnet_init called (unconditional stub)\n");
}

EMSCRIPTEN_KEEPALIVE
void ggml_bitnet_free(void) {
    printf("STUB: ggml_bitnet_free called (unconditional stub, with KEEPALIVE)\n");
}

// No keepalive for this one yet
void ggml_bitnet_mul_mat_task_compute(
    void * src0, void * scales, void * qlut, 
    void * lut_scales, void * lut_biases, void * dst, 
    int n, int k, int m, int bits) {
    
    (void)src0; (void)scales; (void)qlut; (void)lut_scales; (void)lut_biases; 
    (void)n; (void)k; (void)m; (void)bits;
    printf("STUB: ggml_bitnet_mul_mat_task_compute called (unconditional stub)\n");
}

// No keepalive for this one yet
void ggml_bitnet_transform_tensor(struct ggml_tensor * tensor) {
    (void)tensor;
    printf("STUB: ggml_bitnet_transform_tensor called (unconditional stub)\n");
}

} // end extern "C"
