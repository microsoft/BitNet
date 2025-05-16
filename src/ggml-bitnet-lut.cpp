#include <stdio.h>
#include <string.h>
#include "ggml-bitnet.h"
#include <emscripten.h> // For EMSCRIPTEN_KEEPALIVE

extern "C" {

EMSCRIPTEN_KEEPALIVE
void ggml_bitnet_init(void) {
    printf("STUB: ggml_bitnet_init called (KEEPALIVE)\n");
}

EMSCRIPTEN_KEEPALIVE
void ggml_bitnet_free(void) {
    printf("STUB: ggml_bitnet_free called (KEEPALIVE)\n");
}

EMSCRIPTEN_KEEPALIVE
void ggml_bitnet_mul_mat_task_compute(
    void * src0, void * scales, void * qlut, 
    void * lut_scales, void * lut_biases, void * dst, 
    int n, int k, int m, int bits) {
    
    (void)src0; (void)scales; (void)qlut; (void)lut_scales; (void)lut_biases; 
    (void)n; (void)k; (void)m; (void)bits;
    printf("STUB: ggml_bitnet_mul_mat_task_compute called (KEEPALIVE)\n");
}

EMSCRIPTEN_KEEPALIVE
void ggml_bitnet_transform_tensor(struct ggml_tensor * tensor) {
    (void)tensor;
    printf("STUB: ggml_bitnet_transform_tensor called (KEEPALIVE)\n");
}

} // end extern "C"
