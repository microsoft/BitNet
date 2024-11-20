
    #pragma unroll
    for (int i = 0; i < BM{{ pre }}; i += {{ bm }}) {
        #pragma unroll
        for (int i=0; i<{{ bm // 8 }}; i++) {
            vec_c[i] = vandq_s16(vec_c[i], vec_zero);
        }

