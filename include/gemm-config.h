#define ACT_PARALLEL
#if defined(__AVX__) || defined(__AVX2__) || defined(__AVX512F__) || defined(__SSSE3__)
#if defined(ACT_PARALLEL)
    #define ROW_BLOCK_SIZE 4
    #define COL_BLOCK_SIZE 128
    #define PARALLEL_SIZE 4
#else
    #define ROW_BLOCK_SIZE 128
    #define COL_BLOCK_SIZE 32
    #define PARALLEL_SIZE 8
#endif // ACT_PARALLEL
#elif defined(__ARM_NEON)
#if defined(__ARM_FEATURE_DOTPROD)
#if defined(ACT_PARALLEL)
    #define ROW_BLOCK_SIZE 8
    #define COL_BLOCK_SIZE 256
    #define PARALLEL_SIZE 8
#else
    #define ROW_BLOCK_SIZE 64
    #define COL_BLOCK_SIZE 16
    #define PARALLEL_SIZE 2
#endif // ACT_PARALLEL
#else
#if defined(ACT_PARALLEL)
    #define ROW_BLOCK_SIZE 8
    #define COL_BLOCK_SIZE 256
    #define PARALLEL_SIZE 4
#else
    #define ROW_BLOCK_SIZE 128
    #define COL_BLOCK_SIZE 32
    #define PARALLEL_SIZE 4
#endif // ACT_PARALLEL
#endif // __ARM_FEATURE_DOTPROD
#elif defined(__VSX__) || defined(__ALTIVEC__) || defined(__powerpc64__) || defined(__powerpc__) || defined(__ppc__)
// PowerPC (G5 AltiVec / POWER8 VSX)
#if defined(ACT_PARALLEL)
    #define ROW_BLOCK_SIZE 4
    #define COL_BLOCK_SIZE 128
    #define PARALLEL_SIZE 4
#else
    #define ROW_BLOCK_SIZE 128
    #define COL_BLOCK_SIZE 32
    #define PARALLEL_SIZE 8
#endif // ACT_PARALLEL
#endif // __AVX__

