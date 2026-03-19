## Summary

On Windows with ClangCL (as reported in issues #493 and #492), the code at line 811 in `src/ggml-bitnet-mad.cpp` declares `int8_t *` but receives a `const int8_t *`, causing compilation error:

```
error: cannot initialize a variable of type 'int8_t *' (aka 'signed char *') with an rvalue of type 'const int8_t *' (aka 'const signed char *')
```

## Fix

Add `const` qualifier to match the actual pointer type being assigned:

```cpp
// Before
int8_t * y_col = y + col * by;

// After
const int8_t * y_col = y + col * by;
```

## Testing

This fix resolves the Windows ClangCL compilation error. The change is safe as it only adds const-correctness without changing the underlying logic.
