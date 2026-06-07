// test_dense_is_default.cpp — Verify dense is default when no env var set
//
// D-T-01 / actions.md T008: "Sem env var BITNET_SPARSE_TOPK, o dispatch em
// src/ggml-bitnet-dispatch.cpp NÃO invoca sparse_attention_float()".
//
// Abordagem: análise estática do source. Confirma que:
//   1. A função `sparse_attention_float` é chamada em exatamente 1 local
//      (`ggml-bitnet-tropical.cpp:385` é a definição; `ggml-bitnet-dispatch.cpp:349`
//      é o call site dentro de `sparse_float_callback`).
//   2. A função default de dispatch é `tropical_callback` (caminho ternário), que
//      NÃO chama `sparse_attention_float` — o caminho sparse é opt-in via
//      `bitnet_op_sparse_attn` que precisa ser explicitamente wired no llama.cpp.
//   3. O nome BITNET_SPARSE_TOPK aparece no comment header do `sparse_float_callback`,
//      documentando a convention.
//
// Build:
//   clang++ -O2 -std=c++17 \
//     -I/usr/include/c++/13 -I/usr/include/x86_64-linux-gnu/c++/13 \
//     test_dense_is_default.cpp -o build/test_dense_is_default
//
// Convention: hand-rolled `assert(...)` per T003 (no Catch2 in this project).

#ifndef SOURCE_DIR
#define SOURCE_DIR "."
#endif

#include <cstdio>
#include <cstring>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

static int n_pass = 0, n_total = 0;

static void report(const char * name, bool ok, const char * detail = "") {
    n_total++;
    if (ok) n_pass++;
    printf("  %-60s %s   %s\n", name, ok ? "PASS ✓" : "FAIL ✗", detail);
}

/* ── Read source file ──────────────────────────────────────────────────── */

static std::string read_file(const char * path) {
    std::ifstream f(path);
    if (!f) return "";
    std::stringstream ss;
    ss << f.rdbuf();
    return ss.str();
}

/* Strip C++ comments (// and block) to avoid false matches */

static std::string strip_comments(const std::string & src) {
    std::string out;
    out.reserve(src.size());
    size_t i = 0;
    while (i < src.size()) {
        // Block comment
        if (i + 1 < src.size() && src[i] == '/' && src[i + 1] == '*') {
            i += 2;
            while (i + 1 < src.size() && !(src[i] == '*' && src[i + 1] == '/')) i++;
            i += 2;
            continue;
        }
        // Line comment
        if (i + 1 < src.size() && src[i] == '/' && src[i + 1] == '/') {
            while (i < src.size() && src[i] != '\n') i++;
            continue;
        }
        out += src[i++];
    }
    return out;
}

/* Test 1: sparse_attention_float has exactly 1 call site (in dispatch, not llama.cpp) */

static int test_sparse_call_count() {
    printf("\n[1] sparse_attention_float is called from exactly 1 site in dispatch\n");
    std::string raw = read_file("src/ggml-bitnet-dispatch.cpp");
    if (raw.empty()) {
        // Try with absolute path (cmake places tests in build/tests/)
        raw = read_file(SOURCE_DIR "/src/ggml-bitnet-dispatch.cpp");
    }
    if (raw.empty()) {
        report("read source", false, "src/ggml-bitnet-dispatch.cpp not found (cwd or SOURCE_DIR)");
        return 0;
    }
    std::string src = strip_comments(raw);
    // Count occurrences of "sparse_attention_float(" (function call, not definition/declaration)
    int count = 0;
    size_t pos = 0;
    while ((pos = src.find("sparse_attention_float(", pos)) != std::string::npos) {
        count++;
        pos += std::string("sparse_attention_float(").size();
    }
    char det[96];
    std::snprintf(det, sizeof(det), "found %d call site(s) in dispatch", count);
    report("single call site in dispatch.cpp", count == 1, det);
    return count == 1;
}

/* Test 2: default dispatch (tropical_callback) does NOT call sparse */

static int test_default_path_no_sparse() {
    printf("\n[2] default path (tropical_callback) does not call sparse_attention_float\n");
    std::string raw = read_file("src/ggml-bitnet-dispatch.cpp");
    if (raw.empty()) {
        raw = read_file(SOURCE_DIR "/src/ggml-bitnet-dispatch.cpp");
    }
    if (raw.empty()) {
        report("read source", false, "src/ggml-bitnet-dispatch.cpp not found (cwd or SOURCE_DIR)");
        return 0;
    }
    std::string src = strip_comments(raw);

    // Find tropical_callback function body
    size_t tcb = src.find("tropical_callback(");
    if (tcb == std::string::npos) {
        report("tropical_callback defined", false, "function not found");
        return 0;
    }
    // Find the next function definition (heuristic: top-level 'struct' or 'static void' at column 0)
    // Walk forward to find the end of tropical_callback
    size_t end = src.find("\nstatic void ", tcb + 1);
    if (end == std::string::npos) end = src.find("\nstruct ", tcb + 1);
    if (end == std::string::npos) end = src.size();
    std::string body = src.substr(tcb, end - tcb);

    bool has_sparse_call = body.find("sparse_attention_float(") != std::string::npos;
    char det[128];
    std::snprintf(det, sizeof(det), "tropical_callback body calls sparse: %s",
                  has_sparse_call ? "yes (BAD)" : "no (GOOD)");
    report("tropical_callback (default) does NOT call sparse", !has_sparse_call, det);
    return has_sparse_call ? 0 : 1;
}

/* Test 3: BITNET_SPARSE_TOPK is documented in the dispatch comment header */

static int test_sparse_env_documented() {
    printf("\n[3] BITNET_SPARSE_TOPK is documented as opt-in env var\n");
    std::string raw = read_file("src/ggml-bitnet-dispatch.cpp");
    if (raw.empty()) {
        raw = read_file(SOURCE_DIR "/src/ggml-bitnet-dispatch.cpp");
    }
    if (raw.empty()) {
        report("read source", false, "src/ggml-bitnet-dispatch.cpp not found (cwd or SOURCE_DIR)");
        return 0;
    }
    // We keep the comments this time (search in raw)
    bool documented = raw.find("BITNET_SPARSE_TOPK") != std::string::npos;
    char det[96];
    std::snprintf(det, sizeof(det), "found in dispatch: %s", documented ? "yes" : "no");
    report("env var documented in dispatch", documented, det);
    return documented ? 1 : 0;
}

/* Main */

int main() {
    printf("═══════════════════════════════════════════════════════════\n");
    printf("  D-T-01: dense is default when BITNET_SPARSE_TOPK unset\n");
    printf("  (Static analysis of src/ggml-bitnet-dispatch.cpp)\n");
    printf("═══════════════════════════════════════════════════════════\n");
    test_sparse_call_count();
    test_default_path_no_sparse();
    test_sparse_env_documented();
    printf("\n═══════════════════════════════════════════════════════════\n");
    printf("  Resultado: %d/%d checks %s\n", n_pass, n_total,
           n_pass == n_total ? "PASSARAM ✓" : "FALHARAM ✗");
    printf("═══════════════════════════════════════════════════════════\n");
    return n_pass == n_total ? 0 : 1;
}
