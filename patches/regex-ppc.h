// regex-ppc.h - Minimal POSIX regex wrapper for PowerPC big-endian
// std::regex from GCC's libstdc++ is broken on PPC BE (bus error)
// This provides a minimal compatible API using POSIX regex.h

#pragma once

#if defined(__BIG_ENDIAN__) || (defined(__BYTE_ORDER__) && __BYTE_ORDER__ == __ORDER_BIG_ENDIAN__)
#define GGML_PPC_REGEX_COMPAT 1
#endif

#ifdef GGML_PPC_REGEX_COMPAT

#include <regex.h>
#include <string>
#include <vector>
#include <stdexcept>
#include <functional>

namespace std {

// Forward declarations
class regex;
class smatch;

namespace regex_constants {
    enum syntax_option_type {
        ECMAScript = 0,
        icase = REG_ICASE,
        nosubs = REG_NOSUB,
        optimize = 0,
        extended = REG_EXTENDED,
    };

    inline syntax_option_type operator|(syntax_option_type a, syntax_option_type b) {
        return static_cast<syntax_option_type>(static_cast<int>(a) | static_cast<int>(b));
    }
}

class regex_error : public runtime_error {
public:
    explicit regex_error(const string & what) : runtime_error(what) {}
};

class regex {
    regex_t preg;
    bool compiled = false;
public:
    regex() = default;
    explicit regex(const string & pattern, regex_constants::syntax_option_type flags = regex_constants::ECMAScript) {
        int cflags = REG_EXTENDED;
        if (flags & regex_constants::icase) cflags |= REG_ICASE;
        if (flags & regex_constants::nosubs) cflags |= REG_NOSUB;
        int rc = regcomp(&preg, pattern.c_str(), cflags);
        if (rc != 0) {
            char errbuf[256];
            regerror(rc, &preg, errbuf, sizeof(errbuf));
            throw regex_error(string("regex compile error: ") + errbuf);
        }
        compiled = true;
    }
    regex(const regex &) = delete;
    regex & operator=(const regex &) = delete;
    regex(regex && other) noexcept : preg(other.preg), compiled(other.compiled) {
        other.compiled = false;
    }
    regex & operator=(regex && other) noexcept {
        if (compiled) regfree(&preg);
        preg = other.preg;
        compiled = other.compiled;
        other.compiled = false;
        return *this;
    }
    ~regex() { if (compiled) regfree(&preg); }

    const regex_t * native() const { return &preg; }
    bool valid() const { return compiled; }
};

class sub_match {
    string value_;
    bool matched_ = false;
public:
    sub_match() = default;
    sub_match(const string & v, bool m) : value_(v), matched_(m) {}
    string str() const { return value_; }
    operator string() const { return value_; }
    bool matched() const { return matched_; }
    size_t length() const { return value_.length(); }
};

// Suffix result type with .first iterator (needed by json-schema-to-grammar.cpp)
struct match_suffix {
    string value_;
    string::const_iterator first;  // iterator past end of match in original string

    match_suffix() : first() {}
    string str() const { return value_; }
    operator string() const { return value_; }
    size_t length() const { return value_.length(); }
};

class smatch {
    vector<sub_match> matches_;
    string prefix_str_;
    match_suffix suffix_data_;
    int position_ = -1;

public:
    smatch() = default;
    size_t size() const { return matches_.size(); }
    bool empty() const { return matches_.empty(); }
    const sub_match & operator[](size_t i) const { return matches_[i]; }

    int position(size_t i = 0) const { (void)i; return position_; }
    string str(size_t i = 0) const {
        return i < matches_.size() ? matches_[i].str() : "";
    }

    string prefix() const { return prefix_str_; }
    const match_suffix & suffix() const { return suffix_data_; }

    // Set from string-based search
    void _set(const string & input, const regmatch_t * pmatch, size_t nmatch) {
        matches_.clear();
        position_ = -1;
        for (size_t i = 0; i < nmatch; i++) {
            if (pmatch[i].rm_so >= 0) {
                matches_.emplace_back(
                    input.substr(pmatch[i].rm_so, pmatch[i].rm_eo - pmatch[i].rm_so), true);
            } else {
                matches_.emplace_back("", false);
            }
        }
        if (!matches_.empty() && pmatch[0].rm_so >= 0) {
            position_ = pmatch[0].rm_so;
            prefix_str_ = input.substr(0, pmatch[0].rm_so);
            suffix_data_.value_ = input.substr(pmatch[0].rm_eo);
            suffix_data_.first = suffix_data_.value_.begin(); // placeholder
        }
    }

    // Set from iterator-based search (preserves original iterators for suffix().first)
    void _set_with_iters(const string & tmp, const regmatch_t * pmatch, size_t nmatch,
                         string::const_iterator orig_start) {
        _set(tmp, pmatch, nmatch);
        if (position_ >= 0 && !matches_.empty()) {
            // suffix().first must point into the ORIGINAL string, past the match
            suffix_data_.first = orig_start + pmatch[0].rm_eo;
        }
    }
};

inline bool regex_search(const string & s, smatch & m, const regex & re) {
    regmatch_t pmatch[16];
    int rc = regexec(re.native(), s.c_str(), 16, pmatch, 0);
    if (rc == 0) {
        m._set(s, pmatch, 16);
        return true;
    }
    return false;
}

inline bool regex_search(const string & s, const regex & re) {
    int rc = regexec(re.native(), s.c_str(), 0, NULL, 0);
    return rc == 0;
}

// Iterator-based regex_search - preserves original iterators for suffix().first
inline bool regex_search(string::const_iterator first, string::const_iterator last,
                         smatch & m, const regex & re) {
    string s(first, last);
    regmatch_t pmatch[16];
    int rc = regexec(re.native(), s.c_str(), 16, pmatch, 0);
    if (rc == 0) {
        m._set_with_iters(s, pmatch, 16, first);
        return true;
    }
    return false;
}

inline bool regex_match(const string & s, smatch & m, const regex & re) {
    regmatch_t pmatch[16];
    int rc = regexec(re.native(), s.c_str(), 16, pmatch, 0);
    if (rc == 0 && pmatch[0].rm_so == 0 && (size_t)pmatch[0].rm_eo == s.length()) {
        m._set(s, pmatch, 16);
        return true;
    }
    return false;
}

inline bool regex_match(const string & s, const regex & re) {
    smatch m;
    return regex_match(s, m, re);
}

inline string regex_replace(const string & s, const regex & re, const string & replacement) {
    string result;
    string remaining = s;
    regmatch_t pmatch[1];

    while (regexec(re.native(), remaining.c_str(), 1, pmatch, 0) == 0) {
        result += remaining.substr(0, pmatch[0].rm_so);
        result += replacement;
        if (pmatch[0].rm_eo == pmatch[0].rm_so) {
            if ((size_t)pmatch[0].rm_eo < remaining.length()) {
                result += remaining[pmatch[0].rm_eo];
                remaining = remaining.substr(pmatch[0].rm_eo + 1);
            } else {
                break;
            }
        } else {
            remaining = remaining.substr(pmatch[0].rm_eo);
        }
    }
    result += remaining;
    return result;
}

// cmatch - match results for C-string (const char*) regex operations
class cmatch {
    vector<sub_match> matches_;
    int position_ = 0;
public:
    cmatch() = default;
    size_t size() const { return matches_.size(); }
    bool empty() const { return matches_.empty(); }
    const sub_match & operator[](size_t i) const { return matches_[i]; }
    int position(size_t i = 0) const { (void)i; return position_; }
    int length(size_t i = 0) const {
        if (i < matches_.size()) return (int)matches_[i].length();
        return 0;
    }
    string str(size_t i = 0) const {
        return i < matches_.size() ? matches_[i].str() : "";
    }

    void _set(const char * base, const regmatch_t * pmatch, size_t nmatch) {
        matches_.clear();
        if (pmatch[0].rm_so >= 0) {
            position_ = pmatch[0].rm_so;
        }
        for (size_t i = 0; i < nmatch; i++) {
            if (pmatch[i].rm_so >= 0) {
                matches_.emplace_back(
                    string(base + pmatch[i].rm_so, base + pmatch[i].rm_eo), true);
            } else {
                matches_.emplace_back("", false);
            }
        }
    }

    void _adjust_position(int offset) { position_ += offset; }
};

// cregex_iterator - iterates over all non-overlapping matches in a C-string range
class cregex_iterator {
    const char * cur_;
    const char * end_;
    const regex * re_;
    cmatch match_;
    bool at_end_;
    const char * base_;

    void find_next() {
        if (!re_ || cur_ >= end_) {
            at_end_ = true;
            return;
        }
        string tmp(cur_, end_);
        regmatch_t pmatch[16];
        int rc = regexec(re_->native(), tmp.c_str(), 16, pmatch, 0);
        if (rc != 0) {
            at_end_ = true;
            return;
        }
        match_._set(tmp.c_str(), pmatch, 16);
        match_._adjust_position((int)(cur_ - base_));
        if (pmatch[0].rm_eo == pmatch[0].rm_so) {
            cur_ += pmatch[0].rm_eo + 1;
        } else {
            cur_ += pmatch[0].rm_eo;
        }
    }

public:
    cregex_iterator() : cur_(nullptr), end_(nullptr), re_(nullptr), at_end_(true), base_(nullptr) {}

    cregex_iterator(const char * first, const char * last, const regex & re)
        : cur_(first), end_(last), re_(&re), at_end_(false), base_(first) {
        find_next();
    }

    const cmatch & operator*() const { return match_; }
    const cmatch * operator->() const { return &match_; }
    cregex_iterator & operator++() { find_next(); return *this; }

    bool operator==(const cregex_iterator & other) const {
        if (at_end_ && other.at_end_) return true;
        if (at_end_ != other.at_end_) return false;
        return cur_ == other.cur_;
    }
    bool operator!=(const cregex_iterator & other) const { return !(*this == other); }
};

// sregex_token_iterator - split string by regex (submatch -1 = parts between matches)
class sregex_token_iterator {
public:
    using iterator_category = input_iterator_tag;
    using value_type = string;
    using difference_type = ptrdiff_t;
    using pointer = const string *;
    using reference = const string &;

private:
    vector<string> tokens_;
    size_t idx_ = 0;
    bool at_end_ = true;

public:
    sregex_token_iterator() : at_end_(true) {}

    sregex_token_iterator(string::const_iterator first, string::const_iterator last,
                          const regex & re, int submatch) {
        string s(first, last);
        if (submatch == -1) {
            // Split mode: return parts between matches
            string remaining = s;
            regmatch_t pmatch[1];
            while (regexec(re.native(), remaining.c_str(), 1, pmatch, 0) == 0) {
                tokens_.push_back(remaining.substr(0, pmatch[0].rm_so));
                if (pmatch[0].rm_eo == pmatch[0].rm_so) {
                    if ((size_t)pmatch[0].rm_eo < remaining.length()) {
                        remaining = remaining.substr(pmatch[0].rm_eo + 1);
                    } else {
                        break;
                    }
                } else {
                    remaining = remaining.substr(pmatch[0].rm_eo);
                }
            }
            tokens_.push_back(remaining);
        }
        at_end_ = tokens_.empty();
        idx_ = 0;
    }

    const string & operator*() const { return tokens_[idx_]; }
    const string * operator->() const { return &tokens_[idx_]; }

    sregex_token_iterator & operator++() {
        ++idx_;
        if (idx_ >= tokens_.size()) at_end_ = true;
        return *this;
    }

    bool operator==(const sregex_token_iterator & other) const {
        if (at_end_ && other.at_end_) return true;
        if (at_end_ != other.at_end_) return false;
        return idx_ == other.idx_;
    }
    bool operator!=(const sregex_token_iterator & other) const { return !(*this == other); }
};

} // namespace std

#else
// On little-endian, use the real <regex>
#include <regex>
#endif
