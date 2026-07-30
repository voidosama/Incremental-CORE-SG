#pragma once
// Minimal assertion harness -- no external test framework so the project
// builds with nothing but a compiler and CMake.

#include <cmath>
#include <cstdio>
#include <string>

namespace coresg_test {

inline int& failures() {
    static int f = 0;
    return f;
}
inline int& checks() {
    static int c = 0;
    return c;
}

inline void report(bool ok, const char* expr, const char* file, int line,
                   const std::string& extra) {
    ++checks();
    if (ok) return;
    ++failures();
    std::printf("  FAIL %s:%d  %s", file, line, expr);
    if (!extra.empty()) std::printf("   [%s]", extra.c_str());
    std::printf("\n");
}

inline int summary(const char* name) {
    if (failures() == 0) {
        std::printf("[PASS] %s (%d checks)\n", name, checks());
        return 0;
    }
    std::printf("[FAIL] %s (%d/%d checks failed)\n", name, failures(), checks());
    return 1;
}

}  // namespace coresg_test

#define CHECK(cond) ::coresg_test::report((cond), #cond, __FILE__, __LINE__, "")
#define CHECK_MSG(cond, msg) ::coresg_test::report((cond), #cond, __FILE__, __LINE__, (msg))
#define CHECK_EQ(a, b)                                                                    \
    ::coresg_test::report((a) == (b), #a " == " #b, __FILE__, __LINE__,                   \
                          std::to_string(a) + " vs " + std::to_string(b))
#define CHECK_NEAR(a, b, eps)                                                             \
    ::coresg_test::report(std::fabs(double(a) - double(b)) <= double(eps),                \
                          #a " ~= " #b, __FILE__, __LINE__,                               \
                          std::to_string(double(a)) + " vs " + std::to_string(double(b)))
#define CHECK_GE(a, b)                                                                    \
    ::coresg_test::report((a) >= (b), #a " >= " #b, __FILE__, __LINE__,                   \
                          std::to_string(a) + " vs " + std::to_string(b))
#define CHECK_LE(a, b)                                                                    \
    ::coresg_test::report((a) <= (b), #a " <= " #b, __FILE__, __LINE__,                   \
                          std::to_string(a) + " vs " + std::to_string(b))
