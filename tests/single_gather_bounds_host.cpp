// Exercise the actual host/device helpers without PyTorch or a working GPU.
#include "../cuda/common.cuh"
#include <cassert>
#include <iostream>
#include <memory>
#include <random>
#include <utility>
#include <vector>

static bool sees(int query, int key, int win) {
    return win <= 0 || (key <= query && query - key < win);
}

static void check_coverage(int N, int win, int bj, int bk, int side) {
    for (int a = 0; a < N; ++a) {
        const int stride = N + bk;
        std::vector<unsigned char> cover((N + bj) * stride, 0);
        int lo, hi;
        sg_row_bounds(side, a, N, win, bj, lo, hi);
        int first_row = N, last_row = -1;
        for (int j = 0; j < N; ++j) {
            if (side == SG_QUERY_ANCHOR ? sees(a, j, win) : sees(j, a, win)) {
                if (first_row == N) first_row = j;
                last_row = j;
            }
        }
        assert(lo == first_row && hi == last_row + 1);
        for (int j0 = lo; j0 < hi; j0 += bj) {
            int kl, kh;
            sg_col_bounds(side, a, j0, bj, N, win, bk, kl, kh);
            assert(kl >= 0 && kl < N && kh > kl && kh < N + bk);
            assert((kh - kl) % bk == 0);
            // Independently enumerate the mask's column union for the real
            // rows in this tile. Padding must not broaden that union.
            int first_col = N, last_col = -1;
            for (int k = 0; k < N; ++k) {
                bool needed = false;
                for (int j = j0; j < N && j < j0 + bj; ++j) {
                    needed |= side == SG_QUERY_ANCHOR
                        ? sees(a, j, win) && sees(a, k, win)
                        : sees(j, a, win) && sees(j, k, win);
                }
                if (needed) {
                    if (first_col == N) first_col = k;
                    last_col = k;
                }
            }
            assert(kl == first_col && kh > last_col && kh - bk <= last_col);
            for (int j = j0; j < j0 + bj; ++j)
                for (int k = kl; k < kh; ++k)
                    assert(++cover[j * stride + k] == 1);
        }
        for (int j = 0; j < N; ++j) {
            for (int k = 0; k < N; ++k) {
                const bool needed = side == SG_QUERY_ANCHOR
                    ? sees(a, j, win) && sees(a, k, win)
                    : sees(j, a, win) && sees(j, k, win);
                if (needed) assert(cover[j * stride + k] == 1);
            }
        }
    }
}

static void check_mask_packing() {
    std::mt19937 rng(7);
    for (int offset = 0; offset < 8; ++offset) {
        for (int lim = 0; lim <= 32; ++lim) {
            for (int trial = 0; trial < 16; ++trial) {
                // Exact allocation size makes tail overreads visible to ASan.
                std::unique_ptr<bool[]> storage(new bool[offset + lim]);
                for (int t = 0; t < offset + lim; ++t)
                    storage[t] = trial == 0 || (trial != 1 && (rng() & 1));
                const bool* row = storage.get() + offset;
                uint32_t expected = 0;
                for (int t = 0; t < lim; ++t)
                    if (row[t]) expected |= uint32_t(1) << t;
                assert(sg_pack_mask32(row, lim) == expected);
            }
        }
    }
}

static int count_cells(int side, int N, int win, int bj, int bk) {
    int cells = 0;
    for (int a = 0; a < N; ++a) {
        int lo, hi;
        sg_row_bounds(side, a, N, win, bj, lo, hi);
        int anchor_cells = 0;
        for (int j0 = lo; j0 < hi; j0 += bj) {
            int kl, kh;
            sg_col_bounds(side, a, j0, bj, N, win, bk, kl, kh);
            anchor_cells += bj * (kh - kl);
        }
        // At w32, Q-owned interior tiles must contain exactly 32x32 cells.
        // R/S-owned tiles have at most a 32x64 rectangle (triangular mask
        // waste within that rectangle remains).
        if (side == SG_QUERY_ANCHOR && a >= win - 1)
            assert(anchor_cells == 32 * 32);
        assert(anchor_cells <= (side == SG_QUERY_ANCHOR ? 32 * 32 : 32 * 64));
        cells += anchor_cells;
    }
    return cells;
}

int main() {
    const std::vector<std::pair<int, int>> cases = {
        {1, 1}, {15, 7}, {16, 0}, {16, 1}, {17, 16}, {31, 32}, {32, 32},
        {33, 7}, {33, 33}, {48, 48}, {63, 31}, {64, 0}, {64, 7}, {64, 16},
        {65, 32}, {80, 33}, {96, 32}, {96, 40}, {127, 32}, {128, 0},
        {128, 128}, {129, 1}, {129, 136}, {256, 32}, {257, 32}, {272, 40}
    };
    const std::vector<std::pair<int, int>> tiles = {{32, 16}, {32, 32}, {64, 32}, {128, 64}};
    for (auto [N, win] : cases)
        for (auto [bj, bk] : tiles)
            for (int side : {SG_QUERY_ANCHOR, SG_QUERY_ROWS})
                check_coverage(N, win, bj, bk, side);
    check_mask_packing();
    std::cout << "PASS: " << cases.size() * tiles.size() * 2
              << " coverage configurations; 4224 mask alignment/tail cases\n";
    std::cout << "N256 w32 score cells per head: forward="
              << count_cells(SG_QUERY_ANCHOR, 256, 32, 32, 16)
              << " Q_backward=" << count_cells(SG_QUERY_ANCHOR, 256, 32, 32, 32)
              << " each_RS_backward=" << count_cells(SG_QUERY_ROWS, 256, 32, 32, 16)
              << '\n';
}
