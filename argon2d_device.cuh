#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <stdint.h>
#include <string.h>
#include <stdio.h>

//=== Argon2 定数 ===//
#define ARGON2_BLOCK_SIZE 1024
#define ARGON2_QWORDS_IN_BLOCK (ARGON2_BLOCK_SIZE / 8)
#define ARGON2_OWORDS_IN_BLOCK (ARGON2_BLOCK_SIZE / 16)
#define ARGON2_HWORDS_IN_BLOCK (ARGON2_BLOCK_SIZE / 32)
#define ARGON2_SYNC_POINTS 4
#define ARGON2_PREHASH_DIGEST_LENGTH 64
#define ARGON2_PREHASH_SEED_LENGTH 72
#define ARGON2_VERSION_10 0x10
#define ARGON2_VERSION_13 0x13
#define ARGON2_ADDRESSES_IN_BLOCK 128

//=== Blake2b 定数 ===//
#define BLAKE2B_BLOCKBYTES 128
#define BLAKE2B_OUTBYTES 64
#define BLAKE2B_KEYBYTES 64
#define BLAKE2B_SALTBYTES 16
#define BLAKE2B_PERSONALBYTES 16
#define BLAKE2B_ROUNDS 12

//=== 構造体定義 ===//
typedef struct __align__(64) block_ {
    uint64_t v[ARGON2_QWORDS_IN_BLOCK];
} block;

typedef struct Argon2_instance_t {
    block *memory;          /* Memory pointer */
    uint32_t version;
    uint32_t passes;        /* Number of passes */
    uint32_t memory_blocks; /* Number of blocks in memory */
    uint32_t segment_length;
    uint32_t lane_length;
    uint32_t lanes;
    uint32_t threads;
    int print_internals; /* whether to print the memory blocks */
} argon2_instance_t;

/*
 * Argon2 position: where we construct the block right now. Used to distribute
 * work between threads.
 */
typedef struct Argon2_position_t {
    uint32_t pass;
    uint32_t lane;
    uint8_t slice;
    uint32_t index;
} argon2_position_t;

typedef struct __blake2b_state {
    uint64_t h[8];
    uint64_t t[2];
    uint64_t f[2];
    uint8_t buf[BLAKE2B_BLOCKBYTES];
    unsigned buflen;
    unsigned outlen;
    uint8_t last_node;
} blake2b_state;

typedef struct __blake2b_param {
    uint8_t digest_length;                   /* 1 */
    uint8_t key_length;                      /* 2 */
    uint8_t fanout;                          /* 3 */
    uint8_t depth;                           /* 4 */
    uint32_t leaf_length;                    /* 8 */
    uint64_t node_offset;                    /* 16 */
    uint8_t node_depth;                      /* 17 */
    uint8_t inner_length;                    /* 18 */
    uint8_t reserved[14];                    /* 32 */
    uint8_t salt[BLAKE2B_SALTBYTES];         /* 48 */
    uint8_t personal[BLAKE2B_PERSONALBYTES]; /* 64 */
} blake2b_param;

//=== 定数メモリ ===//
__constant__ uint64_t blake2b_IV[8] = {
    0x6a09e667f3bcc908ULL, 0xbb67ae8584caa73bULL,
    0x3c6ef372fe94f82bULL, 0xa54ff53a5f1d36f1ULL,
    0x510e527fade682d1ULL, 0x9b05688c2b3e6c1fULL,
    0x1f83d9abfb41bd6bULL, 0x5be0cd19137e2179ULL
};

__constant__ uint8_t blake2b_sigma[12][16] = {
    {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15},
    {14, 10, 4, 8, 9, 15, 13, 6, 1, 12, 0, 2, 11, 7, 5, 3},
    {11, 8, 12, 0, 5, 2, 15, 13, 10, 14, 3, 6, 7, 1, 9, 4},
    {7, 9, 3, 1, 13, 12, 11, 14, 2, 6, 5, 10, 4, 0, 15, 8},
    {9, 0, 5, 7, 2, 4, 10, 15, 14, 1, 11, 12, 6, 8, 3, 13},
    {2, 12, 6, 10, 0, 11, 8, 3, 4, 13, 7, 5, 15, 14, 1, 9},
    {12, 5, 1, 15, 14, 13, 4, 10, 0, 7, 6, 3, 9, 2, 8, 11},
    {13, 11, 7, 14, 12, 1, 3, 9, 5, 0, 15, 4, 8, 6, 2, 10},
    {6, 15, 14, 9, 11, 3, 0, 8, 12, 2, 13, 7, 1, 4, 10, 5},
    {10, 2, 8, 4, 7, 6, 1, 5, 15, 11, 9, 14, 3, 12, 13, 0},
    {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15},
    {14, 10, 4, 8, 9, 15, 13, 6, 1, 12, 0, 2, 11, 7, 5, 3}
};

//=== 共通ヘルパー関数 ===//
__device__ __forceinline__ uint64_t rotr64(uint64_t x, uint32_t n) {
    return (x >> n) | (x << (64 - n));
}

// fBlaMka関数をCリファレンス実装と完全に一致させる
__device__ __forceinline__ uint64_t fBlaMka(uint64_t x, uint64_t y) {
    const uint64_t m = 0xFFFFFFFFULL;
    uint64_t xy = (x & m) * (y & m);
    return x + y + 2 * xy;
}

// Blake2b G関数 - リファレンス実装と完全に一致させる
__device__ __forceinline__ void blake2b_G(uint64_t& a, uint64_t& b, uint64_t& c, uint64_t& d, uint64_t m1, uint64_t m2) {
    a = a + b + m1;
    d = rotr64(d ^ a, 32);
    c = c + d;
    b = rotr64(b ^ c, 24);
    a = a + b + m2;
    d = rotr64(d ^ a, 16);
    c = c + d;
    b = rotr64(b ^ c, 63);
}

// リトルエンディアンでの32ビット値の格納
__device__ __forceinline__ void store32(void *dst, uint32_t w) {
    #if defined(NATIVE_LITTLE_ENDIAN)
        memcpy(dst, &w, sizeof w);
    #else
        uint8_t *p = (uint8_t *)dst;
        *p++ = (uint8_t)w;
        w >>= 8;
        *p++ = (uint8_t)w;
        w >>= 8;
        *p++ = (uint8_t)w;
        w >>= 8;
        *p++ = (uint8_t)w;
    #endif
    }
__device__ __forceinline__ void blake2b_increment_counter(blake2b_state *S,
    uint64_t inc) {
S->t[0] += inc;
S->t[1] += (S->t[0] < inc);
}

__device__ __forceinline__ void blake2b_set_lastnode(blake2b_state *S) {
    S->f[1] = (uint64_t)-1;
}

__device__ __forceinline__ void blake2b_set_lastblock(blake2b_state *S) {
    if (S->last_node) {
        blake2b_set_lastnode(S);
    }
    S->f[0] = (uint64_t)-1;
}

// Add structure-specific memset function
__device__ void blake2b_state_memset(blake2b_state* S) {
    for (int i = 0; i < sizeof(blake2b_state); i++) {
        ((uint8_t*)S)[i] = 0;
    }
}


// Add missing xor_block function
__device__ void xor_block(block* dst, const block* src) {
    for (int i = 0; i < ARGON2_QWORDS_IN_BLOCK; i++) {
        dst->v[i] ^= src->v[i];
    }
}

// custom memcpy, apparently cuda's memcpy is slow 
// when called within a kernel
__device__ void c_memcpy(void *dest, const void *src, size_t n) {
    uint8_t *d = (uint8_t*)dest;
    const uint8_t *s = (const uint8_t*)src;
    for (size_t i = 0; i < n; i++) {
        d[i] = s[i];
    }
}

// Add missing copy_block function
__device__ void copy_block(block* dst, const block* src) {
    c_memcpy(dst->v, src->v, sizeof(uint64_t) * ARGON2_QWORDS_IN_BLOCK);
}

// fill_blockをCリファレンス実装と完全に一致させる
__device__ void fill_block(const block* prev_block, const block* ref_block, block* next_block, int with_xor) {
    block blockR = {};
    block block_tmp = {};
    unsigned i;

    copy_block(&blockR, ref_block);
    xor_block(&blockR, prev_block);
    copy_block(&block_tmp, &blockR);
    
    if (with_xor) {
        xor_block(&block_tmp, next_block);
    }

    // G function without macro
    auto g = [](uint64_t& a, uint64_t& b, uint64_t& c, uint64_t& d) {
        a = fBlaMka(a, b);
        d = rotr64(d ^ a, 32);
        c = fBlaMka(c, d);
        b = rotr64(b ^ c, 24);
        a = fBlaMka(a, b);
        d = rotr64(d ^ a, 16);
        c = fBlaMka(c, d);
        b = rotr64(b ^ c, 63);
    };

    // BLAKE2_ROUND_NOMSG function without macro
    auto blake2_round = [&g](uint64_t& v0, uint64_t& v1, uint64_t& v2, uint64_t& v3,
                            uint64_t& v4, uint64_t& v5, uint64_t& v6, uint64_t& v7,
                            uint64_t& v8, uint64_t& v9, uint64_t& v10, uint64_t& v11,
                            uint64_t& v12, uint64_t& v13, uint64_t& v14, uint64_t& v15) {
        do {                                                                       
            g(v0, v4, v8, v12);                                                    
            g(v1, v5, v9, v13);                                                    
            g(v2, v6, v10, v14);                                                   
            g(v3, v7, v11, v15);                                                   
            g(v0, v5, v10, v15);                                                   
            g(v1, v6, v11, v12);                                                   
            g(v2, v7, v8, v13);                                                    
            g(v3, v4, v9, v14);                                                    
        } while ((void)0, 0);
    };

    // Apply Blake2 on columns
    for (i = 0; i < 8; ++i) {
        blake2_round(
            blockR.v[16 * i], blockR.v[16 * i + 1], blockR.v[16 * i + 2],
            blockR.v[16 * i + 3], blockR.v[16 * i + 4], blockR.v[16 * i + 5],
            blockR.v[16 * i + 6], blockR.v[16 * i + 7], blockR.v[16 * i + 8],
            blockR.v[16 * i + 9], blockR.v[16 * i + 10], blockR.v[16 * i + 11],
            blockR.v[16 * i + 12], blockR.v[16 * i + 13], blockR.v[16 * i + 14],
            blockR.v[16 * i + 15]
        );
    }

    // Apply Blake2 on rows
    for (i = 0; i < 8; i++) {
        blake2_round(
            blockR.v[2 * i], blockR.v[2 * i + 1], blockR.v[2 * i + 16],
            blockR.v[2 * i + 17], blockR.v[2 * i + 32], blockR.v[2 * i + 33],
            blockR.v[2 * i + 48], blockR.v[2 * i + 49], blockR.v[2 * i + 64],
            blockR.v[2 * i + 65], blockR.v[2 * i + 80], blockR.v[2 * i + 81],
            blockR.v[2 * i + 96], blockR.v[2 * i + 97], blockR.v[2 * i + 112],
            blockR.v[2 * i + 113]
        );
    }

    copy_block(next_block, &block_tmp);
    xor_block(next_block, &blockR);
}

template<typename T, typename ptr_t>
__device__ void c_memset(ptr_t dest, T val, int count) {
    for(int i=0; i<count; i++)
        dest[i] = val;
}

__device__ void init_block_value(block *b, uint8_t in) { c_memset(b->v, in, sizeof(b->v)); }

__device__  void next_addresses(block *address_block, block *input_block,
    const block *zero_block) {
input_block->v[6]++;
fill_block(zero_block, input_block, address_block, 0);
fill_block(zero_block, address_block, address_block, 0);
}

__device__ void G1(uint64_t& a, uint64_t& b, uint64_t& c, uint64_t& d, uint64_t x, uint64_t y) {
    a = a + b + x;
    d = rotr64(d ^ a, 32);
    c = c + d;
    b = rotr64(b ^ c, 24);
    a = a + b + y;
    d = rotr64(d ^ a, 16);
    c = c + d;
    b = rotr64(b ^ c, 63);
}

// Blake2b compression function F
__device__ void blake2b_compress(blake2b_state* S, const uint8_t block[BLAKE2B_BLOCKBYTES]) {
    uint64_t m[16];
    uint64_t v[16];

    // Load message block into m[16]
    for (int i = 0; i < 16; i++) {
        const uint8_t* p = block + i * 8;
        m[i] = ((uint64_t)p[0])
             | ((uint64_t)p[1] << 8)
             | ((uint64_t)p[2] << 16)
             | ((uint64_t)p[3] << 24)
             | ((uint64_t)p[4] << 32)
             | ((uint64_t)p[5] << 40)
             | ((uint64_t)p[6] << 48)
             | ((uint64_t)p[7] << 56);
    }

    // Initialize v[0..15]
    for (int i = 0; i < 8; i++) {
        v[i] = S->h[i];
        v[i + 8] = blake2b_IV[i];
    }

    v[12] ^= S->t[0];
    v[13] ^= S->t[1];
    v[14] ^= S->f[0];
    v[15] ^= S->f[1];

    for (int r = 0; r < BLAKE2B_ROUNDS; r++) {
        const uint8_t* s = blake2b_sigma[r];

        // Column step
        G1(v[0], v[4], v[8], v[12], m[s[0]], m[s[1]]);
        G1(v[1], v[5], v[9], v[13], m[s[2]], m[s[3]]);
        G1(v[2], v[6], v[10], v[14], m[s[4]], m[s[5]]);
        G1(v[3], v[7], v[11], v[15], m[s[6]], m[s[7]]);

        // Diagonal step
        G1(v[0], v[5], v[10], v[15], m[s[8]], m[s[9]]);
        G1(v[1], v[6], v[11], v[12], m[s[10]], m[s[11]]);
        G1(v[2], v[7], v[8], v[13], m[s[12]], m[s[13]]);
        G1(v[3], v[4], v[9], v[14], m[s[14]], m[s[15]]);
    }

    // Finalization
    for (int i = 0; i < 8; i++) {
        S->h[i] ^= v[i] ^ v[i + 8];
    }
}

// Helper functions to load/store 64-bit values in little-endian order
__device__ __forceinline__ uint64_t load64(const void* src) {
    const uint8_t* p = (const uint8_t*)src;
    return ((uint64_t)(p[0]))
        | ((uint64_t)(p[1]) << 8)
        | ((uint64_t)(p[2]) << 16)
        | ((uint64_t)(p[3]) << 24)
        | ((uint64_t)(p[4]) << 32)
        | ((uint64_t)(p[5]) << 40)
        | ((uint64_t)(p[6]) << 48)
        | ((uint64_t)(p[7]) << 56);
}

__device__ __forceinline__ void store64(void* dst, uint64_t w) {
    uint8_t* p = (uint8_t*)dst;
    p[0] = (uint8_t)(w);
    p[1] = (uint8_t)(w >> 8);
    p[2] = (uint8_t)(w >> 16);
    p[3] = (uint8_t)(w >> 24);
    p[4] = (uint8_t)(w >> 32);
    p[5] = (uint8_t)(w >> 40);
    p[6] = (uint8_t)(w >> 48);
    p[7] = (uint8_t)(w >> 56);
}

__device__ void load_block(block *dst, const void *input) {
    unsigned i;
    for (i = 0; i < ARGON2_QWORDS_IN_BLOCK; ++i) {
        dst->v[i] = load64((const uint8_t *)input + i * sizeof(dst->v[i]));
    }
}

__device__ void store_block(void *output, const block *src) {
    unsigned i;
    for (i = 0; i < ARGON2_QWORDS_IN_BLOCK; ++i) {
        store64((uint8_t *)output + i * sizeof(src->v[i]), src->v[i]);
    }
}

// Blake2b init function to match reference implementation exactly
__device__ int blake2b_init(blake2b_state* S, size_t outlen) {
    blake2b_param P;
    // Clear state using our custom function
    blake2b_state_memset(S);

    // Set parameters according to Blake2b spec
    P.digest_length = (uint8_t)outlen;
    P.key_length = 0;
    P.fanout = 1;
    P.depth = 1;
    P.leaf_length = 0;
    P.node_offset = 0;
    P.node_depth = 0;
    P.inner_length = 0;
    c_memset(P.reserved, 0, sizeof(P.reserved));
    c_memset(P.salt, 0, sizeof(P.salt));
    c_memset(P.personal, 0, sizeof(P.personal));

    // Initialize state vector with IV
    for (int i = 0; i < 8; i++) {
        S->h[i] = blake2b_IV[i];
    }

    const unsigned char *p = (const unsigned char *)(&P);
    /* IV XOR Parameter Block */
    for (int i = 0; i < 8; ++i) {
        S->h[i] ^= load64(&p[i * sizeof(S->h[i])]);
    }
    S->outlen = P.digest_length;
    return 0; // Success
}

__device__ int FLAG_clear_internal_memory = 0;
__device__ void clear_internal_memory(void *v, size_t n) {
  if (FLAG_clear_internal_memory && v) {
//    secure_wipe_memory(v, n);
  }
}

// Blake2b update function to match reference implementation
__device__ int blake2b_update(blake2b_state* S, const uint8_t* in, size_t inlen) {
    const uint8_t *pin = (const uint8_t *)in;

    if (inlen == 0) {
        return 0;
    }

    /* Sanity check */
    if (S == NULL || in == NULL) {
        return -1;
    }

    /* Is this a reused state? */
    if (S->f[0] != 0) {
        return -1;
    }

    if (S->buflen + inlen > BLAKE2B_BLOCKBYTES) {
        /* Complete current block */
        size_t left = S->buflen;
        size_t fill = BLAKE2B_BLOCKBYTES - left;
        c_memcpy(&S->buf[left], pin, fill);
        blake2b_increment_counter(S, BLAKE2B_BLOCKBYTES);
        blake2b_compress(S, S->buf);
        S->buflen = 0;
        inlen -= fill;
        pin += fill;
        /* Avoid buffer copies when possible */
        while (inlen > BLAKE2B_BLOCKBYTES) {
            blake2b_increment_counter(S, BLAKE2B_BLOCKBYTES);
            blake2b_compress(S, pin);
            inlen -= BLAKE2B_BLOCKBYTES;
            pin += BLAKE2B_BLOCKBYTES;
        }
    }
    c_memcpy(&S->buf[S->buflen], pin, inlen);
    S->buflen += (unsigned int)inlen;
    return 0; // Success
}

// Blake2b final function to match reference implementation
__device__ int blake2b_final(blake2b_state* S, uint8_t* out, size_t outlen) {
    if (!S || !out)
        return -1;

    uint8_t buffer[BLAKE2B_OUTBYTES] = {0};
    unsigned int i;
    blake2b_increment_counter(S, S->buflen);
    blake2b_set_lastblock(S);
    c_memset(&S->buf[S->buflen], 0, BLAKE2B_BLOCKBYTES - S->buflen); /* Padding */
    blake2b_compress(S, S->buf);

    for (i = 0; i < 8; ++i) { /* Output full hash to temp buffer */
        store64(buffer + sizeof(S->h[i]) * i, S->h[i]);
    }

    c_memcpy(out, buffer, S->outlen);
    return 0;
}

__device__ int blake2b_init_key(blake2b_state *S, size_t outlen, const void *key,
    size_t keylen) {
blake2b_param P;

if (S == NULL) {
return -1;
}

/* Setup Parameter Block for keyed BLAKE2 */
P.digest_length = (uint8_t)outlen;
P.key_length = (uint8_t)keylen;
P.fanout = 1;
P.depth = 1;
P.leaf_length = 0;
P.node_offset = 0;
P.node_depth = 0;
P.inner_length = 0;
c_memset(P.reserved, 0, sizeof(P.reserved));
c_memset(P.salt, 0, sizeof(P.salt));
c_memset(P.personal, 0, sizeof(P.personal));

    // Initialize state vector with IV
    for (int i = 0; i < 8; i++) {
        S->h[i] = blake2b_IV[i];
    }

    // XOR first element with param
    const unsigned char *p = (const unsigned char *)(&P);
    /* IV XOR Parameter Block */
    for (int i = 0; i < 8; ++i) {
        S->h[i] ^= load64(&p[i * sizeof(S->h[i])]);
    }
    S->outlen = P.digest_length;

uint8_t block[BLAKE2B_BLOCKBYTES];
c_memset(block, 0, BLAKE2B_BLOCKBYTES);
c_memcpy(block, key, keylen);
blake2b_update(S, block, BLAKE2B_BLOCKBYTES);
/* Burn the key from stack */
clear_internal_memory(block, BLAKE2B_BLOCKBYTES);
return 0;
}

// Blake2b all-in-one function
__device__ int blake2b(void *out, size_t outlen, const void *in, size_t inlen,
    const void *key, size_t keylen) {
blake2b_state S;
int ret = -1;

/* Verify parameters */
if (NULL == in && inlen > 0) {
goto fail;
}

if (NULL == out || outlen == 0 || outlen > BLAKE2B_OUTBYTES) {
goto fail;
}

if ((NULL == key && keylen > 0) || keylen > BLAKE2B_KEYBYTES) {
goto fail;
}

if (keylen > 0) {
if (blake2b_init_key(&S, outlen, key, keylen) < 0) {
    goto fail;
}
} else {
if (blake2b_init(&S, outlen) < 0) {
    goto fail;
}
}

if (blake2b_update(&S, (const uint8_t*)in, inlen) < 0) {
goto fail;
}
ret = blake2b_final(&S, (uint8_t*)out, outlen);

fail:
clear_internal_memory(&S, sizeof(S));
return ret;
}

// index_alpha関数を完全にCリファレンス実装と一致させる（関数のシグネチャも含め）
__device__ uint32_t index_alpha(const argon2_instance_t *instance,
    const argon2_position_t *position, uint32_t pseudo_rand,
    int same_lane) {
        uint32_t reference_area_size;
        uint64_t relative_position;
        uint32_t start_position, absolute_position;
    
        if (0 == position->pass) {
            /* First pass */
            if (0 == position->slice) {
                /* First slice */
                reference_area_size =
                    position->index - 1; /* all but the previous */
            } else {
                if (same_lane) {
                    /* The same lane => add current segment */
                    reference_area_size =
                        position->slice * instance->segment_length +
                        position->index - 1;
                } else {
                    reference_area_size =
                        position->slice * instance->segment_length +
                        ((position->index == 0) ? (-1) : 0);
                }
            }
        } else {
            /* Second pass */
            if (same_lane) {
                reference_area_size = instance->lane_length -
                                      instance->segment_length + position->index -
                                      1;
            } else {
                reference_area_size = instance->lane_length -
                                      instance->segment_length +
                                      ((position->index == 0) ? (-1) : 0);
            }
        }
    
        /* 1.2.4. Mapping pseudo_rand to 0..<reference_area_size-1> and produce
         * relative position */
        relative_position = pseudo_rand;
        relative_position = relative_position * relative_position >> 32;
        relative_position = reference_area_size - 1 -
                            (reference_area_size * relative_position >> 32);
    
        /* 1.2.5 Computing starting position */
        start_position = 0;
    
        if (0 != position->pass) {
            start_position = (position->slice == ARGON2_SYNC_POINTS - 1)
                                 ? 0
                                 : (position->slice + 1) * instance->segment_length;
        }
    
        /* 1.2.6. Computing absolute position */
        absolute_position = (start_position + relative_position) %
                            instance->lane_length; /* absolute position */
        return absolute_position;
}

// fill_segment関数を追加（Cリファレンス実装と完全に一致）
__device__ void fill_segment(const argon2_instance_t *instance,
    argon2_position_t position) {
        block *ref_block = NULL, *curr_block = NULL;
    block address_block, input_block, zero_block;
    uint64_t pseudo_rand, ref_index, ref_lane;
    uint32_t prev_offset, curr_offset;
    uint32_t starting_index;
    uint32_t i;
    int data_independent_addressing;


    data_independent_addressing = false;

    if (data_independent_addressing) {
        init_block_value(&zero_block, 0);
        init_block_value(&input_block, 0);

        input_block.v[0] = position.pass;
        input_block.v[1] = position.lane;
        input_block.v[2] = position.slice;
        input_block.v[3] = instance->memory_blocks;
        input_block.v[4] = instance->passes;
        input_block.v[5] = 0;
    }

    starting_index = 0;

    if ((0 == position.pass) && (0 == position.slice)) {
        starting_index = 2; /* we have already generated the first two blocks */

        /* Don't forget to generate the first block of addresses: */
        if (data_independent_addressing) {
            next_addresses(&address_block, &input_block, &zero_block);
        }
    }

    /* Offset of the current block */
    curr_offset = position.lane * instance->lane_length +
                  position.slice * instance->segment_length + starting_index;

    if (0 == curr_offset % instance->lane_length) {
        /* Last block in this lane */
        prev_offset = curr_offset + instance->lane_length - 1;
    } else {
        /* Previous block */
        prev_offset = curr_offset - 1;
    }

    for (i = starting_index; i < instance->segment_length;
         ++i, ++curr_offset, ++prev_offset) {
        /*1.1 Rotating prev_offset if needed */
        if (curr_offset % instance->lane_length == 1) {
            prev_offset = curr_offset - 1;
        }

        /* 1.2 Computing the index of the reference block */
        /* 1.2.1 Taking pseudo-random value from the previous block */
        if (data_independent_addressing) {
            if (i % ARGON2_ADDRESSES_IN_BLOCK == 0) {
                next_addresses(&address_block, &input_block, &zero_block);
            }
            pseudo_rand = address_block.v[i % ARGON2_ADDRESSES_IN_BLOCK];
        } else {
            pseudo_rand = instance->memory[prev_offset].v[0];
        }

        /* 1.2.2 Computing the lane of the reference block */
        ref_lane = ((pseudo_rand >> 32)) % instance->lanes;

        if ((position.pass == 0) && (position.slice == 0)) {
            /* Can not reference other lanes yet */
            ref_lane = position.lane;
        }

        /* 1.2.3 Computing the number of possible reference block within the
         * lane.
         */
        position.index = i;
        ref_index = index_alpha(instance, &position, pseudo_rand & 0xFFFFFFFF,
                                ref_lane == position.lane);

        /* 2 Creating a new block */
        ref_block =
            instance->memory + instance->lane_length * ref_lane + ref_index;
        curr_block = instance->memory + curr_offset;
        if (ARGON2_VERSION_10 == instance->version) {
            /* version 1.2.1 and earlier: overwrite, not XOR */
            fill_block(instance->memory + prev_offset, ref_block, curr_block, 0);
        } else {
            if(0 == position.pass) {
                fill_block(instance->memory + prev_offset, ref_block,
                           curr_block, 0);
            } else {
                fill_block(instance->memory + prev_offset, ref_block,
                           curr_block, 1);
            }
        }
    }
}

// fill_memory関数をCリファレンス実装と完全に一致させる
__device__ void fill_memory(block* memory, uint32_t passes, uint32_t lanes, uint32_t lane_length, uint32_t segment_length) {
    argon2_instance_t instance;
    instance.version = ARGON2_VERSION_13;
    instance.passes = passes;
    instance.memory = memory;
    instance.memory_blocks = lanes * lane_length;
    instance.segment_length = segment_length;
    instance.lane_length = lane_length;
    instance.lanes = lanes;
    instance.threads = lanes;
    instance.print_internals = 0;

    argon2_position_t position;
    for (uint32_t pass = 0; pass < passes; ++pass) {
        position.pass = pass;
        for (uint32_t slice = 0; slice < ARGON2_SYNC_POINTS; ++slice) {
            position.slice = slice;
            for (uint32_t lane = 0; lane < lanes; ++lane) {
                position.lane = lane;
                fill_segment(&instance, position);
            }
        }
    }
}

// blake2b_long関数をCリファレンス実装と完全に一致させる
__device__ int blake2b_long(void *pout, size_t outlen, const void *in, size_t inlen) {
    uint8_t *out = (uint8_t *)pout;
    blake2b_state blake_state;
    uint8_t outlen_bytes[sizeof(uint32_t)] = {0};
    int ret = -1;

    if (outlen > UINT32_MAX) {
        goto fail;
    }

    /* Ensure little-endian byte order! */
    store32(outlen_bytes, (uint32_t)outlen);

#define TRY(statement)                                                         \
    do {                                                                       \
        ret = statement;                                                       \
        if (ret < 0) {                                                         \
            goto fail;                                                         \
        }                                                                      \
    } while ((void)0, 0)

    if (outlen <= BLAKE2B_OUTBYTES) {
        TRY(blake2b_init(&blake_state, outlen));
        TRY(blake2b_update(&blake_state, outlen_bytes, sizeof(outlen_bytes)));
        TRY(blake2b_update(&blake_state, (const uint8_t*)in, inlen));
        TRY(blake2b_final(&blake_state, out, outlen));
    } else {
        uint32_t toproduce;
        uint8_t out_buffer[BLAKE2B_OUTBYTES];
        uint8_t in_buffer[BLAKE2B_OUTBYTES];
        TRY(blake2b_init(&blake_state, BLAKE2B_OUTBYTES));
        TRY(blake2b_update(&blake_state, outlen_bytes, sizeof(outlen_bytes)));
        TRY(blake2b_update(&blake_state, (const uint8_t*)in, inlen));
        TRY(blake2b_final(&blake_state, out_buffer, BLAKE2B_OUTBYTES));
        c_memcpy(out, out_buffer, BLAKE2B_OUTBYTES / 2);
        out += BLAKE2B_OUTBYTES / 2;
        toproduce = (uint32_t)outlen - BLAKE2B_OUTBYTES / 2;

        while (toproduce > BLAKE2B_OUTBYTES) {
            c_memcpy(in_buffer, out_buffer, BLAKE2B_OUTBYTES);
            TRY(blake2b(out_buffer, BLAKE2B_OUTBYTES, in_buffer, BLAKE2B_OUTBYTES, NULL, 0));
            c_memcpy(out, out_buffer, BLAKE2B_OUTBYTES / 2);
            out += BLAKE2B_OUTBYTES / 2;
            toproduce -= BLAKE2B_OUTBYTES / 2;
        }

        c_memcpy(in_buffer, out_buffer, BLAKE2B_OUTBYTES);
        TRY(blake2b(out_buffer, toproduce, in_buffer, BLAKE2B_OUTBYTES, NULL,
                    0));
        c_memcpy(out, out_buffer, toproduce);
    }
fail:
    clear_internal_memory(&blake_state, sizeof(blake_state));
    return ret;
#undef TRY
}

// device_argon2d_hash関数を完全にCリファレンス実装と一致させる
__device__ void device_argon2d_hash(
    uint8_t* output,
    const uint8_t* input, size_t input_len,
    uint32_t t_cost, uint32_t m_cost, uint32_t lanes,
    block* memory,
    const uint8_t* salt, size_t salt_len
) {
    argon2_instance_t instance;
    // 1. メモリサイズの調整
    uint32_t memory_blocks = m_cost;
    if (memory_blocks < 2 * ARGON2_SYNC_POINTS * lanes) {
        memory_blocks = 2 * ARGON2_SYNC_POINTS * lanes;
    }
    
    uint32_t segment_length = memory_blocks / (lanes * ARGON2_SYNC_POINTS);
    memory_blocks = segment_length * (lanes * ARGON2_SYNC_POINTS);
    uint32_t lane_length = segment_length * ARGON2_SYNC_POINTS;
    
    // Initialize instance with the provided memory pointer
    instance.version = ARGON2_VERSION_13;
    instance.memory = memory;  // Use the provided memory pointer
    instance.passes = t_cost;
    instance.memory_blocks = memory_blocks;
    instance.segment_length = segment_length;
    instance.lane_length = lane_length;
    instance.lanes = lanes;
    instance.threads = 1;

    // 2. 初期ハッシュの計算
    uint8_t blockhash[ARGON2_PREHASH_DIGEST_LENGTH];
    blake2b_state BlakeHash;

    blake2b_init(&BlakeHash, ARGON2_PREHASH_DIGEST_LENGTH);
    
    uint8_t value[sizeof(uint32_t)];

    store32(&value, lanes);
    blake2b_update(&BlakeHash, (uint8_t*)&value, sizeof(value));
    
    store32(&value, 32);
    blake2b_update(&BlakeHash, (uint8_t*)&value, sizeof(value));
    
    store32(&value, memory_blocks);
    blake2b_update(&BlakeHash, (uint8_t*)&value, sizeof(value));
    
    store32(&value, t_cost);
    blake2b_update(&BlakeHash, (uint8_t*)&value, sizeof(value));
    
    store32(&value, ARGON2_VERSION_13);
    blake2b_update(&BlakeHash, (uint8_t*)&value, sizeof(value));
    
    store32(&value, 0);
    blake2b_update(&BlakeHash, (uint8_t*)&value, sizeof(value));
    
    store32(&value, input_len);
    blake2b_update(&BlakeHash, (uint8_t*)&value, sizeof(value));

    blake2b_update(&BlakeHash, (const uint8_t *)input, input_len);
    
    store32(&value, salt_len);
    blake2b_update(&BlakeHash, (uint8_t*)&value, sizeof(value));

    blake2b_update(&BlakeHash, (const uint8_t *)salt, salt_len);
    store32(&value, 0);

    blake2b_update(&BlakeHash, (uint8_t*)&value, sizeof(value));
    
    store32(&value, 0);

    blake2b_update(&BlakeHash, (uint8_t*)&value, sizeof(value));
    
    
    blake2b_final(&BlakeHash, blockhash, ARGON2_PREHASH_DIGEST_LENGTH);

    // 3. Initialize first blocks in each lane
    uint8_t blockhash_bytes[ARGON2_BLOCK_SIZE];
    uint8_t initial_hash[ARGON2_PREHASH_SEED_LENGTH];
    c_memcpy(initial_hash, blockhash, ARGON2_PREHASH_DIGEST_LENGTH);
    c_memset(initial_hash + ARGON2_PREHASH_DIGEST_LENGTH, 0, ARGON2_PREHASH_SEED_LENGTH - ARGON2_PREHASH_DIGEST_LENGTH);

    for (uint32_t l = 0; l < lanes; ++l) {
        store32(initial_hash + ARGON2_PREHASH_DIGEST_LENGTH, 0);
        store32(initial_hash + ARGON2_PREHASH_DIGEST_LENGTH + 4, l);
        
        blake2b_long(blockhash_bytes, ARGON2_BLOCK_SIZE, initial_hash, ARGON2_PREHASH_SEED_LENGTH);
        load_block(&memory[l * lane_length], blockhash_bytes);
        
        store32(initial_hash + ARGON2_PREHASH_DIGEST_LENGTH, 1);
        blake2b_long(blockhash_bytes, ARGON2_BLOCK_SIZE, initial_hash, ARGON2_PREHASH_SEED_LENGTH);
        load_block(&memory[l * lane_length + 1], blockhash_bytes);
    }

    // 4. Fill memory
    fill_memory(memory, t_cost, lanes, lane_length, segment_length);

    // 5. Final block mixing
    block final_block;
    copy_block(&final_block, &memory[0 * lane_length + (lane_length - 1)]);
    
    for (uint32_t l = 1; l < lanes; ++l) {
        uint32_t last_block_in_lane = l * lane_length + (lane_length - 1);
        xor_block(&final_block, &memory[last_block_in_lane]);
    }

    // 6. Final hash
    uint8_t final_block_bytes[ARGON2_BLOCK_SIZE];
    store_block(final_block_bytes, &final_block);

    blake2b_long(output, 32, final_block_bytes, ARGON2_BLOCK_SIZE);

}

//=== __global__ カーネル例（salt 指定版）===//
// ホスト側でブロック用メモリをあらかじめ確保し、そのポインタ（memory_ptr）を渡すことを前提としています。
__global__ void argon2d_hash_device_kernel(
    uint8_t* output,
    const uint8_t* input, size_t input_len,
    uint32_t t_cost, uint32_t m_cost, uint32_t lanes,
    block* memory_ptr,   // ホスト側で確保したメモリ領域へのポインタ
    const uint8_t* salt, size_t salt_len
) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        device_argon2d_hash(output, input, input_len, t_cost, m_cost, lanes, memory_ptr, salt, salt_len);
    }
}
