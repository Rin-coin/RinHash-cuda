# RinHash CUDA Implementation

🚀 High-performance GPU implementation of RinHash – an ASIC-resistant hashing algorithm designed for RinCoin mining.

## 🔧 Algorithm Overview

RinHash is a custom Proof-of-Work algorithm designed to resist ASICs by combining three cryptographic hash functions:

1. **BLAKE3** – Fast and modern hashing.
2. **Argon2d** – Memory-hard password hashing (64KB, 2 iterations).
3. **SHA3-256** – Secure final hash.

The final output is a 32-byte SHA3-256 digest of the Argon2d result, which itself is applied to the BLAKE3 hash of the input block header.

---

## 💻 CUDA Implementation

This repository contains a full GPU-based implementation of RinHash, ported to CUDA for use in high-efficiency miners. Key features include:

- Full GPU parallelization of BLAKE3, Argon2d, and SHA3-256
- Memory-hard Argon2d executed entirely on device memory
- Batch processing support for multiple nonces
- Matching hash output with official CPU implementation

---
