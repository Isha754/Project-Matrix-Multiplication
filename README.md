# Matrix Multiplication in C

## Introduction

This project is about implementing and comparing different matrix multiplication algorithms in C.

The main goal is to first write the normal matrix multiplication algorithm (GEMM), then study and implement faster algorithms such as Strassen and Coppersmith–Winograd, and finally compare their performance.


---

## Tasks

### Task 1:

Implement the basic matrix multiplication in C.

Requirements:
- Input three integers: `M`, `N`, and `K`
- Matrix sizes are from `512` to `2048`
- Randomly generate matrix `A (M×N)` and matrix `B (N×K)`
- Compute matrix `C`
- Output the execution time

### Task 2:

Study and implement optimized matrix multiplication algorithms.

Algorithms to include:
- Strassen algorithm
- Coppersmith–Winograd algorithm

Requirements:
- Explain the idea of both algorithms
- Optimize the matrix multiplication code using them
- Compare the runtime with normal GEMM
- Add a verification function to check whether the results are correct

---

## Algorithms

### 1. GEMM

This is the normal matrix multiplication using 3 nested loops.

**Time complexity:** `O(n^3)`

It is simple and easy to implement, but it becomes slow for large matrices.

### 2. Strassen Algorithm

Strassen’s algorithm reduces the number of multiplications in recursive matrix multiplication.

**Time complexity:** about `O(n^2.807)`

It is faster than the naive method for large enough matrices, but the implementation is more complicated.

### 3. Coppersmith–Winograd Algorithm

This algorithm improves the theoretical time complexity even more.

**Time complexity:** about `O(n^2.3727)`

It is mostly important in theory, because the real implementation is very complex. In practice, many students use a simplified version or explain the method and compare it conceptually.

---


## How to Compile

To compile this project, open a terminal in the project folder and use gcc.

Example:

gcc -O2 -o matrix_mul main.c gemm.c strassen.c coppersmith_winograd.c utils.c -lm

## How to Run

Run the program like this:

./matrix_mul M N Kv
