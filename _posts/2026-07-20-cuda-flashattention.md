---
title: "CUDA 메모리 구조와 타일링으로 이해하는 FlashAttention"
date: 2026-07-20 16:00:00 +0900
categories: [AI Systems, GPU]
tags: [cuda, gpu, flashattention, tiling, attention, memory-hierarchy, sram, hbm]
description: GPU 구조와 CUDA 프로그래밍 모델, 메모리 계층, 행렬곱 타일링을 차례로 짚으면서 FlashAttention이 왜 빠른지 밑바닥부터 정리했습니다.
math: true
image:
  path: /assets/img/posts/flashattention/10_tiling_2.png
  alt: 2×2 타일링으로 행렬곱을 수행하면 A와 B의 각 원소가 HBM에서 2번씩만 읽히는 모습
---

> **TL;DR**
>
> - FlashAttention을 밑바닥부터 이해하기 위한 개념 정리 노트다.
> - GPU 구조 → 연산 단위 → 타일링 → 어텐션 과정 → HBM 접근 순으로 단계적으로 본다.
> - 관련 논문: Tri Dao et al., [*FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness*](https://arxiv.org/abs/2205.14135), NeurIPS 2022
{: .prompt-info }

## 1. Attention의 문제: 왜 느린가

### 표준 attention 구조

![S = QKᵀ로 L×L 점수 행렬을 만들고, √d_k로 나눠 행별 softmax로 P를 만든 뒤, PV로 L×d 출력을 만드는 표준 attention 3단계](/assets/img/posts/flashattention/01_standard_attention.png){: w="1266" h="986" }

- `QKᵀ → softmax → PV` 3단계로 이루어진다.
  - 각 단계의 중간 결과물을 DRAM(HBM)에 저장한다.
- QKᵀ 결과로 **L×L 중간 행렬(S, P)**이 생긴다. (L = 시퀀스 길이, d = 토큰의 차원)

### 학습: O(L²)

- Q도 토큰 L개 전부 → S = (L×d)(d×L) = **(L×L)**
- 중간 행렬이 L×L → **메모리 O(L²)**, 계산 O(L²d)
- QKᵀ, PV 둘 다 계산량이 O(L²d)에 비례한다.

### 추론: KV cache 읽기 병목

![디코딩 한 스텝에서 1×d 쿼리 벡터가 KV cache의 key와 곱해져 1×L 점수를 만들고, softmax 후 value와 곱해져 1×d 출력이 되는 과정](/assets/img/posts/mqa/02_decode_step_attention.png){: w="1287" h="597" }

- Q는 지금 생성 중인 **토큰 1개** → S = (1×d)(d×L) = **(1×L)** (L² 폭발은 없음)
- 대신 매 스텝 **KV cache 전체(L개)를 DRAM에서 읽어야** 한다.
- 계산은 적은데 읽을 데이터가 많다 → 심한 **memory-bound**

### 핵심

- Attention이 느린 건 **계산량이 아니라 메모리** 때문이다.
- 빠른 연산에 비해 **느린 HBM 왕복이 진짜 병목**이다 → compute-bound가 아니라 **memory-bound**

## 2. FlashAttention의 해결책

- 핵심 한 줄: **L×L 중간 행렬을 HBM에 만들지 않고, SRAM에서 블록(타일) 단위로 융합 처리한다.**
- QKᵀ → softmax → PV를 각각 따로 돌리며 큰 행렬을 HBM에 쓰고 읽는 대신, 빠른 온칩 SRAM 안에서 블록 단위로 이어 붙여 계산해 HBM 왕복을 없앤다.

---

## 3. GPU 구조

**엔비디아 AI GPU 세대별 로드맵**

1. **암페어 (Ampere)**: `A100` (2020년)
2. **호퍼 (Hopper)**: `H100`, `H200` (2022~2024년)
3. **블랙웰 (Blackwell)**: `B100`, `B200`, `B300` 등 (2024~2025년)
4. **루빈 (Rubin)**: `R100` 등 (2026년)
   - "베라 루빈"에서 베라(Vera)는 함께 쓰는 CPU를 의미한다.

### Ampere SM 구조 (A100 기준)

![A100 SM 하나가 4개의 서브파티션으로 나뉘고, 각 서브파티션에 워프 스케줄러, 레지스터 파일, FP32·INT32·FP64 코어, Tensor Core, LD/ST, SFU가 있는 구조](/assets/img/posts/flashattention/02_a100_sm.png){: w="743" h="981" }
_출처: NVIDIA A100 Tensor Core GPU Architecture 백서_

- **GPU = SM 108개**
  - SM: Streaming Multiprocessor
  - Ampere도 모델마다 다르다 (RTX 3090 = 82개).
- **SM = 서브파티션(워프 스케줄러) 4개**로 구성
  - 상주 워프: SM당 **최대 64개** (= 2,048 스레드)
    - 워프를 많이 상주시켜서 메모리 레이턴시를 숨긴다.
  - 클럭당 발행: **4워프** (서브파티션마다 1워프씩 명령 발행)
- **서브파티션(워프 스케줄러)**
  - FP32 16 + INT32 16 + FP64 8 + Tensor Core 1 + SFU + LD/ST, 레지스터 파일 16,384 × 32bit
    - FP32, INT32 같은 연산기가 CUDA Core다. CPU의 ALU 같은 것으로, 그냥 연산기다.
  - ❓ 워프는 32스레드 묶음인데 FP32 코어는 16개다.
    - → FP32 워프 명령은 2클럭에 걸쳐 실행된다.
    - **워프 = 32 스레드 (고정)**

## 4. CUDA 프로그래밍 모델

- 소프트웨어 관점의 구조다.
- GPU를 사용하는 함수 호출 ⇒ **커널**
  - 예: A, B 행렬곱 연산을 해라
- **Grid > Block > Warp > Thread** 구조
- CUDA 프로그래밍 모델(Grid·Block·Thread)로 코드를 짜면, 그게 GPU 아키텍처(SM·스케줄러·코어) 위에서 실행된다.

![Grid 안에 Block (0,0)부터 (2,1)까지 6개가 있고, 블록 하나를 확대하면 Warp 0부터 Warp 7까지 32개씩 스레드가 묶여 있는 구조](/assets/img/posts/flashattention/03_grid_block_warp.png){: w="387" h="516" }

### Grid (그리드)

- **커널 호출 한 번 = Grid 하나.** 그 커널을 실행하는 **전체 스레드 집합**이다.
- 개발자가 `gridDim`으로 **블록을 몇 개 둘지**(x, y, z) 정한다.
- 하드웨어 대응: **GPU 전체** (모든 SM에 블록이 분배됨)

### Block (블록)

- **협업하는 스레드 묶음.** 같은 블록의 스레드끼리만 **공유 메모리를 공유하고 `__syncthreads()`로 동기화**할 수 있다.
- `blockDim`으로 크기를 지정한다.
  - 최대 1,024 스레드
  - 1D/2D/3D로 데이터 모양에 맞춰 잡을 수 있다 (예: 16×16).
- **블록끼리는 독립**이다. 직접 통신·동기화가 불가능하고, 실행 순서도 보장되지 않는다.
- 하드웨어 대응: **한 SM에서 실행** (SM은 여러 블록을 담을 수 있음)

### Warp (워프)

- 블록 안의 스레드를 **하드웨어가 32개씩 자동으로 묶은** 실행 단위다.
- 블록 스레드를 **1차원으로 펼친 순서**(x가 가장 빠르게)로 32개씩 끊는다.
- SIMT: 32스레드가 같은 명령을 동시에 실행한다 → **분기(if/else)로 갈리면 워프 다이버전스**로 느려진다.
- 하드웨어 대응: **워프 스케줄러가 굴리는 단위**

### Thread (스레드)

- **가장 작은 실행 단위.** 자기만의 **레지스터, 로컬 변수**를 갖고, 보통 데이터 원소 하나를 담당한다.
  - `threadIdx` + `blockIdx`로 자기 위치를 안다.
  - 전역 인덱스: `blockIdx * blockDim + threadIdx`
- 하드웨어 대응: **CUDA 코어 레인** (코어가 여러 스레드를 시분할)

> **실행 흐름**
>
> 블록들이 대기 → SM이 여러 블록을 동시에 가져옴 → 각 블록의 워프들이 4개 서브파티션에 분산 → 서브파티션 스케줄러가 워프 단위(32스레드)로 실행
>
> 정확히 정리하면, **한 SM에 같이 올라간 블록들끼리**는
>
> - **공유 메모리 용량(SRAM)**: 나눠 쓴다. (A100은 SM당 L1/공유 메모리 합계 192KB, 이 중 공유 메모리로 최대 164KB를 블록들이 각자 몫으로 분할)
> - **데이터**: 공유하지 않는다. (블록 A는 블록 B의 공유 메모리를 볼 수 없고, 서로 격리됨)
>
> 반면 **같은 블록 안에서는** 공유된다.
>
> - **블록 사이**: 용량만 분할, 데이터는 격리
> - **블록 안 스레드끼리**: 공유 메모리 데이터 공유 O (이게 공유 메모리의 존재 이유)
{: .prompt-tip }

## 5. CUDA 메모리 계층 구조 + 실행 모델

![Grid 안의 각 Block이 Shared Memory를 갖고, 각 Thread가 Registers를 가지며, 모든 블록이 Global Memory와 Constant Memory에 접근하고 Host와 연결되는 CUDA 메모리 모델](/assets/img/posts/flashattention/04_cuda_memory_model.png){: w="605" h="556" }

- 블록 단위로 스레드들끼리 공유 메모리를 사용한다.

### 메모리 계층별 대역폭

![GPU SRAM 19TB/s(20MB), GPU HBM 1.5TB/s(40GB), CPU DRAM 12.8GB/s(1TB 이상)로 이루어진 메모리 계층과 대역폭](/assets/img/posts/flashattention/05_memory_hierarchy_bandwidth.png){: w="314" h="272" }
_출처: FlashAttention 논문 Figure 1_

### 메모리 구조

![Device 안에 Multiprocessor 1부터 N까지 있고, 각 Multiprocessor 안에 Shared Memory, 프로세서별 Registers, Instruction Unit, Constant Cache, Texture Cache가 있으며, 맨 아래 Device Memory를 모두가 공유하는 구조](/assets/img/posts/flashattention/06_device_sm_memory.png){: w="736" h="622" }

- **레지스터(Registers)**: 스레드 전용, 가장 빠름
  - 블록별로 레지스터 용량을 나눠 쓰지만, 스레드끼리 데이터는 공유하지 않는다.
- **공유 메모리 / L1 캐시**: SM별 온칩 SRAM, 빠름
  - 그림의 Multiprocessor = SM
  - SM당 shared memory를 같이 사용한다.
  - 단, 블록끼리 공유 메모리 용량은 나눠 쓰지만 블록 간에 데이터 공유는 하지 않는다.

> ❓ **하나의 SM에 블록이 너무 많이 할당되면 공유 메모리는 그만큼 줄어드나?**
>
> - 공유 메모리는 블록당 고정으로 선언한다.
> - 하드웨어는 들어갈 만큼만 블록을 올린다.
> - 블록이 많아서 공유 메모리가 쪼개지는 게 아니라, 블록당 요구량이 크면 동시에 올라가는 블록 수가 줄어든다 **(occupancy)**.
{: .prompt-tip }

- **L2 캐시**: 전체가 공유하는 온칩 메모리
- **글로벌 메모리(DRAM = HBM)**: 오프칩, 크지만 느림, 모든 SM이 공유
- 읽기 전용: **텍스처·상수 메모리** (SM당 캐시 있음)

### 계층 구조

- **Device(GPU)** 안에 **SM(Streaming Multiprocessor)**이 여러 개(1~N) 있다.
- 맨 아래 **Device Memory = 글로벌 메모리(HBM)**는 모든 SM이 공유한다.
- SM 하나 안에는 레지스터 · 코어(Processor) · 공유 메모리 · 상수/텍스처 캐시 · 명령 유닛이 있다.

### 캐시 vs 공유 메모리

- **캐시(L1·텍스처·상수, CPU 스타일)**: 하드웨어가 자동으로 관리한다(LRU 등). 프로그래머가 제어할 수 없다.
- **공유 메모리**: 같은 SRAM이지만 **프로그래머가 "여기 올려라"라고 직접 지정**할 수 있다.
  - 👉 FlashAttention이 가능한 이유: 이 제어 가능한 SRAM에 데이터를 명시적으로 올려 **HBM 왕복을 줄인다**.

### 공유 메모리를 왜 쓰나

- 온칩 SRAM이라 글로벌 메모리(HBM)보다 약 10배 빠르다.
- 단, SM당 수십~수백 KB로 작다 → **타일(블록) 크기를 이 용량에 맞게 잘라야** 한다 (너무 크면 올라가지 않음).
  - 이 제약이 **타일링**을 하는 이유다.

## 6. 타일링이란

### Naive 행렬곱 (타일링 없음)

![4×4 행렬 A와 B를 곱할 때 c(0,0)을 계산하는 스레드가 A의 0행과 B의 0열을 글로벌 메모리에서 읽는 모습](/assets/img/posts/flashattention/07_naive_matmul_1.png){: w="1816" h="1106" }

![모든 결과 원소를 계산하고 나면 A의 각 행과 B의 각 열이 HBM에서 4번씩 읽히는 모습](/assets/img/posts/flashattention/08_naive_matmul_2.png){: w="1864" h="1110" }

- 각 스레드가 자기 좌표를 계산해 결과 한 칸을 맡는다.
- 계산량: 결과 한 칸 = A의 y행 · B의 x열 내적 = 곱셈-덧셈 4회
  - 스레드당 8 FLOP
  - 전체: 16 × 8 = 128 FLOP

### ⚠️ 낭비 포인트: 중복 글로벌 메모리 읽기

- 모든 스레드가 A의 한 행, B의 한 열을 **매번 글로벌 메모리(HBM)에서 새로** 읽는다.
- 이웃 스레드들이 같은 행/열을 중복해서 읽는다 → 같은 데이터가 HBM에서 수십~수백 번 다시 로드된다.
- 계산(FMA)은 빠른데 느린 HBM 접근이 병목이다 → **memory-bound**

### 타일링의 아이디어

![A와 B의 2×2 타일을 공유 메모리에 올리고, 부분 행렬곱 결과를 레지스터에 누적하는 모습](/assets/img/posts/flashattention/09_tiling_1.png){: w="1848" h="1110" }

- A·B를 2×2 타일로 쪼개 **공유 메모리(SRAM)에 한 번 올리고**, 블록 안 스레드들이 재사용한다.
- HBM 재로딩을 크게 줄인다 → memory-bound 완화
- 이게 FlashAttention의 핵심 아이디어(SRAM에 블록을 올려 HBM 왕복 줄이기)이기도 하다.

결과적으로는 한 원소당 HBM에 2번만 접근하게 된다.

![2×2 타일링을 끝까지 수행하면 A와 B의 각 원소가 HBM에서 2번씩만 읽히는 모습](/assets/img/posts/flashattention/10_tiling_2.png){: w="1492" h="952" }

| 구분        | 원소당 FLOP | 전체 FLOP | Global 메모리 읽기          |
| ----------- | ----------- | --------- | --------------------------- |
| 타일링 없음 | 8           | 128       | 128 (원소 32개를 4번씩)     |
| 2×2 타일링  | 8           | 128       | 64 (원소 32개를 2번씩)      |

타일링을 하든 안 하든 FLOP은 같지만, Global 메모리 읽기에서 차이가 난다.

| 타일 크기 T          | DRAM 접근 | 재사용(감소) 배수 |
| -------------------- | --------- | ----------------- |
| 1 (타일링 없음)      | 128번     | 1배               |
| 2                    | 64번      | 2배               |
| 4 (전체가 한 타일)   | 32번      | 4배               |

## 7. Attention 연산 구조

- 아래 예시는 추론 시의 모습이다. (학습 시에는 Query만 L × d가 되면 된다.)

![디코딩 한 스텝에서 1×d 쿼리 벡터가 KV cache의 key와 곱해져 1×L 점수를 만들고, softmax 후 value와 곱해져 1×d 출력이 되는 과정](/assets/img/posts/mqa/02_decode_step_attention.png){: w="1287" h="597" }

## 8. FlashAttention의 핵심 아이디어

### 한 줄 요약

- 행렬곱 타일링에서 **더 나아가**, `행렬곱(QKᵀ) → softmax → 행렬곱(·V)` **세 연산을 하나의 파이프라인으로 융합(fusion)**한다.
- 큰 중간 행렬(S, P = L×L)을 **DRAM(HBM)에 만들지 않고** SRAM 안에서 블록 단위로 처리·누적한다.

### 일반 Attention의 문제

세 단계를 **각각 따로**, 전체 행렬을 다 만들며 실행한다.

1. `S = QKᵀ` → (L×L) 스코어 행렬을 HBM에 쓴다.
2. `P = softmax(S)` → 다시 읽어 softmax하고 또 쓴다.
3. `O = PV` → 다시 읽어 V와 곱한다.

- 문제: 중간 행렬이 L×L → 메모리가 **O(L²)**로 폭발하고, 큰 행렬의 HBM 왕복이 병목이 된다 (**memory-bound**).

**표준 attention: 중간 행렬을 HBM에 저장하는 방식**

| 단계     | 계산량     | 중간 행렬 크기 | HBM 메모리 접근            |
| -------- | ---------- | -------------- | -------------------------- |
| QKᵀ      | O(L²d)     | (L×L)          | S(L×L) **쓰기** → O(L²)    |
| softmax  | O(L²)      | (L×L)          | S 읽기 + P 쓰기 → O(L²)    |
| PV       | O(L²d)     | (L×d)          | P(L×L) **읽기** → O(L²)    |
| **합계** | **O(L²d)** | —              | **O(L²)**                  |

### FlashAttention이 한 일

- Q·K·V를 블록으로 쪼갠다 → 한 블록에 대해 `QKᵀ → softmax → V곱`까지 **SRAM 안에서 연속 처리**하고 부분 결과를 누적한다.
- 큰 S, P 행렬을 **아예 HBM에 만들지 않는다** → 추가 메모리 O(L), HBM 접근 대폭 감소

> "세 연산을 각각 타일링"한 게 아니라, **"하나의 파이프라인으로 합쳤다(fused)"**가 핵심이다.
{: .prompt-warning }

**FlashAttention: 중간 행렬을 HBM에 만들지 않는 방식**

| 항목         | 계산량 | 추가 메모리 | HBM 메모리 접근                    |
| ------------ | ------ | ----------- | ---------------------------------- |
| 전체 (fused) | O(L²d) | O(L)        | **O(L²d² / M)** (M = SRAM 크기)    |

- 타일 크기는 SRAM 크기 M에 맞춰 정해진다. SRAM이 클수록 한 번에 올릴 수 있는 타일이 커지고, 그만큼 HBM 왕복 횟수가 줄어든다.
- 보통 d²이 M보다 훨씬 작기 때문에(예: d = 64~128, M ≈ 100KB), 표준 attention의 O(L²)보다 HBM 접근이 몇 배 적다.

### 어려운 부분: softmax (자세한 내용은 생략)

- 행렬곱 타일링은 부분합을 그냥 더하면 된다 (덧셈은 순서와 무관).
- 하지만 softmax는 분모에 **행 전체의 합(Σ exp)**이 필요하다 → 블록을 나누면 그 시점에는 전체 합을 모른다.
- **해결: online softmax**
  - running max와 running sum을 유지하며 점진적으로 갱신한다.
  - 새 블록이 올 때마다 지금까지의 max·합을 갱신하고, 이전까지 누적한 출력 O를 스케일로 **보정**한다.
  - → 전체 행렬을 한 번에 보지 않아도 **근사 없이 수학적으로 정확히 동일한** softmax 결과를 얻는다.
  - (max를 빼는 건 exp 오버플로 방지용이다.)

### 일반 Attention vs FlashAttention

| 구분                 | 일반 Attention  | FlashAttention                     |
| -------------------- | --------------- | ---------------------------------- |
| 중간 행렬 S, P (L×L) | HBM에 저장      | 만들지 않음 (SRAM 블록 처리)       |
| 세 연산              | 순차·개별 실행  | 하나로 융합(fused)                 |
| softmax              | 전체 행 한 번에 | online softmax로 점진 누적·보정    |
| HBM 접근             | O(L²)           | O(L²d²/M)                          |
| 추가 메모리          | O(L²)           | O(L)                               |
