---
title: "[논문 정리] MQA (2019): Multi-Query Attention으로 디코딩 메모리 병목 줄이기"
date: 2026-09-16 21:00:00 +0900
categories: [Paper Review, LLM Inference]
tags: [mqa, gqa, mla, attention, kv-cache, llm, inference]
description: K/V head를 하나로 공유해 디코딩 시 KV cache 읽기 부담을 줄인 Multi-Query Attention(2019)과, 이후 GQA·MLA로 이어진 흐름을 정리했습니다.
math: true
image:
  path: /assets/img/posts/mqa/04_mha_vs_mqa.png
  alt: MHA는 head마다 K/V를 따로 두고, MQA는 모든 head가 K/V 하나를 공유하는 구조 비교
---

> **논문**: Noam Shazeer (Google), [*Fast Transformer Decoding: One Write-Head is All You Need*](https://arxiv.org/abs/1911.02150), 2019
{: .prompt-info }

## Abstract

- 트랜스포머의 MHA(Multi-Head Attention) 레이어는 시퀀스 내부 및 시퀀스 간 정보를 전달하는 데 강력하다.
- 시퀀스 길이 방향으로 병렬화가 가능해서 학습이 빠르다.
- 하지만 병렬화가 불가능한 점진적 추론(incremental inference)은 K, V를 반복해서 로드해야 해서 메모리 대역폭 비용 때문에 느리다.
- head 간에 key와 value를 공유하는 **Multi-Query Attention**을 제안한다.
  - 텐서 크기를 줄여 디코딩에 필요한 메모리 대역폭 요구를 줄인다.
- 실험 결과 실제로 디코딩 속도가 크게 빨라진다.

## 1. Introduction

> One major challenge with Transformer is the speed of incremental inference.

> **Incremental inference(점진적 추론)**
>
> decoder가 토큰을 하나씩 생성하는 과정을 말한다.
{: .prompt-tip }

- 점진적 추론의 속도 문제
  - Key, Value 텐서를 다시 로드하는 데 메모리 대역폭 제한이 있다.
- KV cache가 없앤 것은 재계산이다(이전 토큰의 K/V를 매번 다시 계산하지 않음).
  - 하지만 어텐션 연산을 할 때는 이 cache를 전부 읽어야 하고, 이 부분이 memory bound다.
  - 즉 HBM에 있는 KV cache를 어텐션 행렬 연산 때마다 SRAM으로 불러오는 과정이 병목이다.
- 이런 추론 속도를 높이면서 품질 저하는 최소화하는 방법이 MQA(Multi-Query Attention)다.

## 2. Background: Neural Attention

![Attention(Q, K, V) = softmax(QKᵀ / √d_k) V 수식에서 QKᵀ 부분을 강조한 그림](/assets/img/posts/mqa/01_attention_formula.png){: w="1972" h="362" }

- QKᵀ: score 행렬
- Attention(Q, K, V): 어텐션 행렬

### 2.1 Dot-product Attention

query가 행렬이 아니라 벡터인 이유는, 추론 시에는 디코딩 한 스텝만 계산하기 때문이다.

![디코딩 한 스텝에서 1×d 쿼리 벡터가 KV cache의 key와 곱해져 1×L 점수를 만들고, softmax 후 value와 곱해져 1×d 출력이 되는 과정](/assets/img/posts/mqa/02_decode_step_attention.png){: w="1287" h="597" }

- q는 위 그림에서 "you" 토큰의 쿼리 벡터다.
- 앞 토큰들의 정보는 이미 KV cache에 다 들어 있다.

### 2.2 Multi-head Attention

각 가중치 행렬이 head 개수만큼 생긴다.

![여러 head가 각자 Queries, Keys, Values를 만들어 어텐션을 계산한 뒤 결과를 합치는 Multi-head attention 구조](/assets/img/posts/mqa/03_multi_head_attention.png){: w="819" h="908" }

- 즉 기존 cache 차원이 d × L이었던 게 h × d × L로 바뀐다.
  - 메모리 필요량이 h배가 된다.
  - (다만 head를 늘리는 만큼 head당 차원 d를 낮추는 방식도 있어서, 무조건 몇 배라는 개념은 아니다.)

### 2.3 Multi-head Attention 학습 / prefill (batch)

- 학습 시에는 n개의 query가 병렬로 쓰이고, batch를 이용해 가중치도 재사용한다.
- 즉 한 번 읽은 K로 N번의 연산을 한다.
  - 읽기 부하가 아니라 compute bound다.
  - GPU를 놀리지 않고 잘 쓰고 있다는 뜻이다.

### 2.4 Multi-head Attention decode

- 점진적 decode, 즉 토큰 생성
  - 순차 생성이라 병렬화가 불가능하고, KV cache가 필수다.
- K cache를 읽는데 query는 1개뿐이다.
  - 읽기 1회당 연산 1회
  - memory bound → GPU가 논다.

**해결책으로는**

- 시퀀스 자체를 줄인다 → 성능 이슈
- multi-head에서 head는 유지한 채, K/V 텐서에서 head 차원을 제거한다 → **이 논문의 주제**

## 3. Multi-Query Attention

![MHA는 head마다 Key, Value가 따로 있고, MQA는 Key와 Value 하나를 모든 Query head가 공유하는 구조 비교](/assets/img/posts/mqa/04_mha_vs_mqa.png){: w="845" h="640" }

- **Multi-head attention**
  - query, key, value에 서로 다른 선형 변환을 적용하는 여러 attention layer(head)가 병렬로 구성된 형태
- **Multi-query attention**
  - 아이디어: K, V를 하나만 두고 서로 다른 head들이 공유하도록 한다.

> **⇒ 기존에 HBM에서 가져와야 했던 h × d × L 크기의 캐시가 d × L 한 번으로 줄어든다.**
{: .prompt-warning }

메모리 접근 대비 연산 비율(memory-to-compute ratio)을 수식으로 보면 다음과 같다.

- Before (MHA)

$$
\Theta\!\left(\frac{n}{d}+\frac{1}{b}\right)
$$

- After (MQA)

$$
\Theta\!\left(\frac{1}{d}+\frac{n}{dh}+\frac{1}{b}\right)
$$

→ 병목이던 $\frac{n}{d}$ 항이 $\frac{1}{h}$배로 줄어든다.

## 4. Experiments and Results

### Table 1: 번역 품질 (BLEU 0.2 차이)

| 모델                  | h   | d_k, d_v | ln(PPL)↓  | BLEU(dev)↑ | BLEU(test) greedy / beam4 |
| --------------------- | --- | -------- | --------- | ---------- | ------------------------- |
| multi-head (baseline) | 8   | 128      | **1.424** | **26.7**   | 27.7 / 28.4               |
| multi-query           | 8   | 128      | 1.439     | 26.5       | 27.5 / **28.5**           |

### Table 2: 속도 (디코더 약 12배 차이)

| 모델        | Training | Inference (enc + dec) |
| ----------- | -------- | --------------------- |
| multi-head  | 13.2     | 1.7 + **46**          |
| multi-query | **13.0** | 1.5 + **3.8**         |

### Table 3: 언어 모델링

| 모델        | h   | d_k, d_v | dev-PPL  |
| ----------- | --- | -------- | -------- |
| multi-head  | 8   | 128      | **29.9** |
| multi-query | 8   | 128      | 30.2     |

**디코더는 약 12배 빨라지고, BLEU 점수는 0.2 손실**

## 5. Conclusion

- 디코딩에서 메모리 대역폭 요구가 컸다.
- 그 대안으로 multi-query attention을 제안했다.

> 🤔 **표현력, 문맥 정보가 많이 줄지 않을까?**
>
> - multi-head는 각각의 Q, K, V로 토큰의 문맥 정보를 여러 관점에서 보는 것인데
> - K/V가 하나뿐이라면 문맥을 보는 관점이 줄어들어 표현력이 떨어지는 게 아닐까?
{: .prompt-tip }

## 이후의 흐름: GQA와 MLA

현재 MQA 자체는 거의 쓰이지 않고, 대신 이를 보완한 GQA를 사용한다.

**MHA**

![Self-Attention을 여러 head로 나누고 각 head가 Queries, Keys, Values를 따로 갖는 MHA 구조](/assets/img/posts/mqa/05_mha_heads.png){: w="837" h="517" }

**Grouped Query Attention**

![Attention head 두 개씩 그룹을 이루어 Keys, Values를 공유하는 GQA 구조](/assets/img/posts/mqa/06_gqa_heads.png){: w="847" h="465" }

**→ MQA에서 조금 더 타협해서, 그룹을 만들어 그룹마다 K/V를 할당하자는 아이디어**

> **논문**: [*GQA: Training Generalized Multi-Query Transformer Models from Multi-Head Checkpoints*](https://arxiv.org/abs/2305.13245), 2023
{: .prompt-info }

- MQA와 GQA 사이에는 4년의 간격이 있다.
  - 2019년에는 모델 컨텍스트가 짧아서, KV cache 크기 `d × L` 중 L이 작았기 때문에 memory bound 이슈가 크지 않았다. 그래서 당시에는 MQA가 주목받지 못했다.
- 이후 모델 컨텍스트가 길어지면서 주목받았고, 이를 개선한 GQA가 나왔다.

---

최근에는 DeepSeek에서 발표한 **MLA(Multi-head Latent Attention)**를 사용하기도 한다.

> **논문**: [*DeepSeek-V2: A Strong, Economical, and Efficient Mixture-of-Experts Language Model*](https://arxiv.org/abs/2405.04434), 2024
{: .prompt-info }

![MHA, GQA, MQA, MLA에서 추론 중 캐시되는 Key, Value를 비교한 그림. MLA는 Key와 Value를 압축된 latent 벡터 하나로 저장하고 projection으로 복원한다](/assets/img/posts/mqa/07_deepseek_v2_mla.png){: w="2080" h="860" }
_출처: DeepSeek-V2 논문 Figure 3_

![MHA는 head마다 KV, GQA는 그룹끼리 KV 공유, MQA는 KV 하나, MLA는 압축된 latent c 하나를 저장하고 head마다 복원하는 비교](/assets/img/posts/mqa/08_mha_gqa_mqa_mla.png){: w="1050" h="768" }

- MLA는 compressed KV를 사용해 차원을 축소해서 저장하므로, 추론 시 memory bound를 줄인다.

## FlashAttention vs MQA

|                       | FlashAttention                                       | MQA                                            |
| --------------------- | ---------------------------------------------------- | ---------------------------------------------- |
| **무엇이 병목인가**   | N×N 어텐션 중간 행렬을 HBM에 썼다가 다시 읽는 왕복   | KV cache 자체가 너무 커서 읽는 데 시간이 걸림  |
| **해법**              | 타일링 + online softmax로 SRAM 안에서 계산 완료      | K/V head를 1개로 만들어 저장량을 h분의 1로     |
| **무엇을 줄이나**     | 메모리 **왕복 횟수**                                 | 읽어야 할 **데이터 양**                        |
| **원본 데이터 크기**  | 그대로                                               | **줄어듦**                                     |
| **모델 출력**         | **완전히 동일**                                      | 달라짐 (근사)                                  |
| **재학습 필요**       | 없음                                                 | **필요** (석사 단계에서는 실험하기 쉽지 않음)  |
| **품질 손실**         | 없음                                                 | 약간 (BLEU −0.2)                               |
| **효과가 나는 단계**  | 학습, prefill                                        | **decode**                                     |
| **decode 효과**       | 작음 (Q=1이라 N×N이 안 생김)                         | **큼 (12배)**                                  |
| **prefill 효과**      | **큼**                                               | 거의 없음                                      |
| **적용 계층**         | 커널 (CUDA)                                          | 아키텍처                                       |
