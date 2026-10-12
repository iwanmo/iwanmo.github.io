---
title: "[논문 정리] Transformer (NIPS 2017): Attention Is All You Need"
date: 2026-04-26 16:00:00 +0900
categories: [Paper Review, NLP]
tags: [transformer, attention, self-attention, multi-head-attention, nlp]
description: 트랜스포머 인코더의 구성 요소(입력 임베딩, positional encoding, layer norm, residual)와 self-attention, multi-head attention의 계산 과정을 정리했습니다.
math: true
image:
  path: /assets/img/posts/transformer/01_paper_figure1.png
  alt: 입력 임베딩과 positional encoding, Multi-Head Attention, Feed Forward, Add & Norm으로 이루어진 인코더와 디코더를 N번 쌓은 트랜스포머 전체 구조
---

> **논문**: Ashish Vaswani et al., [*Attention Is All You Need*](https://arxiv.org/abs/1706.03762), NIPS 2017
{: .prompt-info }

## 트랜스포머 전체 구조

![입력 임베딩과 positional encoding, Multi-Head Attention, Feed Forward, Add & Norm으로 이루어진 인코더와 디코더를 N번 쌓은 트랜스포머 전체 구조](/assets/img/posts/transformer/01_paper_figure1.png){: w="568" h="746" }
_출처: 논문 Figure 1_

**인코더 구성 요소**

- input embedding
- positional encoding
- residual connection
- layer norm
- multi-head attention
  - self-attention
- feed forward network
  - FC + ReLU + FC

## Input Embedding

Input Embedding이란 특정 토크나이저의 인덱스에 해당하는 벡터값이다.

```text
"I am a student"
   ↓ 토크나이저
[40, 939, 261, 6760]   ← 인덱스(정수)
   ↓ Embedding lookup
[[0.2, -0.5, ...], [0.8, 0.1, ...], ...]   ← 벡터(실수)
```

![OpenAI 토크나이저에 I am a student를 입력하면 토큰 4개, 토큰 ID 40, 939, 261, 6760으로 변환되는 화면](/assets/img/posts/transformer/02_tokenizer.png){: w="463" h="598" }
_[OpenAI Tokenizer](https://platform.openai.com/tokenizer)로 직접 확인한 결과_

## Positional Encoding

- Q, K, V로 이루어진 attention 구조는 순서와 상관없이 모든 토큰이 연결되어 있어서, 순서 정보 없이는 각 토큰의 위치를 알 수 없다.
  - 트랜스포머가 모든 토큰을 병렬로 처리하기 때문이다.
  - "I am a student"와 "student a am I"를 똑같이 본다.
- 그래서 positional encoding을 더해 위치 정보를 넣는다.
- 논문에서는 삼각함수(sin, cos)를 사용했는데, 아직 잘은 모르겠다.

## Layer Norm

- CNN에서 batch norm을 하는 것처럼,
- 각 층의 입력 분포가 너무 다르면 학습이 불안정해지니 이 범위를 안정화하는 것이다.
- 학습 속도도 빨라진다.

## Residual Connection

- 구조 그림에서 attention을 거치지 않고 바로 이어지는 연결(Add & Norm으로 들어가는 화살표)이다.
- 사용 이유
  - CNN의 ResNet에서 처음 나온 개념이다.
  - 기울기 소실을 줄일 수 있다.
  - 층이 아무것도 하지 않는 것(항등 함수)도 쉽게 표현할 수 있게 해준다.

## Self-Attention

> self-attention을 통해 각 단어들이 서로(Q, K)를 보면서 문맥을 파악하고, 이 attention matrix를 V와 행렬곱해서 임베딩을 업데이트한다.
{: .prompt-tip }

임베딩 벡터로부터 Query, Key, Value를 뽑아낸다.

### 1. Q, K, V 생성

입력 벡터를 활성화 함수가 없는 서로 다른 Linear Layer에 통과시킨다.

![입력 X의 각 토큰 벡터에 W_Q, W_K, W_V를 각각 곱해 Query, Key, Value 행렬을 만드는 과정](/assets/img/posts/transformer/03_qkv_projection.png){: w="1474" h="860" }

- 목적
  - 차원 축소
  - multi-head attention에서 공간을 나누기 위해
- 각 텐서의 역할 (아직 완전히 와닿지는 않음)
  - **Query**: 이 단어는 어떤 의미를 갖는지 → "나를 수식하는 게 무엇인지" 묻는 질문
  - **Key**: 단어를 설명하는 키 → "내가 수식한다", Q와 벡터가 유사하면 걸리는 키
  - **Value**: 이 단어가 다른 단어에게 전달할 실제 내용

### 2. 관계성 계산

![한 head 안에서 Q와 K 전치의 내적으로 점수 행렬을 만들고, √d_k로 나눈 뒤 행별 softmax로 attention weight를 구해 V와 곱해 출력을 만드는 과정](/assets/img/posts/transformer/04_attention_in_head.png){: w="1474" h="1400" }

**Score matrix**

- Q와 K 전치의 내적으로 단어 간 유사도 점수를 계산한다.
- 단어 간 유사도가 높으면 값이 커진다.
- 공분산 행렬 같은 느낌이다.

**Attention matrix**

$$
\mathrm{Attention}(Q, K, V) = \mathrm{softmax}\!\left(\frac{QK^\top}{\sqrt{d_k}}\right) V
$$

- score matrix에 행(row) 방향으로 softmax를 적용해 확률값으로 만든다.
- softmax 전에 $\sqrt{d_k}$로 나누는 이유
  - 값이 극단적으로 커지지 않게 스무딩하는 역할이다.

### 3. Attention Value 계산

- Attention matrix와 V 행렬을 행렬곱한다.
- 행렬곱에 의해 단어 간 상관관계가 가중치 비율만큼 반영된다.
- 결과적으로 문맥이 반영된 새로운 임베딩 벡터가 만들어진다.

## Multi-Head Attention

![Scaled Dot-Product Attention 내부 구조와, 이를 h개 병렬로 수행한 뒤 Concat과 Linear로 합치는 Multi-Head Attention 구조](/assets/img/posts/transformer/05_paper_figure2.png){: w="2302" h="1236" }
_출처: 논문 Figure 2_

- 기존 self-attention을 H개로 늘린 게 전부다.
  - H개의 각 attention matrix는 서로 다른 정보를 갖는다고 한다.
    - 예: 문법, 시간 순서, 대명사 관계 등
  - 여기서 궁금했던 점: CNN은 층이 깊어지면서 특성이 점점 세분화되는 게 이해됐는데, attention은 병렬이라 특성이 왜 나눠지는지 이해가 안 갔다.
  - 초기화의 영향이 전부일까?
- H개에서 나온 Attention Value를 이어 붙이고(concat), linear layer로 각 정보를 섞어준다.

![head 1부터 4까지의 attention 출력을 feature 차원으로 이어 붙인 뒤 W_O를 곱해 합치는 과정](/assets/img/posts/transformer/06_multihead_concat.png){: w="1474" h="1120" }

- 각 attention value를 더하지 않고 이어 붙인 다음 linear layer를 거치는 이유는, 정보가 그냥 섞이지 않고 가중치에 의해 섞이도록 하기 위해서다.
- 이 작업이 N번 반복되면서, 임베딩 값이 다른 임베딩에 의해 계속 업데이트된다.

→ 이렇게 나온 값은 각 단어 간의 문맥 정보를 나타낸다.
