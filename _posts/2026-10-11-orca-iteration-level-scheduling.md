---
title: "[논문 정리] Orca (OSDI '22): Iteration-level Scheduling과 Selective Batching"
date: 2026-10-11 16:30:00 +0900
categories: [Paper Review, LLM Serving]
tags: [orca, llm, inference, serving, batching, kv-cache]
description: 요청 단위가 아닌 iteration 단위로 스케줄링해 LLM 서빙의 지연과 처리량을 함께 개선한 Orca(OSDI '22)를 정리했습니다.
image:
  path: /assets/img/posts/orca/08_system_overview.png
  alt: Orca 시스템 구조. 스케줄러가 request pool에서 요청을 골라 실행 엔진에 1 iteration씩 넘기고 결과를 돌려받는다. 출처는 Orca 논문(OSDI '22)
---

> **논문**: Gyeong-In Yu et al., [*Orca: A Distributed Serving System for Transformer-Based Generative Models*](https://www.usenix.org/conference/osdi22/presentation/yu), OSDI '22
{: .prompt-info }

## Abstract

### 배경

- 트랜스포머 기반 모델을 서비스하기 위해 시스템 차원의 지원이 점점 더 필요해지고 있다.
- 모델은 다음 토큰을 autoregressive 방식으로 생성한다.
  - 즉, 모델을 한 번 반복(iteration)할 때마다 요청당 출력 토큰이 하나씩 생성된다.

### 문제

- 기존 시스템은 처리 중인 요청 배치를 바꿀 수 없는 경직된 스케줄링 구조다.
  - 여러 번 반복해야 하는 이런 작업에서는 성능이 좋지 않다.
  - 배치 안에서 다른 요청보다 먼저 끝난 요청도 클라이언트로 바로 반환할 수 없다.
  - 새로 도착한 요청은 현재 배치가 완전히 끝날 때까지 기다려야 한다.

### 아이디어

- **Iteration-level scheduling**
  - 실행을 요청 단위가 아닌 iteration 단위로 스케줄링하는 새로운 메커니즘이다.
  - 매 iteration마다 제어권이 스케줄러로 돌아온다.
  - 그래서 끝난 요청은 바로 응답하고, 새 요청은 바로 넣을 수 있다.
- **Selective batching**
  - 트랜스포머 모델의 배치 처리와 iteration-level scheduling을 함께 쓰기 위해, 선택한 일부 연산에만 배치를 적용한다.
  - Linear 연산처럼 가중치를 공유하는 연산은 배치로 묶고, attention은 배치에서 제외한다.
- 예시
  - A: "나는 오늘 아침에 밥을 먹고" → "학교에"
  - B: "날씨가 오늘 정말" → "좋다."
  - C: "오늘 저녁 메뉴는" → "치킨"

## 시각화: 기존 request-level 배칭의 KV 캐시

기존 request-level 배칭에서 KV 캐시가 iteration마다 어떻게 쌓이는지 보여주는 데모다. 재생 버튼이나 슬라이더로 iteration을 넘겨볼 수 있다.

<iframe src="{{ '/assets/demos/orca/request_level_kv_cache.html' | relative_url }}" title="Request-level 배칭의 KV 캐시 데모" loading="lazy" style="width:100%;height:900px;border:0;border-radius:12px;" onload="var f=this,b=f.contentDocument.body;new ResizeObserver(function(){f.style.height=b.scrollHeight+'px'}).observe(b)"></iframe>

[데모를 새 탭에서 열기]({{ '/assets/demos/orca/request_level_kv_cache.html' | relative_url }})

## 1. Introduction

기존 추론 엔진은 여러 클라이언트의 요청을 배치로 처리하는데, 요청마다 필요한 반복 횟수가 다르다는 문제가 있다.

> As different client requests may require different numbers of iterations for processing, requests that have finished earlier than others in the batch cannot return to the client, resulting in an increased latency.

- 즉, 요청마다 출력하는 토큰 길이가 다르다.
- 그래서 하나의 배치 안에서 낭비되는 부분이 생긴다.

논문은 이를 해결하기 위해 엔진 실행을 iteration 단위로 스케줄링하자고 제안한다.

> In this paper, we propose to schedule the execution of the engine at the granularity of iteration instead of request.

- 서빙 시스템은 엔진을 호출할 때 1 iteration 단위로 호출한다.

하지만 이 새로운 스케줄링 방식에는 문제가 생길 수 있다.

> Unlike the canonical request-level scheduling, the proposed scheduling can issue a batch of requests where each request has so far processed a different number of tokens. In such a case, the requests to the Transformer model cannot be processed in the batched manner because the attention mechanism calls for non-batchable tensor operations whose input tensors have variable shapes depending on the number of processed tokens.

- 즉, 여러 요청이 들어올 때 요청마다 지금까지 진행된 토큰 양이 다르다.
- 그렇기 때문에 attention 연산을 할 때 배치로 묶지 못할 수 있다.

## 2. Background & 3. Solution

Background에서는 주로 KV 캐시를 설명한다.

### Iteration-level scheduling

**문제: 하나의 배치에서 먼저 끝난 요청**

![기존 request-level 방식과 Orca iteration-level 방식에서 요청 A, B, C가 처리되는 타임라인 비교](/assets/img/posts/orca/01_request_finish_early.png){: w="1496" h="1048" }
_A, B 요청이 동시에 도착해 하나의 배치로 묶였다고 가정_

**해결**

- 실행을 요청 단위가 아닌 iteration 단위로 스케줄링한다.
- 매 iteration마다 제어권이 스케줄러로 돌아온다.
- 그래서 끝난 요청은 바로 응답하고, 새 요청을 넣을 수 있다.
### Selective batching

**기존의 요청 단위 배치**

![기존 방식 iter 2에서 요청 A, B의 Q와 K 전치 행렬이 같은 shape이라 batched matmul 한 번으로 처리되는 모습](/assets/img/posts/orca/02_request_level_batch.png){: w="1080" h="766" }

**Orca가 제안하는 iteration 단위 배치에서 문제가 되는 부분**

![Orca iter 3에서 요청 A와 C의 Q, K 전치 행렬 shape이 달라 batched matmul이 불가능한 모습](/assets/img/posts/orca/03_iteration_batch_problem.png){: w="1220" h="822" }

> 그럼 패딩을 넣어서 배치 처리하면 되지 않을까? 라는 생각이 들 수 있다.
> 결론부터 말하면 패딩된 빈칸은 연산과 메모리를 모두 낭비한다. 요청마다 진행된 토큰 양이 너무 다르기 때문이다.
{: .prompt-tip }

- 위 문제를 해결하기 위해 **selective batching**을 제안한다.
  - 선택된 연산 집합에 대해서만 배치를 적용한다.
  - 요청마다 KV라는 activation 값이 모두 다르므로, attention을 배치로 묶어도 GPU 메모리 읽기(캐싱) 측면에서 크게 얻는 효율이 없기 때문이다.

**Selective batching 구조**

![QKV linear와 out linear, MLP는 4개 토큰을 한 번에 배치 처리하고 attention만 요청 A와 C로 분리해 처리하는 흐름도](/assets/img/posts/orca/04_selective_batching.png){: w="1218" h="1034" }

- QKV linear 연산, MLP 연산은 배치로 처리한다.
- Attention은 요청별로 개별 처리한다.

## 4. Orca 디자인

Orca의 iteration-level 스케줄링을 기존 요청 단위 방식과 비교해 볼 수 있는 데모다. 상단 버튼으로 두 방식을 전환할 수 있다.

<iframe src="{{ '/assets/demos/orca/iteration_level_scheduling.html' | relative_url }}" title="Orca iteration-level 스케줄링 데모" loading="lazy" style="width:100%;height:900px;border:0;border-radius:12px;" onload="var f=this,b=f.contentDocument.body;new ResizeObserver(function(){f.style.height=b.scrollHeight+'px'}).observe(b)"></iframe>

[데모를 새 탭에서 열기]({{ '/assets/demos/orca/iteration_level_scheduling.html' | relative_url }})

### 4.1 Distributed Architecture

![Scheduler, Engine Master, 그리고 Controller와 GPU로 구성된 Worker들로 이루어진 Orca 시스템 구조](/assets/img/posts/orca/05_distributed_architecture.png){: w="624" h="344" }
_출처: Orca 논문 (OSDI '22)_

- 분산 GPU 처리
  - 간단하게 말하면, 매우 큰 모델은 GPU 하나에 들어가지 않으니 나눠서 병렬로 돌린다.
  - 모델이 175B라면 INT8이어도 175GB이고, KV cache까지 포함하면 그 이상으로 넉넉히 잡아야 한다.
    - GPU 하나의 메모리가 175GB가 될 수 없으니 병렬 처리하도록 나눈다.
  - 이런 개념은 기존 다른 논문에서도 동일하게 쓰던 것이다.
- 새로운 구조 제안
  - **Scheduler**: iteration을 관리한다.
  - **Engine Master**: worker와 scheduler 사이에서 전달만 한다.
  - **Control Plane**
    - Controller(CPU)가 GPU에 어떤 연산을 해야 하는지 전달한다.
    - 매 iteration마다 스케줄러와 엔진이 통신해야 하는데, 이 동기화를 control plane으로 분리해 통신 오버헤드를 줄인다.
    - gRPC로 통신한다.
  - **Data Plane**
    - GPU끼리 중간 텐서를 NCCL(NVIDIA의 GPU 간 통신 라이브러리)로 주고받는다.

즉, CPU와 GPU의 역할을 나눈 이기종 구조다.

- **GPU**: 대량의 행렬 연산과 텐서 전달 (data plane, NCCL)
- **CPU**: 작은 메타데이터 처리, 커널 실행 지시, 흐름 제어 (control plane, gRPC)

### 4.2 Scheduling Algorithm

스케줄러가 매 iteration마다 어떤 요청을 골라 처리할지 결정한다. 고른 요청들은 selective batching으로 실행된다.

- **FCFS (First Come First Served)**
  - request pool에 먼저 도착한 요청부터 고른다.
- **최대 배치 크기**
  - 관리자가 정한다.
  - 배치가 크면 처리량은 늘지만, iteration 하나가 오래 걸려 지연이 커진다.
  - 지연과 처리량 사이의 밸런스를 잡는 값이다.
- **KV cache 슬롯**
  - GPU 메모리에는 KV cache를 저장할 공간이 있다.
  - 새 요청을 배치에 넣을 때, 그 요청이 최대로 생성할 수 있는 길이(max tokens)만큼을 미리 예약한다.
  - 남은 슬롯이 부족하면 이번에는 그 요청이 들어가지 못한다.

> **미리 예약하는 이유**
>
> - 생성 도중 메모리가 바닥나면 요청을 멈추거나 쫓아내야 한다.
> - Orca는 이런 상황이 처음부터 생기지 않게 한다.
> - 안전하지만, 실제 요청은 max tokens보다 짧으니 예약 공간이 항상 놀게 된다.
>   - **이후 vLLM에서 이 예약 낭비 문제를 지적한다.**
{: .prompt-warning }

**Pipeline 병렬화**

![Orca 실행 파이프라인과 FasterTransformer 실행 파이프라인의 시간축 비교](/assets/img/posts/orca/06_pipeline_parallelism.png){: w="382" h="297" }
_출처: Orca 논문 (OSDI '22)_

- 기존 방식(b)과 Orca(a) 모두 여러 GPU로 나눠 파이프라인을 통해 병렬성을 얻을 수 있다.
- 하지만 Orca는 iteration 단위이기 때문에 더 많은 요청을 한 번에 처리할 수 있다.

**→ iteration 단위이기 때문에 동시에 파이프라인에 띄울 수 있는 배치가 여러 개이므로, 버블을 줄인다.**

- **기존 방식**: 스케줄링 단위가 요청 전체라서 배치도 요청들의 묶음으로 고정된다. 한 번 묶이면 가장 긴 요청이 끝날 때까지 기다린다.
- **Orca**: 요청을 iteration 단위로 잘게 쪼개서, 매 iteration마다 "지금 돌릴 수 있는 요청들"을 모아 배치를 만든다. 그래서 배치 구성을 매번 바꿀 수 있고, 여러 배치를 동시에 파이프라인에 띄울 수도 있다.

## 실험

![입력과 생성 길이가 32와 256일 때 FasterTransformer와 Orca의 처리량 대비 지연 그래프](/assets/img/posts/orca/07_experiment.png){: w="615" h="483" }
_출처: Orca 논문 (OSDI '22)_

- Orca는 처리량이 늘어도 지연이 거의 늘어나지 않는 구간이 있다.

## 정리

LLM 추론 최적화 관련 논문들을 각각 어떤 질문에 답하는지로 비교하면 다음과 같다.

|                | 질문                                                  | 레벨                 |
| -------------- | ----------------------------------------------------- | -------------------- |
| FlashAttention | attention 연산 한 번을 어떻게 메모리 효율적으로 계산할까 | 커널                 |
| MQA            | 모델 구조를 바꿔 K/V 자체를 얼마나 줄일까              | 모델 구조            |
| Orca           | 여러 요청을 언제 넣고 빼며 어떻게 묶어 돌릴까          | 시스템 (스케줄링)    |
| vLLM           | 여러 요청의 KV cache를 메모리에 어떻게 배치할까        | 시스템 (메모리 관리) |
