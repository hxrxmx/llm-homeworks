# Домашние работы по обучению LLM

В репозитории собраны работы по обучению и инференсу LLM.

| Задание | Что сделано | Основной результат |
|---|---|---|
| [HW1 — мини-претрейн](hw1/) | Обучение Qwen3 1B с нуля на русской Википедии | `eval_loss = 4.6805` |
| [HW2 — оптимизация и коммуникации](hw2/) | FlashAttention2, packing, padding-free и кольцевые collectives | 22 341 токен/с, ускорение 3.18x |
| [HW3 — SFT для function calling](hw3/) | Дообучение Qwen2.5 0.5B и агентный цикл для умного дома | `exact match = 0.90` на EN и RU |
| [HW4 — Full FT и LoRA](hw4_lora/hw4_lora.ipynb) | Qwen2.5 1.5B на AG News и четыре абляции LoRA | Full FT: 92.80%, LoRA all-linear: 93.82% accuracy |
| [HW5 — RL для вызова функций](hw5_rlvr/report.pdf) | Qwen3.5 2B, ToolACE, GRPO и Dr. GRPO | BFCL overall: 68.93% → 78.93% |
| [HW6 — инференс](hw6_inference/) | Настройки vLLM, спекулятивное декодирование и нагрузочный тест Qwen3-8B | 150 RPM при P90 14.25 с |

В папках заданий лежат отчёты, код и результаты запусков.
