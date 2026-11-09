# HW5: RL для вызова функций

Модель - `Qwen/Qwen3.5-2B`. Данные для обучения - только `Team-ACE/ToolACE`.
Сравниваются обычный GRPO с точной наградой, GRPO с частичной наградой и Dr. GRPO с той же частичной наградой.
Все три запуска начинаются с исходных весов. SFT и другие обучающие датасеты не используются.

[Отчёт в PDF](report.pdf) содержит подготовку данных, награды, графики обучения, сравнение и разбор пяти ошибок.

| Запуск | BFCL overall | Изменение к base |
|---|---:|---:|
| Исходная модель | 68.93% | — |
| GRPO, точная награда | 70.53% | +1.60 п.п. |
| GRPO, составная награда | **78.93%** | **+10.00 п.п.** |
| Dr. GRPO, составная награда | 78.40% | +9.47 п.п. |

У GRPO с составной наградой 95% парный bootstrap CI прироста к base: `[4.80; 15.47]` п.п.
Его отличие от Dr. GRPO невелико: `+0.53` п.п., CI `[-3.07; 4.00]` п.п.
Точная награда лучше подавляет лишние вызовы, но ухудшает часть категорий с нужными вызовами.
Во всех трёх запусках выбран заранее заданный 300-й шаг; на каждую конфигурацию приходится один seed.

Среда экспериментов: Aviflow `students`, одна A100 80 GB, Python 3.12.16 и PyTorch 2.11.0 с CUDA 13.0.
Исходный Dockerfile задания сохранён отдельно; в Aviflow использовано изолированное окружение с теми же
основными версиями библиотек. CUDA toolkit приведён к 13.0, `causal-conv1d` 1.7.0 собран для `sm_80`.

## Воспроизведение

Нужны Linux, NVIDIA GPU с BF16 и Python 3.12. В `evaluate.py` используется синтаксис Python 3.12.
В репозитории уже лежат метрики и логи. Для нового обучения сначала скопируйте исходники
в пустую рабочую папку, находясь в `hw5_rlvr`. Так готовые результаты не будут приняты за новый запуск:

```bash
run_dir=$(mktemp -d /tmp/hw5-rlvr.XXXXXX)
cp *.py *.sh *.txt *.json *.jsonl Dockerfile "$run_dir/"
cd "$run_dir"
```

В папке сохранён исходный Dockerfile из задания:

```bash
docker build -t rlvr-homework .
docker run --rm -it --gpus device=0 --ipc=host \
  -v "$PWD":/workspace -v rlvr-hf-cache:/cache rlvr-homework
bash run_all.sh
```

Отдельные этапы:

```bash
source runtime.sh
python prepare.py
python reward.py
python evaluate.py Qwen/Qwen3.5-2B --output results/base
python validate.py Qwen/Qwen3.5-2B --name base

python train.py grpo_exact --steps 300
python export.py grpo_exact
python validate.py checkpoints/grpo_exact --name grpo_exact
python evaluate.py checkpoints/grpo_exact --output results/grpo_exact
```

Для двух остальных запусков имя заменяется на `grpo_shaped` или `dr_grpo_shaped`.
`run_all.sh` пропускает уже завершённые этапы. Обучение после прерывания продолжается с последнего
сохранённого checkpoint, если он есть. Если нужен новый запуск с другими настройками,
для него стоит выбрать отдельную папку, чтобы не смешать результаты.

`requirements.txt` фиксирует основные библиотеки. При установке в своё окружение
нужно передать `uv pip install --override numpy-override.txt -r requirements.txt`:
это то же переопределение NumPy, что и в Dockerfile задания. После установки PyTorch
нужен `causal-conv1d==1.7.0`, собранный с совместимым CUDA toolkit:

```bash
CAUSAL_CONV1D_FORCE_BUILD=TRUE MAX_JOBS=4 \
  uv pip install --no-build-isolation causal-conv1d==1.7.0
```

## Данные и настройки

- Seed: `42`.
- Rollout engine использует стандартный для одного процесса TRL 1.16.0 seed `0`.
  В финальном evaluator seed `42`, как в исходном скрипте задания.
- Модель: ревизия `15852e8c16360a2fea060d615a32b45270f8a8fc`.
- ToolACE: ревизия `6bda777c88d21e5a204703c1ee45597a8fa4f734`.
- Train/validation делятся по целым исходным диалогам до нарезки на префиксы.
- Цели из ToolACE используются только для вычисления награды сгенерированным ответам.
- Промпты длиннее 2048 токенов исключаются целиком. Токенизатор и chat template исходной модели сохраняются.
- Бюджет каждого RL-запуска - 300 optimizer steps, по 4 ответа на запрос.
- В оценке используется полный чекпоинт после `merge_and_unload`, без отдельного LoRA-адаптера.

`experiments.json` содержит гипотезы и правило выбора шага. Полные фактические параметры каждого запуска
сохраняются в `runs/<имя>/config.json`, версии библиотек и ресурсы - в `summary.json`.

Для пересборки таблиц, графиков и PDF по сохранённым результатам обучение запускать не нужно:

```bash
python analyze.py
python make_report.py
```

Для этих двух скриптов достаточно NumPy, pandas, Matplotlib и ReportLab из `requirements.txt`.
Текст разбора и выбранные ошибки хранятся в `results/analysis_notes.json` и `results/error_cases.json`.

## Файлы

- `prepare.py` - загрузка, преобразование диалогов, фильтрация, разбиение и EDA.
- `reward.py` - разбор ответов, обе награды и небольшая проверка на искусственных примерах.
- `train.py`, `export.py` - RL через TRL и слияние LoRA с исходными весами.
- `validate.py` - отдельная проверка на 192 отложенных префиксах ToolACE.
- `evaluate.py`, `bfcl.jsonl` - исходные файлы задания, без изменений.
- `results/<имя>/metrics.json`, `predictions.jsonl` - результаты неизменённого BFCL evaluator.
- `results/comparison.csv`, `resources.csv`, `bootstrap.json` - таблицы и интервалы; `results/figures/` - графики.
- `results/<имя>/validation_metrics.json`, `validation_predictions.jsonl` - диагностика на ToolACE.
- `results/data_manifest.json`, `split_ids.json`, `rejected.json` - версии, состав выборок, индексы и причины исключения.

Веса, скачанный датасет и промежуточные чекпоинты в Git не добавляются.
