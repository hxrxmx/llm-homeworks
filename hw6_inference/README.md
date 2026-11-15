# ДЗ 6. Инференс Qwen3-8B

Из семи конфигураций на 120 RPM лучшую среднюю задержку показал дефолтный vLLM, 7,96 с. В пятиминутном тесте он прошёл SLA на 150 RPM с P90 14,25 с. На 180 RPM P90 вырос до 17,80 с. Таблицы, графики и разбор замеров в [отчёте](report.pdf).

Замеры сделаны на одной A100 80 ГБ PCIe в Aviflow. Движок vLLM 0.22.0 запущен в Python-окружении с PyTorch 2.11.0, Transformers 5.9.0 и CUDA 13.0. Клиент AIPerf 0.13.0 использовал отдельное окружение с Transformers 5.22.1. Dockerfile для воспроизведения использует образ vLLM 0.22.0 с CUDA 12.9.

## Как запустить

Скопируйте `Homework/data/requests.jsonl` из архива в `data/requests.jsonl` этой папки. В файле 640 запросов, SHA256 `67c48379e5ae2608f7ca08650fd346facdc01f766a3e8eecea6459d7cec00ebc`. В каждом запросе температура 0, seed 42, максимум 512 токенов, thinking отключён. Модель и токенизатор взяты из ревизии `b968826d9c46dd6066d109eabc6255188de91218`. [Архив задания](https://drive.google.com/file/d/1yb_6uAY-ydYOMGM3dq_HpbHekgFJbXMD/view).

```bash
docker build -t hw6-inference .
docker run --rm --gpus device=0 --ipc=host \
  -v "$PWD":/homework -v hw6-hf-cache:/root/.cache/huggingface \
  -e RESULTS=rerun hw6-inference python run.py
```

Для одиночного замера в том же окружении запустите сервер в одном терминале и клиент в другом.

```bash
bash configs.sh baseline
```

```bash
bash benchmark.sh rerun/baseline-120 120
bash load_test.sh 150 rerun/baseline-load-150
```

В `configs.sh` перечислены все настройки. `chunk1024` и `chunk4096` меняют `max_num_batched_tokens`, `seq32` ограничивает `max_num_seqs`. Пара со спекдеком и без него запускается командами `bash configs.sh ngram baseline` и `bash configs.sh v1 baseline`. В обоих случаях используется V1 Model Runner с выключенным async scheduling. Вариант `ngram_gpu` запускается так же, как `ngram`. Спекдек предлагает до пяти токенов по совпадениям n-грамм длины 3–5 из текущей последовательности.

`benchmark.sh` подаёт 640 запросов по порядку с заданным RPM. `load_test.sh` держит подачу 300 секунд и ждёт завершения запросов. На большой нагрузке строки файла идут по кругу. Кеш очищается перед прогревом и после замера. Первые 40 запросов идут в отдельную фазу прогрева. Каждый замер записывается в отдельную папку, а `run.py` подхватывает уже завершённые.

## Результаты

В `results/tune`, `results/spec` и `results/load` лежат агрегаты AIPerf, фазы замеров и серверные метрики. Сводки сохранены в [tuning.csv](results/tuning.csv), [speculation.csv](results/speculation.csv) и [load.csv](results/load.csv).

```bash
python -m pip install matplotlib reportlab
python analyze.py
python make_report.py
```
