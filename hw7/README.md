# ДЗ 7. Квантизация и дистилляция

На запрос из MASSIVE `ru-RU` модель возвращает intent и список пар «тип слота, значение». Бейзлайн, квантизированные модели и ученики начинаются с `Qwen/Qwen3-1.7B`. Учитель — `Qwen/Qwen3-32B`. Ревизии моделей, MASSIVE и WikiText зафиксированы в коде. Seed 42, оценка на 256 одинаковых примерах из validation.

Quality на dev у бейзлайна 17,41%, у учителя 58,05%, у лучшего ученика 61,10%. В таблицах отчёта есть результаты всех пяти конфигураций квантизации и двух вариантов SFT. [Отчёт](report.pdf).

## Запуск

Из этой папки на CUDA GPU с 80 ГБ памяти.

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python experiment.py all
```

`experiment.py` сначала оценивает бейзлайн и учителя, затем собирает 2048 жадных ответов учителя на train и запускает два варианта full SFT. Для обучения берутся валидные ответы из начала этого списка. Ученик получает те же prompt и answer IDs, а токены промпта и паддинг исключены из loss. Лучшая модель по dev сохраняется в `artifacts_compression_hw/student_sft`.

Калибровка берёт по 128 примеров из MASSIVE train или WikiText-2 raw train. Фрагменты WikiText подрезаны по длинам запросов MASSIVE в токенах. В обоих случаях используется тот же chat template. У вариантов AWQ одинаковый W4A16, у двух GPTQ меняется список слоёв в BF16. У SmoothQuant α = 0,5 и W8A8. Квантизированные модели сохраняются и заново загружаются для оценки через Transformers и compressed-tensors.

В `artifacts_compression_hw/eval` есть метрики и ответы на dev по каждому запуску. В `training` — логи двух обучений, в `teacher_answers/answers.jsonl` — жадные ответы учителя. Идентификаторы dev записаны в `data_manifest.json`. У завершённых прогонов уже есть `metrics.json`, их запуск пропускает. Для повторного прогона с нуля задайте новую папку через `HW7_ARTIFACTS`.

```bash
python -m ipykernel install --sys-prefix --name hw7
python -m nbconvert --to notebook --execute homework_compression_student.ipynb \
  --ExecutePreprocessor.kernel_name=hw7 --ExecutePreprocessor.timeout=-1 \
  --output homework_compression_student.executed.ipynb
```

## Веса

Чекпоинты квантизации и выбранный `student_sft` собраны в части архивов в релизе. Команды выполняются из `hw7` и восстанавливают каталоги с весами в `artifacts_compression_hw`. [Чекпоинты](https://github.com/hxrxmx/llm-homeworks/releases/tag/hw7-checkpoints-v1).

```bash
mkdir -p .weights
while read -r digest file; do
  curl -fL "https://github.com/hxrxmx/llm-homeworks/releases/download/hw7-checkpoints-v1/$file" \
    -o ".weights/$file"
done < checkpoints.sha256
(cd .weights && sha256sum -c ../checkpoints.sha256) &&
for name in awq_massive awq_wikitext gptq_massive gptq_edges_bf16 smoothquant_massive student_sft; do
  cat .weights/"$name".tar.gz.part-* | tar -xzf - -C .
done
```

Чекпоинты получены квантизацией и дообучением Qwen3-1.7B. В каждом сохранён оригинальный Apache 2.0 LICENSE.

Источники. [MASSIVE, CC BY 4.0](https://huggingface.co/datasets/AmazonScience/massive), [WikiText](https://huggingface.co/datasets/Salesforce/wikitext), [Qwen3-1.7B, Apache 2.0](https://huggingface.co/Qwen/Qwen3-1.7B), [Qwen3-32B, Apache 2.0](https://huggingface.co/Qwen/Qwen3-32B).
