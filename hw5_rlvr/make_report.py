import json
from html import escape

import pandas as pd
from matplotlib import font_manager
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image, PageBreak

from prepare import ROOT


for name, weight in [("text", "normal"), ("bold", "bold")]:
    path = font_manager.findfont(font_manager.FontProperties(family="DejaVu Sans", weight=weight))
    pdfmetrics.registerFont(TTFont(name, path))

body = ParagraphStyle("body", fontName="text", fontSize=9.5, leading=14, spaceAfter=8)
small = ParagraphStyle("small", parent=body, fontSize=8, leading=11, spaceAfter=4)
heading = ParagraphStyle("heading", fontName="bold", fontSize=15, leading=20, spaceAfter=12)
subheading = ParagraphStyle("subheading", fontName="bold", fontSize=10.5, leading=15, spaceBefore=8, spaceAfter=7)
story = []


def p(text, style=body):
    story.append(Paragraph(escape(str(text)).replace("\n", "<br/>"), style))


def h(text, main=False):
    p(text, heading if main else subheading)


def table(rows, widths):
    cells = [[Paragraph(escape(str(x)).replace("\n", "<br/>"), small) for x in row] for row in rows]
    tbl = Table(cells, colWidths=widths, repeatRows=1, hAlign="LEFT")
    tbl.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#e9eef3")),
        ("LINEBELOW", (0, 0), (-1, 0), .6, colors.HexColor("#87929e")),
        ("LINEBELOW", (0, -1), (-1, -1), .4, colors.HexColor("#b2bbc4")),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("LEFTPADDING", (0, 0), (-1, -1), 5), ("RIGHTPADDING", (0, 0), (-1, -1), 5),
        ("TOPPADDING", (0, 0), (-1, -1), 5), ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
    ]))
    story.extend([tbl, Spacer(1, 10)])


def figure(path, width):
    image = Image(str(path))
    image.drawHeight *= width / image.drawWidth
    image.drawWidth = width
    story.extend([image, Spacer(1, 8)])


def footer(canvas, doc):
    canvas.setFont("text", 8)
    canvas.setFillColor(colors.HexColor("#66717d"))
    canvas.drawRightString(A4[0] - 42, 25, f"HW5 | {doc.page}")


results = ROOT / "results"
quality = pd.read_csv(results / "comparison.csv", index_col=0)
resources = pd.read_csv(results / "resources.csv", index_col=0)
manifest = json.loads((results / "data_manifest.json").read_text())
bootstrap = json.loads((results / "bootstrap.json").read_text())
experiments = json.loads((ROOT / "experiments.json").read_text())
manual = json.loads((results / "error_cases.json").read_text())
notes = json.loads((results / "analysis_notes.json").read_text())
names = quality.index.tolist()
best = quality["overall"].idxmax()

h("RL для вызова функций", main=True)
p("Qwen3.5-2B / ToolACE / учебный срез BFCL")
p(notes["result"])
h("1. Данные и подготовка")
p("Для обучения использован только Team-ACE/ToolACE, ревизия " + manifest["dataset_revision"] + ". "
  "В исходных данных 11 300 диалогов. Сначала отложены 565 целых диалогов с seed 42, "
  "затем каждый диалог нарезан на префиксы до ответа ассистента после сообщения пользователя. "
  "Предыдущие ответы и результаты инструментов сохраняются в истории. Ответы сразу после tool-сообщений "
  "отдельными целями не становятся.")
p("Описания функций приведены к JSON Schema, недопустимые символы в именах заменены подчёркиваниями. "
  "Тот же словарь замен применяется к эталонам. Сохраняется указанное в диалоге текущее время. "
  "Промпт собирается исходным chat template с enable_thinking=False; сам эталон в промпт не добавляется.")
table([["Выборка", "Префиксы", "Диалоги", "С вызовом", "Без вызова"]] + [
    [split, s["rows"], s["dialogs"], s["kinds"]["call"], s["kinds"]["no_call"]]
    for split, s in manifest["splits"].items()
], [98, 85, 85, 112, 131])
p("Исключены 518 диалогов с альтернативным описанием инструментов, 49 с неразбираемым синтаксисом вызова "
  "и один с нестандартной разметкой ответа. Также убраны 5 пустых целей, 82 длинных префикса и 36 повторов "
  "промпта. Ограничение длины - 2048 токенов, длинные примеры исключаются целиком. Проверка точного "
  "нормализованного совпадения вопросов с BFCL пересечений не нашла. Идентификаторы и SHA256 сохранены.")
figure(results / "figures/eda.png", 511)
p("Рис. 1. Длины и число эталонных вызовов до фильтрации по длине и удаления повторов. "
  "Медиана промпта - 701 токен, 95-й процентиль - около 1404. Среди оставленных функций "
  f"{manifest['unique_tools_kept']} уникальных имён. "
  "Все эталонные аргументы проверены на соответствие схемам.", small)

story.append(PageBreak())
h("2. Награды и эксперименты", main=True)
p("Ответ 1. Точная награда прямо связана с задачей, но у неё нет промежуточного сигнала: одна ошибка в аргументе "
  "обнуляет весь ответ. Составная награда дополнительно поощряет правильный выбор функции и части аргументов. "
  "Это помогает различать ответы, когда ни один из четырёх ещё не полностью правильный.")
p("E - точное равенство мультимножеств полных вызовов, с учётом повторов и без учёта порядка. "
  "Числа 1 и 1.0 эквивалентны; у строк убираются только крайние пробелы. "
  "V - корректная разметка и соответствие вызовов схемам. Fₙ - F1 по мультимножеству имён функций. "
  "Fₐ - F1 по мультимножеству троек (функция, имя аргумента, значение). "
  "Если аргументов нет ни в эталоне, ни в ответе, Fₐ = Fₙ.")
p("R_exact = E\nR_shaped = 0.1V + 0.2Fₙ + 0.2Fₐ + 0.5E")
p("Обрезанный ответ даёт 0 в обеих наградах. Для задания с вызовом пустой ответ, неразобранная разметка или отсутствие вызова дают 0. "
  "Для задания без вызова обе награды равны 1 только при непустом корректном ответе без инструмента. "
  "Неполные генерации маскируются в policy loss. Используется тот же Qwen3CoderToolParser, что и в evaluator.")
p("У Fₐ есть ограничение: при нескольких вызовах одной функции частичная награда не различает перестановку "
  "аргументов между вызовами. Полное соответствие E при этом остаётся нулевым. "
  "Для ответов без вызова оценивается решение не вызывать инструмент, а не полезность текста.", small)
table([["Ответ при эталоне weather(city='Paris')", "R_exact", "R_shaped"],
       ["weather(city='Paris')", "1.0", "1.0"],
       ["weather(city='London')", "0.0", "0.3"],
       ["Незакрытый <tool_call>", "0.0", "0.0"],
       ["Текст без вызова", "0.0", "0.0"],
       ["Текст без вызова, когда эталон тоже без вызова", "1.0", "1.0"]], [331, 90, 90])
p("Это искусственные примеры для проверки функции награды; в обучение они не входят. "
  "Проверяются также повторы вызовов и разные функции без аргументов.", small)
p("Ответ 4. Если все четыре награды одинаковы, Rᵢ - mean(R) = 0. "
  "Advantage равен нулю и при делении на std + 10⁻⁴, и без деления в Dr. GRPO. "
  "Policy-часть от такого запроса не обучает модель. При β = 0.02 может оставаться градиент KL-штрафа к исходной политике.")

story.append(PageBreak())
h("3. Конфигурации и ресурсы", main=True)
table([["Запуск", "Изменение и гипотеза"]] + [
    [name, experiments["runs"][name]["hypothesis"]] for name in names if name != "base"
], [125, 386])
p(notes["training_setup"])
p("В GRPO advantage равен (Rᵢ - mean(R)) / (std(R) + 10⁻⁴), а токены усредняются внутри каждого ответа. "
  "В Dr. GRPO advantage = Rᵢ - mean(R); сумма по токенам делится на фиксированные 512 × число ответов. "
  "Эти две нормировки меняются вместе. Во всех вариантах ratio считается по токенам и используется "
  "симметричный clipping 0.2.")
table([["Запуск", "Шаг", "Время, с", "Обучаемых параметров", "Torch peak allocated, GiB"]] + [
    [name, int(row["steps"]), f"{row['seconds']:.1f}", f"{int(row['trainable_parameters']):,}".replace(",", " "),
     f"{row['peak_allocated_gib']:.2f}"] for name, row in resources.iterrows()
], [125, 45, 75, 140, 126])
p("Время включает инициализацию trainer и обучение. Память - максимум allocated в PyTorch, "
  "включая совместный процесс с rollout engine. Замер общего потребления через NVML отдельно не проводился.", small)
p("Все основные запуски обучаются с исходных весов и имеют одинаковый бюджет шагов и запросов. "
  "Короткие предварительные проверки по два шага используются только для проверки кода и памяти. "
  "Их веса не переносятся в основные эксперименты.")

story.append(PageBreak())
h("4. Результаты", main=True)
p("Оценка выполнена одним исходным evaluate.py на неизменённых 350 примерах. "
  "Токенизатор взят из фиксированной ревизии исходной модели. Для всех запусков: greedy, "
  "temperature=0, максимум 1024 новых токена, thinking отключён. LoRA перед оценкой объединена с базовыми весами.")
p("Overall усредняет пять групп с одинаковыми весами. Внутри Multiple и Irrelevance сначала усредняются "
  "их две категории. Simple, Parallel и Parallel multiple содержат по 50 примеров, Multiple - 150, Irrelevance - 50.")
cols = ["overall", "simple", "multiple", "parallel", "parallel_multiple", "irrelevance"]
table([["Запуск", "Overall", "Simple", "Multiple", "Parallel", "Parallel + multiple", "Irrel."]] + [
    [name] + [f"{quality.loc[name, c] * 100:.2f}" for c in cols] for name in names
], [117, 64, 60, 64, 62, 80, 64])
p("Таблица 1. BFCL, значения в процентах.", small)
table([["RL относительно base", "Разница, п.п.", "95% bootstrap CI, п.п."]] + [
    [x["run"], f"{x['delta_overall'] * 100:+.2f}", f"[{x['low'] * 100:+.2f}; {x['high'] * 100:+.2f}]"]
    for x in bootstrap if x["reference"] == "base"
], [201, 130, 180])
p("Парный bootstrap: 3000 пересэмплирований внутри семи исходных категорий с сохранением весов overall. "
  "Интервал отражает неопределённость этой небольшой тестовой выборки. Каждый RL-запуск выполнен с одним seed.", small)
val_rows = []
for name in names:
    val = json.loads((results / name / "validation_metrics.json").read_text())
    val_rows.append([name, f"{val['all']['exact'] * 100:.2f}", f"{val['all']['shaped']:.3f}",
                     f"{val['call']['exact'] * 100:.2f}", f"{val['no_call']['exact'] * 100:.2f}"])
table([["ToolACE validation", "Exact, %", "Shaped", "Call, %", "No call, %"]] + val_rows,
      [151, 90, 90, 90, 90])
p("Отдельная диагностическая выборка: 64 префикса без вызова, 64 с одним вызовом и 64 с несколькими. "
  "Они отложены по диалогам до обучения. Для основной серии заранее выбран последний, 300-й шаг.", small)
p(notes["strongest_change"])
p(notes["recommendation"])

story.append(PageBreak())
h("5. Ход обучения", main=True)
figure(results / "figures/training.png", 511)
p("Рис. 2. Для каждого запуска показаны policy loss с KL и сырая награда. Бледная линия - отдельные шаги, "
  "сплошная - среднее по 10 шагам. У GRPO и Dr. GRPO разная нормировка loss, его абсолютные значения напрямую не сравниваются.", small)
p(notes["reward_vs_bfcl"])

story.append(PageBreak())
h("6. Оставшиеся ошибки", main=True)
p("Ответ 5. Ниже разобраны ошибки лучшего наблюдаемого варианта: " + best + ". "
  "Полные ответы и результаты AST-проверки сохранены в predictions.jsonl. "
  "Запросы сокращены, вызовы показаны после разбора parser; текстовые ответы приведены фрагментами.")
for i, case in enumerate(manual):
    if i == 3:
        story.append(PageBreak())
        h("Оставшиеся ошибки: продолжение", main=True)
    h(f"{case['id']} / {case['category']}")
    p("Запрос (кратко): " + case.get("question_summary", case["question"]), small)
    p("Ожидалось: " + case["expected"], small)
    p("Получено: " + case["response_excerpt"], small)
    p(case["analysis"])
h("Ограничения и итог")
p(notes["limitations"])
p(notes["conclusion"])
h("Материалы")
for label, url in [
    ("Модель и её описание", "https://huggingface.co/Qwen/Qwen3.5-2B"),
    ("ToolACE", "https://huggingface.co/datasets/Team-ACE/ToolACE"),
    ("TRL GRPOTrainer 1.16.0", "https://github.com/huggingface/trl/blob/v1.16.0/trl/trainer/grpo_trainer.py"),
    ("Условие задания", "https://drive.google.com/file/d/1GCFSSygmlVZvdhiRckeBrAcYDKxhLiHv/view"),
]:
    story.append(Paragraph(f'<link href="{url}" color="#245b8c">{label}</link>', small))

doc = SimpleDocTemplate(str(ROOT / "report.pdf"), pagesize=A4, rightMargin=42, leftMargin=42,
                        topMargin=40, bottomMargin=42, title="HW5: RL для вызова функций", author="hxrxmx")
doc.build(story, onFirstPage=footer, onLaterPages=footer)
print(ROOT / "report.pdf")
