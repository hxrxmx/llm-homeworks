import json
from html import escape
from io import BytesIO

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
import pandas as pd
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import Image, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

from prepare import ROOT


families = ["Arial", "Liberation Sans", "DejaVu Sans"]
for name, weight in [("text", "normal"), ("bold", "bold")]:
    path = font_manager.findfont(font_manager.FontProperties(family=families, weight=weight))
    pdfmetrics.registerFont(TTFont(name, path))

body = ParagraphStyle("body", fontName="text", fontSize=11, leading=12.65,
                      spaceAfter=8, allowWidows=0, allowOrphans=0)
heading = ParagraphStyle("heading", parent=body, fontName="bold", spaceBefore=6, keepWithNext=True)
cell = ParagraphStyle("cell", parent=body, spaceAfter=0)
story = []


def p(text, style=body):
    story.append(Paragraph(escape(str(text)).replace("\n", "<br/>"), style))


def h(text):
    p(text, heading)


results = ROOT / "results"
quality = pd.read_csv(results / "comparison.csv", index_col=0)
resources = pd.read_csv(results / "resources.csv", index_col=0)
manifest = json.loads((results / "data_manifest.json").read_text())
names = {"base": "Base", "grpo_exact": "GRPO exact", "grpo_shaped": "GRPO shaped",
         "dr_grpo_shaped": "Dr. GRPO"}

h("ДЗ 5. RL для вызова функций")
p("В работе сравниваются три варианта RL для Qwen/Qwen3.5-2B: GRPO с точной наградой, "
  "GRPO с составной наградой и Dr. GRPO с той же составной наградой. Все запуски начинаются "
  "с исходной модели. Для обучения используется только ToolACE, без SFT.")

h("Данные")
p("В Team-ACE/ToolACE 11 300 диалогов. Сначала 565 целых диалогов отложены в validation, "
  "а уже потом каждый диалог разбит на префиксы до ответа ассистента. Так части одного разговора "
  "не попадают одновременно в train и validation. Взяты ответы после сообщений пользователя; "
  "предыдущие сообщения и результаты инструментов остаются в истории. Эталон нужен только для награды, "
  "в запрос модели он не добавляется.")
train, val = manifest["splits"]["train"], manifest["splits"]["validation"]
p(f"После фильтрации осталось {train['rows']} примера для обучения и {val['rows']} для validation. "
  f"В train {train['kinds']['call']} примеров с вызовом и {train['kinds']['no_call']} без вызова. "
  "Медианная длина промпта до фильтрации - 701 токен, 95-й процентиль - около 1404. "
  "Примеры длиннее 2048 токенов, пустые ответы и повторы промптов убраны.")
p("Описания функций приведены к JSON Schema. Недопустимые символы в именах заменены подчёркиваниями, "
  "эти же замены применены к эталонам. Диалоги с неподдерживаемой схемой или неразбираемыми вызовами "
  "исключены. Аргументы оставленных эталонов проверены по схемам. Точных совпадений вопросов с BFCL "
  "после нормализации не найдено. Версии данных, состав сплитов и причины исключения сохранены в results.")
p("Для быстрой проверки на ToolACE взяты 192 отложенных префикса: 64 без вызова, "
  "64 с одним вызовом и 64 с несколькими. Они одинаковы для всех моделей.")

h("Обучение и награда")
p("Точная награда равна 1, если совпал весь набор вызовов и аргументов, иначе 0. "
  "Порядок вызовов не важен, повторы учитываются. Числа 1 и 1.0 считаются равными, "
  "у строк убираются крайние пробелы. Для составной награды добавлены частичные оценки:")
p("R_exact = E\nR_shaped = 0.1V + 0.2F_name + 0.2F_args + 0.5E")
p("Здесь E - точное совпадение, V - корректная разметка и соответствие схемам. "
  "F_name - F1 по именам функций, F_args - F1 по тройкам «функция, аргумент, значение», "
  "в обоих случаях с учётом повторов. Если аргументов нет с обеих сторон, F_args = F_name. "
  "Пустой, обрезанный или неправильно размеченный ответ получает 0. Если вызов нужен, "
  "а его нет, награда тоже 0. Если вызов не нужен, обе награды дают 1 за непустой "
  "ответ без вызова. Полезность самого текста при этом не оценивается. "
  "Обрезанные генерации также исключаются из policy loss.")
p("Например, при эталоне weather(city='Paris') такой же вызов получает 1 по обеим наградам. "
  "За weather(city='London') точная награда даёт 0, а составная - 0.3: функция и схема верны, "
  "но город другой. Незакрытый tool_call даёт 0. Это примеры проверки награды, в обучение они не входят. "
  "Для разбора используется тот же Qwen3CoderToolParser, что и в evaluate.py.")
p("В первом запуске проверяется, хватит ли награды за полное совпадение. Во втором меняется только награда: "
  "частичные оценки должны давать сигнал и тогда, когда ни один из четырёх ответов не полностью верен. "
  "В третьем проверяется Dr. GRPO. Он убирает деление advantage на std группы, а сумму по токенам "
  "делит на фиксированную длину 512, умноженную на число ответов. Это две совместные замены нормировки. "
  "У обычного GRPO токены сначала усредняются внутри каждого ответа.")
p("Каждый запуск обучается 300 шагов на одной A100 80 GB через TRL 1.16.0. "
  "Используется LoRA: r=16, alpha=32, dropout=0, проекции q/k/v/o, gate/up/down, "
  "in_proj_qkv/z/a/b и out_proj. Исходные веса BF16, адаптеры FP32; включены SDPA, TF32 и gradient checkpointing. "
  "Оптимизатор AdamW fused, lr=2e-5, warmup=10 шагов, затем постоянный lr. "
  "Weight decay=0.01, gradient clipping=0.3, beta=0.02, clipping ratio=0.2.")
p("Batch size - 8, gradient accumulation - 4, по 4 генерации на запрос. В сумме это 32 ответа "
  "на 8 запросов за шаг; num_iterations=1. При генерации temperature=0.8, top_p=0.95, top_k=0, "
  "максимум 512 новых токенов. Seed данных и Trainer - 42, seed rollout engine - 0, "
  "как по умолчанию в TRL для одного процесса. Для всех конфигураций заранее выбран последний, "
  "300-й шаг. Короткие пробные запуски использовались только для проверки кода и памяти.")
p("Обучаемых параметров в каждом варианте 16 819 200. Время с инициализацией trainer: "
  f"GRPO exact - {resources.loc['grpo_exact', 'seconds']:.1f} с, "
  f"GRPO shaped - {resources.loc['grpo_shaped', 'seconds']:.1f} с, "
  f"Dr. GRPO - {resources.loc['dr_grpo_shaped', 'seconds']:.1f} с. "
  "Пиковая allocated-память PyTorch - 35.70, 35.50 и 35.73 GiB соответственно, вместе с rollout engine. "
  "Это не замер всей памяти через NVML. По этим временам отдельно сравнивать скорость алгоритмов "
  "не стоит: в них входят инициализация и влияние кэшей.")

h("Результаты")
p("Модели оценены исходным evaluate.py на тех же 350 примерах BFCL. "
  "Параметры не менялись: greedy, temperature=0, максимум 1024 новых токена, thinking отключён, seed=42. "
  "Токенизатор исходный, LoRA перед оценкой объединена с весами модели. Overall - среднее по пяти группам; "
  "в Multiple и Irrelevance сначала усредняются их две категории. Поэтому overall отличается "
  "от обычной доли правильных ответов среди всех 350 примеров.")
cols = ["overall", "simple", "multiple", "parallel", "parallel_multiple", "irrelevance"]
rows = [["Модель", "Overall", "Simple", "Multiple", "Parallel", "Parallel +\nMultiple", "Irrelevance"]]
for name, row in quality.iterrows():
    rows.append([names[name]] + [f"{row[col] * 100:.2f}" for col in cols])
tbl = Table([[Paragraph(escape(str(x)).replace("\n", "<br/>"), cell) for x in row] for row in rows],
            colWidths=[87, 50, 50, 50, 50, 87, 65], repeatRows=1, hAlign="LEFT")
tbl.setStyle(TableStyle([
    ("GRID", (0, 0), (-1, -1), .5, colors.black),
    ("VALIGN", (0, 0), (-1, -1), "TOP"),
    ("LEFTPADDING", (0, 0), (-1, -1), 4), ("RIGHTPADDING", (0, 0), (-1, -1), 4),
    ("TOPPADDING", (0, 0), (-1, -1), 4), ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
]))
story.extend([tbl, Spacer(1, 8)])
p("В таблице значения в процентах. Лучший overall у GRPO shaped: 78.93% против 68.93% у base. "
  "Прирост - 10 п.п., 95% парный bootstrap-интервал [4.80; 15.47] п.п. Для bootstrap сделано 3000 "
  "пересэмплирований внутри семи исходных категорий с сохранением весов overall.")
p("Для этой задачи предпочтителен GRPO с составной наградой, хотя Dr. GRPO почти не отстаёт: "
  "разница 0.53 п.п., интервал [-3.07; 4.00] п.п. Уверенного различия между ними здесь нет. "
  "При этом Dr. GRPO лучше в Irrelevance: 68% против 60%. У точной награды всего +1.60 п.п. к base, "
  "а интервал [-2.93; 6.13] п.п. включает ноль. Base и GRPO exact правильно решили одинаковые "
  "251 из 350 примеров. Разный overall получился из-за разных весов групп.")
p("Ниже loss и reward всех трёх запусков, сглаженные по 10 шагам. У Dr. GRPO другая нормировка loss, "
  "поэтому меньшие значения сами по себе не означают лучшее обучение.")
plt.rcParams["font.family"] = font_manager.FontProperties(fname=font_manager.findfont(
    font_manager.FontProperties(family=families))).get_name()
fig, axes = plt.subplots(1, 2, figsize=(7, 2.4))
for name in list(names)[1:]:
    history = pd.read_json(ROOT / "runs" / name / "history.jsonl", lines=True)
    history = history.dropna(subset=["loss"]).drop_duplicates("step", keep="last").sort_values("step")
    for ax, col in zip(axes, ["loss", "reward"]):
        ax.plot(history["step"], history[col].rolling(10, min_periods=1).mean(), label=names[name], linewidth=1)
        ax.set(xlabel="step", ylabel=col)
        ax.tick_params(labelsize=8)
axes[0].legend(fontsize=8)
fig.tight_layout()
plot = BytesIO()
fig.savefig(plot, format="png", dpi=180)
plt.close(fig)
plot.seek(0)
story.extend([Image(plot, width=439, height=439 * 2.4 / 7), Spacer(1, 8)])

h("Ответы на вопросы")
p("1. У полного совпадения слишком мало промежуточного сигнала: "
  "одна ошибка в аргументе обнуляет ответ. Составная награда позволяет поощрить хотя бы правильную "
  "функцию и часть аргументов. Это согласуется с логами: доля групп с одинаковыми наградами "
  "у exact - 82.75%, у shaped - 75.54%. При этом частичная оценка аргументов не различает "
  "их перестановку между несколькими вызовами одной функции; полное совпадение E это различает.")
p("2. Больше всего помогла замена точной награды на составную: +8.40 п.п. overall при том же GRPO, "
  "интервал [3.87; 13.20] п.п. Веса на старте, данные, seed, LoRA, lr и число шагов одинаковые, "
  "меняется только награда. У Dr. GRPO одновременно меняются две нормировки, так что их отдельные "
  "эффекты по этому запуску разделить нельзя.")
p("3. Рост reward не всегда совпадает с BFCL. Средняя награда за первые и последние 20 шагов: "
  "exact - 0.627 и 0.723, shaped - 0.737 и 0.802, Dr. GRPO - 0.717 и 0.826. "
  "У Dr. GRPO при той же составной награде итоговый reward выше, но BFCL чуть ниже. "
  "При этом разница BFCL мала относительно интервала неопределённости. Награды разных форм "
  "напрямую сравнивать нельзя. Промежуточный BFCL не замерялся.")
p("На ToolACE validation точный exact match у GRPO exact - 77.60%, у shaped - 73.44%, "
  "у Dr. GRPO - 75.00%, у base - 63.02%. По BFCL порядок exact и shaped обратный. "
  "Одна из причин - разный состав оценки: в validation треть примеров без вызова, "
  "а у BFCL группа Irrelevance весит одну пятую. Exact лучше отказывается от лишних вызовов "
  "(84% против 60% в BFCL), зато shaped лучше в Parallel (80% против 48%).")
p("4. Если все ответы на запрос получили одинаковую награду, R_i - mean(R) = 0 и advantage равен нулю. "
  "Policy loss от этой группы не даёт обучающего сигнала. При beta=0.02 при этом может оставаться "
  "градиент KL-штрафа к исходной модели.")
p("5. У лучшего варианта GRPO shaped остались ошибки и в аргументах, и в решении о вызове. "
  "Пять примеров из predictions.jsonl:")
p("simple_150: для вклада под 3% в calculate_roi передано annual_interest_rate=3 вместо 0.03. "
  "Функция правильная, но процент не переведён в долю.")
p("multiple_76: на запрос о бронзовой скульптуре лошади модель перечисляет материалы текстом. "
  "Вызов sculpture.create_custom пропущен. Сам запрос можно прочитать и как вопрос о материалах, "
  "но эталон BFCL ожидает создание скульптуры.")
p("parallel_181: требовались цены Apple за неделю, Microsoft за две, Amazon за три и Tesla за месяц. "
  "Модель сделала четыре get_stock_prices, но в каждом передала сразу все компании. "
  "Число вызовов верно, соответствие компаний периодам потеряно.")
p("parallel_multiple_67: в math.roots.polynomial передан массив [6, -3, 2, -1, 1], "
  "а checker требует float: [6.0, -3.0, 2.0, -1.0, 1.0]. Числа по смыслу те же, но ответ отклонён. "
  "Это пример расхождения reward и BFCL: числовая нормализация в награде такую разницу не учитывает.")
p("irrelevance_233: на вопрос о переводе $20 в фунты вызвана weather_in_location для Лондона. "
  "Доступен только инструмент погоды, так что это лишний вызов: подходящей функции для запроса нет.")
p("В целом составная награда лучше подошла для нужных вызовов, а точная сильнее смещает модель "
  "к отказу от них. Результат относится к этим 350 примерам и одному seed на вариант. "
  "Bootstrap оценивает разброс тестовой выборки, но не разброс повторных обучений. "
  "Чтобы надёжнее сравнить GRPO shaped и Dr. GRPO, потребовались бы повторные запуски и отдельный тест.")

links = [
    ("модель", "https://huggingface.co/Qwen/Qwen3.5-2B"),
    ("ToolACE", "https://huggingface.co/datasets/Team-ACE/ToolACE"),
    ("TRL", "https://github.com/huggingface/trl/blob/v1.16.0/trl/trainer/grpo_trainer.py"),
    ("условие", "https://drive.google.com/file/d/1GCFSSygmlVZvdhiRckeBrAcYDKxhLiHv/view"),
]
story.append(Paragraph("Материалы: " + ", ".join(f'<link href="{url}">{label}</link>'
                                                for label, url in links) + ".", body))
doc = SimpleDocTemplate(str(ROOT / "report.pdf"), pagesize=A4, leftMargin=72, rightMargin=72,
                        topMargin=72, bottomMargin=72, title="ДЗ 5. RL для вызова функций", author="hxrxmx")
doc.build(story)
print(ROOT / "report.pdf")
