import csv
from html import escape
from pathlib import Path

from matplotlib import font_manager
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.utils import ImageReader
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import Image, PageBreak, Paragraph, SimpleDocTemplate, Spacer, Table


root = Path(__file__).resolve().parent
results = root / 'results'

for name, weight in [('text', 'normal'), ('bold', 'bold')]:
    font = font_manager.FontProperties(family=['Arial', 'Liberation Sans', 'DejaVu Sans'], weight=weight)
    pdfmetrics.registerFont(TTFont(name, font_manager.findfont(font)))

body = ParagraphStyle('body', fontName='text', fontSize=11, leading=14, spaceAfter=9)
heading = ParagraphStyle('heading', parent=body, fontName='bold', keepWithNext=True, spaceBefore=6)
cell = ParagraphStyle('cell', parent=body, fontSize=10, leading=12, spaceAfter=0)
header = ParagraphStyle('header', parent=cell, fontName='bold')
story = []


def read_csv(name):
    with (results / name).open(newline='') as file:
        return list(csv.DictReader(file))


def p(text, style=body):
    story.append(Paragraph(escape(text), style))


def table(columns, rows, widths):
    cells = [[Paragraph(escape(str(value)), header) for value in columns]]
    cells += [[Paragraph(escape(str(value)), cell) for value in row] for row in rows]
    story.append(Table(cells, colWidths=widths, hAlign='LEFT', style=[
        ('GRID', (0, 0), (-1, -1), 0.4, colors.grey),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ('TOPPADDING', (0, 0), (-1, -1), 5),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 5),
    ]))
    story.append(Spacer(1, 10))


def figure(name, width=432):
    path = str(results / name)
    pixels_w, pixels_h = ImageReader(path).getSize()
    image = Image(path, width=width, height=width * pixels_h / pixels_w, hAlign='LEFT')
    story.extend([image, Spacer(1, 6)])


def fmt(value, digits=2):
    return f'{float(value):.{digits}f}'.replace('.', ',')


tune = read_csv('tuning.csv')
spec = read_csv('speculation.csv')
load = read_csv('load.csv')

p('ДЗ 6. Инференс Qwen3-8B', heading)
p('Замеры проведены на одной A100 80 ГБ PCIe с vLLM 0.22.0 и AIPerf 0.13.0. '
  'В Aviflow использованы отдельные окружения Python с PyTorch 2.11.0 и CUDA 13.0. '
  'Dockerfile из задания рассчитан на CUDA 12.9.')
p('Для всех запусков взяты одна ревизия Qwen/Qwen3-8B и 640 исходных запросов из задания. '
  'Токены идут стримом, thinking отключён, температура 0, seed 42 и лимит ответа 512 токенов. '
  'Запросы подаются по порядку с одинаковыми интервалами. Перед каждым замером очищается '
  'prefix cache, затем идут 40 запросов прогрева. В таблицах только фаза замера.')
p('1. Настройки vLLM при 120 RPM', heading)
p('Бейзлайн запущен с настройками движка по умолчанию, с V2 Model Runner и async scheduling. '
  'Задержка считается от отправки запроса до конца получения ответа. '
  'Ускорение S равно средней задержке бейзлайна, делённой на среднюю задержку конфигурации.')
changes = {
    'baseline': 'по умолчанию',
    'chunk1024': 'бюджет батча 1024 токена',
    'chunk4096': 'бюджет батча 4096 токенов',
    'seq32': 'до 32 последовательностей',
    'v1': 'V1, async выключен',
    'ngram': 'V1, async выключен, CPU ngram',
    'ngram_gpu': 'V1, async выключен, GPU ngram',
}
table(['Запуск', 'Что меняется', 'Mean, с', 'P90, с', 'S', 'Ошибки'], [
    [row['config'], changes[row['config']], fmt(row['mean_s'], 3), fmt(row['p90_s'], 3),
     fmt(row['speedup'], 3), row['errors']] for row in tune
], [67, 158, 54, 51, 47, 55])
base, chunk = tune[0], next(row for row in tune if row['config'] == 'chunk4096')
p(f'Минимальная средняя задержка осталась у бейзлайна, {fmt(base["mean_s"])} с. '
  f'У chunk4096 P90 чуть ниже ({fmt(chunk["p90_s"])} против {fmt(base["p90_s"])} с), '
  'а средняя выше на 12 мс. Это один замер на настройку, разницу в десятки миллисекунд '
  'трудно отделить от разброса. У seq32 один запрос оборвался с Connection reset by peer.')
p('На 120 RPM у бейзлайна очередь почти пустая. Изменения бюджета батча мало повлияли '
  'на время между токенами (около 18,5 мс). Основное время уходит на генерацию, '
  'средний ответ содержит около 420 токенов.')

story.append(PageBreak())
p('2. Спекулятивное декодирование', heading)
p('Для сравнения взяты V1 Model Runner и выключенный async scheduling в обоих запусках. '
  'CPU ngram выбран по меньшей средней задержке среди двух вариантов спекдека при 120 RPM. '
  'Он предлагает до пяти токенов по совпадениям n-грамм длины 3–5 в текущей последовательности. '
  'На каждой из шести нагрузок подано по 640 запросов.')
table(['RPM', 'P90 без, с', 'P90 ngram, с', 'Ускорение', 'Таймауты без / с'], [
    [row['rpm'], fmt(row['off_p90_s']), fmt(row['on_p90_s']),
     fmt(row['p90_speedup'], 3) if row['p90_speedup'] else '—',
     f'{row["off_errors"]} / {row["on_errors"]}'] for row in spec
], [59, 85, 93, 88, 107])
figure('speculation.png')
p('Пустые кружки отмечают прогоны с таймаутами. Их P90 относится только к успешным ответам. '
  'Ускорение по P90 посчитано для пар, где ответили на все запросы.')
at60, at120, at240 = spec[:3]
p(f'По P90 спекдек проиграл уже при {at60["rpm"]} RPM, хотя средняя там чуть ниже '
  f'({fmt(at60["on_mean_s"])} против {fmt(at60["off_mean_s"])} с). '
  f'При {at120["rpm"]} RPM средняя тоже выросла. Время между токенами выросло '
  f'с {fmt(at120["off_itl_mean_ms"])} до {fmt(at120["on_itl_mean_ms"])} мс при близкой длине ответов. '
  'Модель принимала около трети предложенных токенов, экономии от проверки пачки в этом замере не хватило.')
p(f'При {at240["rpm"]} RPM очередь с ngram доходила до {int(float(at240["on_waiting_max"]))} запросов '
  f'против {int(float(at240["off_waiting_max"]))} без него. На 360 RPM начались таймауты. '
  'На 720 RPM P90 у ngram выглядит чуть ниже, но у него 279 таймаутов против 57 без спекдека. '
  'При таком числе пропусков по одному P90 судить об ускорении нельзя.')

story.append(PageBreak())
p('3. Нагрузка при P90 ниже 15 секунд', heading)
p('Для load test оставлен бейзлайн, он показал лучшую среднюю задержку при 120 RPM. '
  'На каждой точке запросы подаются 300 секунд через load_test.sh, затем ожидаются оставшиеся ответы. '
  'После 120 и 240 RPM интервал уточнён с шагом 30 RPM. Для зачёта нужны P90 строго меньше 15 с, '
  'ноль ошибок и хотя бы 99% ожидаемых ответов.')
table(['RPM', 'Mean, с', 'P90, с', 'Успешно / план', 'Ошибки', 'SLA'], [
    [row['rpm'], fmt(row['mean_s']), fmt(row['p90_s']),
     f'{row["ok"]} / {5 * int(row["rpm"])}', row['errors'], 'да' if row['sla'] == 'True' else 'нет']
    for row in load
], [52, 70, 70, 112, 67, 61])
figure('load.png')
passed = max((row for row in load if row['sla'] == 'True'), key=lambda row: int(row['rpm']))
failed = next(row for row in load if row['sla'] == 'False')
p(f'Максимальная прошедшая нагрузка на этой сетке — {passed["rpm"]} RPM. '
  f'Получено {passed["ok"]} ответов, P90 {fmt(passed["p90_s"])} с. '
  f'Уже при {failed["rpm"]} RPM P90 вырос до {fmt(failed["p90_s"])} с. '
  f'Время между токенами выросло с {fmt(passed["itl_mean_ms"])} до '
  f'{fmt(failed["itl_mean_ms"])} мс при почти пустой очереди. '
  'На 240 RPM появилась очередь, и задержка выросла ещё сильнее.')
p('Граница для этой A100 и набора запросов лежит между 150 и 180 RPM по замерам с шагом 30 RPM. '
  'На каждой точке был один пятиминутный прогон.')

SimpleDocTemplate(str(root / 'report.pdf'), pagesize=A4, title='ДЗ 6. Инференс Qwen3-8B',
                  leftMargin=72, rightMargin=72, topMargin=60, bottomMargin=60).build(story)
