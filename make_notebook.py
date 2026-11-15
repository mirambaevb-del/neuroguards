"""Generate the standalone Colab teaching notebook from this checked-in recipe."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import textwrap


def normalize_content(content: str) -> list[str]:
    """Keep the first left-aligned line, dedent the remaining template lines."""
    lines = content.strip("\n").splitlines()
    if lines and not lines[0].startswith((" ", "\t")):
        normalized = "\n".join([lines[0], textwrap.dedent("\n".join(lines[1:]))]).strip()
    else:
        normalized = textwrap.dedent(content).strip()
    return normalized.splitlines(True)


def markdown(content: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": normalize_content(content)}


def code(content: str) -> dict:
    return {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [],
            "source": normalize_content(content)}


cells = [
    markdown("""# NeuroGuard AI — финальный исследовательский ноутбук

    **Задача первой версии:** определить `INTERICTAL` или `ICTAL` по 4-секундным EEG-окнам. **Не прогнозирует приступы заранее**. Предсказание за 10–30 минут — отдельный будущий эксперимент.

    **Это исследовательский прототип, не медицинское устройство.** Не принимайте клинические решения по его результатам.

    Перед запуском: загрузите **всю папку** `NeuroGuard_Final` в `/content/NeuroGuard_Final` и подготовьте настоящие CHB-MIT EDF + `chbXX-summary.txt` в `/content/neuroguard_data`. Исходные файлы сохраняются без изменений.
    """),
    markdown("## Этап 1. Проверка окружения"),
    code("""import os, sys, platform, importlib.util
    from pathlib import Path
    print('Python:', sys.version)
    print('CPU:', os.cpu_count(), '| platform:', platform.platform())
    try:
        import psutil
        print('RAM GB:', round(psutil.virtual_memory().total / 1e9, 2))
    except ImportError:
        print('RAM: unavailable')
    for name in ['numpy','scipy','pandas','mne','sklearn','torch','fastapi']:
        spec = importlib.util.find_spec(name)
        if spec:
            try:
                module = __import__(name)
                print(name, getattr(module, '__version__', 'installed'))
            except Exception as exc: print(name, 'import error:', exc)
        else: print(name, 'NOT INSTALLED')
    try:
        import torch
        print('CUDA GPU available:', torch.cuda.is_available())
    except ImportError: print('PyTorch not needed for the baseline')
    """),
    markdown("## Этап 2. Установка необходимых библиотек"),
    code("""import os
    project = '/content/NeuroGuard_Final' if os.path.isdir('/content/NeuroGuard_Final') else os.getcwd()
    os.chdir(project)
    print('Project directory:', os.getcwd())
    assert os.path.isfile('requirements.txt'), 'Загрузите всю папку NeuroGuard_Final в Colab'
    %pip install -q -r requirements.txt
    """),
    markdown("## Этап 3. Проверка настоящих EDF и официальной разметки"),
    code("""from pathlib import Path
    from neuroguard_core import Config, discover_recordings
    from train import write_json
    data_dir = Path('/content/neuroguard_data')
    output_dir = Path('/content/NeuroGuard_Final') if Path('/content/NeuroGuard_Final').is_dir() else Path.cwd()
    audit = discover_recordings(data_dir, Config())
    write_json(output_dir/'results'/'dataset_audit.json', audit)
    print('Audit status:', audit['status'])
    for key in ['edf_files_found','edf_files_readable','patients','verified_seizures',
                'recording_hours','ictal_windows','interictal_windows',
                'boundary_windows_excluded','rejected_recordings']:
        print(key, ':', audit[key])
    """),
    markdown("""## Этап 4. Признаки EEG

    Данные: 8 заранее выбранных каналов; 4 секунды × 256 Hz = 1024 значения на канал. Фильтр 0.5–40 Hz, затем 20 признаков на канал (160 признаков). Фильтр действует на отдельное окно для одинакового поведения при обучении и инференсе; это ограничение для научного исследования.
    """),
    code("""from neuroguard_core import feature_names, iter_record_windows
    print('Number of features:', len(feature_names()))
    print('First 10 feature names:', feature_names()[:10])
    if audit['accepted_recordings']:
        try:
            feature_vector, label, info = next(iter_record_windows(audit['accepted_recordings'][0]))
            print('Verified real window:', label, info, 'features:', feature_vector.shape)
        except StopIteration: print('Record has no usable windows')
    else:
        print('No EDF: real feature demonstration is blocked; no fake signal is used here.')
    """),
    markdown("""## Этапы 5–6. Пациент-независимое обучение и честная оценка

    Train, validation и test формируются из непересекающихся пациентов **после аудита**. По validation выбираем параметры Random Forest или Logistic Regression, затем проверяем выбранную модель на независимом test. Если EDF или официальные summaries отсутствуют, обучение **не запускается**.
    """),
    code("""import subprocess, sys
    if audit['status'] == 'audited' and audit['accepted_recordings']:
        print('Real EEG available; starting real training...')
        subprocess.run([sys.executable, 'train.py', '--data', str(data_dir),
                        '--out', str(output_dir), '--train'], check=True)
    else:
        print('Real training SKIPPED: no verified EDF dataset. See results/dataset_audit.json')
    """),
    markdown("## Этап 7. Проверка инференса без нового обучения"),
    code("""from inference import NeuroGuardDetector
    model_path = output_dir/'model'/'seizure_detector.joblib'
    if model_path.exists():
        detector = NeuroGuardDetector(output_dir/'model')
        print('Loaded:', detector.manifest)
        test_patients = detector.manifest.get('held_out_cases', [])
        test_record = next((r for r in audit['accepted_recordings'] if r['patient'] in test_patients), None)
        if test_record:
            from neuroguard_core import load_edf, select_channels
            raw = load_edf(test_record['file'])
            try:
                real_samples = raw.get_data(picks=select_channels(raw.ch_names), start=0, stop=1024)
                print(detector.predict_array(real_samples, list(detector.config.channels), 256, unit='V')['windows'][0])
            finally:
                raw.close()
    else:
        print('No trained model. This is correct until the real training stage succeeds.')
    """),
    markdown("## Этап 8. Локальная проверка API (без публикации EEG)"),
    code("""from fastapi.testclient import TestClient
    from api import app
    with TestClient(app) as client:
        print('GET /health:', client.get('/health').json())
        print('GET /model-info status:', client.get('/model-info').status_code)
    """),
    markdown("## Этап 9. Графики настоящей оценки"),
    code("""from IPython.display import display, Image
    for filename in ['confusion_matrix.png','roc_curve.png','precision_recall_curve.png',
                     'eeg_with_annotations.png','model_comparison.png']:
        path = output_dir/'results'/filename
        if path.exists():
            print(filename)
            display(Image(filename=str(path)))
        else: print(filename, 'not generated without a real trained model')
    """),
    markdown("""## Этап 10. Подготовка отдельного Prediction-исследования

    Изучаем **возможность** отличать EEG за 10–30 минут до приступа от обычного interictal. Интервал 10–30 минут — только гипотеза. Предварительная разметка реализована в `prediction_research.py`; не используется в detection и не обучалась. Для серьёзной проверки требуются независимые пациенты, непрерывные записи, исключение postictal-периодов и отчёт о ложных тревогах в день.

    Результат текущего ноутбука — detection, **не** подтверждённое предсказание, раннее предупреждение или медицинская диагностика.

    Для следующих шагов см. `README.md` и `integration_guide.md`.
    """),
    code("""import subprocess, sys
    completed = subprocess.run([sys.executable, '-m', 'unittest', 'discover', '-s', 'tests', '-v'],
                               capture_output=True, text=True)
    print(completed.stdout[-2000:])
    print(completed.stderr[-3000:])
    print('Software test return code:', completed.returncode)
    """),
]

notebook = {
    "cells": cells,
    "metadata": {"colab": {"name": "NeuroGuard_AI_Final.ipynb", "provenance": []},
                 "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                 "language_info": {"name": "python"}},
    "nbformat": 4, "nbformat_minor": 4,
}

parser = argparse.ArgumentParser(description="Generate the NeuroGuard Colab notebook")
parser.add_argument(
    "--output",
    type=Path,
    default=Path(__file__).with_name("NeuroGuard_AI_Final.ipynb"),
    help="Output notebook path (defaults beside this script)",
)
args = parser.parse_args()
path = args.output
path.parent.mkdir(parents=True, exist_ok=True)
path.write_text(json.dumps(notebook, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
print(path)
