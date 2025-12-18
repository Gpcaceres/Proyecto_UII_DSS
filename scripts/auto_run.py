"""Orquestador automático: convierte el JSON grande, ejecuta la búsqueda de hiperparámetros
y entrena el modelo final. Registra logs en logs/* y guarda artefactos en models/experiments.
Se reintenta cada paso hasta 3 veces con backoff exponencial.
"""
from __future__ import annotations
import subprocess
import time
import pathlib
import os
from datetime import datetime

ROOT = pathlib.Path(__file__).resolve().parents[1]
LOGS = ROOT / 'logs'
LOGS.mkdir(exist_ok=True)

PY = os.environ.get('PYTHON', 'python')
ENV = os.environ.copy()
ENV['PYTHONPATH'] = str(ROOT / 'src')


def run_cmd(cmd, stdout_path, stderr_path, retries=3, backoff=10):
    attempt = 0
    while attempt < retries:
        attempt += 1
        with open(stdout_path, 'ab') as out_f, open(stderr_path, 'ab') as err_f:
            print(f"[{datetime.now().isoformat()}] Ejecutando: {' '.join(cmd)} (intento {attempt})")
            proc = subprocess.Popen(cmd, stdout=out_f, stderr=err_f, env=ENV)
            rc = proc.wait()
        if rc == 0:
            return True
        else:
            print(f"Comando falló (rc={rc}), esperando {backoff} s antes de reintentar...")
            time.sleep(backoff)
            backoff *= 2
    return False


def main():
    # Step 1: convert
    convert_stdout = LOGS / 'convert.log'
    convert_stderr = LOGS / 'convert.err'
    # Prefer CSV input if present (faster to parse), otherwise fall back to JSON
    raw_csv = ROOT / 'data' / 'MSR_data_cleaned.csv'
    raw_json = ROOT / 'data' / 'MSR_data_cleaned.json'
    if raw_csv.exists():
        input_path = str(raw_csv)
    else:
        input_path = str(raw_json)
    # Run converter via script path (reliable in various environments)
    converter_script = str(ROOT / 'src' / 'secure_pipeline' / 'convert_bigvul.py')
    convert_cmd = [PY, converter_script, '--input', input_path, '--output', 'data/bigvul_pipeline.csv']
    ok = run_cmd(convert_cmd, convert_stdout, convert_stderr, retries=3, backoff=30)
    if not ok:
        print('La conversión falló después de reintentos; revisa logs/convert.*')
        return 1

    # Step 2: optimize
    optimize_stdout = LOGS / 'optimize.log'
    optimize_stderr = LOGS / 'optimize.err'
    optimize_cmd = [PY, 'scripts/optimize.py', '--dataset', 'data/bigvul_pipeline.csv', '--sample-size', '100000', '--n-iter', '80', '--out-model', 'models/security_classifier_opt.joblib']
    ok = run_cmd(optimize_cmd, optimize_stdout, optimize_stderr, retries=2, backoff=60)
    if not ok:
        print('La búsqueda de hiperparámetros falló; revisa logs/optimize.*')
        return 1

    # Step 3: final training (train.py) — reentrena sobre todo el CSV
    train_stdout = LOGS / 'train.log'
    train_stderr = LOGS / 'train.err'
    train_cmd = [PY, '-m', 'secure_pipeline.train', '--dataset', 'data/bigvul_pipeline.csv', '--model-path', 'models/security_classifier.joblib']
    ok = run_cmd(train_cmd, train_stdout, train_stderr, retries=1, backoff=60)
    if not ok:
        print('Entrenamiento final falló; revisa logs/train.*')
        return 1

    print('Pipeline completado correctamente. Artefactos en models/ y logs/.')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
