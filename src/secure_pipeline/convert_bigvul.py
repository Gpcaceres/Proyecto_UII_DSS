from __future__ import annotations

import argparse
import pathlib
from typing import List, Optional

import pandas as pd

DEFAULT_RAW_PATH = pathlib.Path("../../data/MSR_data_cleaned.json")
DEFAULT_OUTPUT_PATH = pathlib.Path("../../data/bigvul_pipeline.csv")


def _normalize_language(value: Optional[str]) -> str:
    if value is None:
        return "unknown"
    lang = value.strip().lower()
    mapping = {
        "c++": "cpp",
        "c/c++": "cpp",
        "cpp": "cpp",
    }
    return mapping.get(lang, lang or "unknown")


def _safe_str(value) -> str:
    if value is None:
        return ""
    if isinstance(value, float):
        return "" if pd.isna(value) else str(value)
    return str(value)


def _build_records(df: pd.DataFrame) -> List[dict]:
    records: List[dict] = []
    for idx, row in df.iterrows():
        lang = _normalize_language(_safe_str(row.get("lang")))
        commit_id = _safe_str(row.get("commit_id")) or f"row{idx}"
        base_id = f"{commit_id}_{idx}"

        vul_flag = _safe_str(row.get("vul"))
        is_vulnerable = vul_flag not in {"", "0", "false", "False"}

        func_before = _safe_str(row.get("func_before")).strip()
        func_after = _safe_str(row.get("func_after")).strip()

        if func_before:
            records.append(
                {
                    "id": f"{base_id}_before",
                    "label": "vulnerable" if is_vulnerable else "seguro",
                    "language": lang,
                    "code": func_before,
                }
            )

        if func_after:
            records.append(
                {
                    "id": f"{base_id}_after",
                    "label": "seguro",
                    "language": lang,
                    "code": func_after,
                }
            )
    return records


def _build_records_from_dicts(rows: List[dict], chunk_no: int = 0) -> List[dict]:
    """Build the same record structure from a list of plain dicts (faster for CSV chunks)."""
    records: List[dict] = []
    for idx, row in enumerate(rows):
        lang = _normalize_language(_safe_str(row.get("lang")))
        commit_id = _safe_str(row.get("commit_id")) or f"chunk{chunk_no}_row{idx}"
        base_id = f"{commit_id}_{idx}"

        vul_flag = _safe_str(row.get("vul"))
        is_vulnerable = vul_flag not in {"", "0", "false", "False"}

        func_before = _safe_str(row.get("func_before")).strip()
        func_after = _safe_str(row.get("func_after")).strip()

        if func_before:
            records.append(
                {
                    "id": f"{base_id}_before",
                    "label": "vulnerable" if is_vulnerable else "seguro",
                    "language": lang,
                    "code": func_before,
                }
            )

        if func_after:
            records.append(
                {
                    "id": f"{base_id}_after",
                    "label": "seguro",
                    "language": lang,
                    "code": func_after,
                }
            )
    return records


def _open_text(path: pathlib.Path):
    """Open path transparently if plain text or gz/bz2 compressed. Returns text file-like object."""
    import gzip
    import bz2
    # read magic bytes safely
    with path.open('rb') as fh:
        head = fh.read(4)
    if head.startswith(b'\x1f\x8b'):
        # gzip
        return gzip.open(path, mode='rt', encoding='utf-8', errors='ignore')
    if head.startswith(b'BZh'):
        # bzip2
        return bz2.open(path, mode='rt', encoding='utf-8', errors='ignore')
    # plain text
    return path.open('r', encoding='utf-8', errors='ignore')


def _iter_json_array(path: pathlib.Path):
    """Itera objetos dentro de un JSON que contiene un array grande sin cargar todo en memoria."""
    import json
    decoder = json.JSONDecoder()
    with _open_text(path) as f:
        buf = ''
        # Skip until first '['
        while True:
            chunk = f.read(65536)
            if not chunk:
                return
            buf += chunk
            idx = buf.find('[')
            if idx != -1:
                buf = buf[idx+1:]
                break
        # Now repeatedly decode JSON objects from buffer
        while True:
            # Attempt to find next object
            try:
                obj, offset = decoder.raw_decode(buf)
                yield obj
                buf = buf[offset:]
                # Skip possible commas and whitespace
                i = 0
                while i < len(buf) and buf[i] in ', \n\r\t]':
                    if buf[i] == ']':
                        return
                    i += 1
                buf = buf[i:]
            except ValueError:
                # Need more data
                more = f.read(65536)
                if not more:
                    return
                buf += more


def _iter_jsonl(path: pathlib.Path):
    import json
    with path.open('r', encoding='utf-8', errors='ignore') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except ValueError:
                # skip malformed lines
                continue


def _regex_stream_extractor(path: pathlib.Path, writer, seen, written):
    """Extract func_before/func_after blocks from a large text file using regex streaming.
    Returns the (possibly updated) written counter."""
    import re, hashlib
    func_before_re = re.compile(r'"func_before"\s*:\s*"((?:[^"\\]|\\.)*)"', re.DOTALL)
    func_after_re = re.compile(r'"func_after"\s*:\s*"((?:[^"\\]|\\.)*)"', re.DOTALL)
    vul_re = re.compile(r'"vul"\s*:\s*"([^"\\]*)"')
    commit_re = re.compile(r'"commit_id"\s*:\s*"([^"\\]*)"')

    buf = ''
    with _open_text(path) as f:
        while True:
            chunk = f.read(2_000_000)
            if not chunk:
                break
            buf += chunk
            # Keep buffer limited
            if len(buf) > 5_000_000:
                buf = buf[-3_000_000:]

            # Find all func_before matches
            for m in func_before_re.finditer(buf):
                code = m.group(1)
                # find nearest vul and commit_id before the match
                start = max(0, m.start()-1000)
                context = buf[start:m.start()]
                vul_m = vul_re.search(context)
                commit_m = commit_re.search(context)
                vul_flag = vul_m.group(1) if vul_m else ''
                commit_id = commit_m.group(1) if commit_m else f'row_regex_{m.start()}'
                is_vulnerable = vul_flag not in ('', '0', 'false', 'False')
                rec = {
                    'id': f'{commit_id}_rbefore_{m.start()}',
                    'label': 'vulnerable' if is_vulnerable else 'seguro',
                    'language': 'unknown',
                    'code': code,
                }
                key = (rec['label'], hashlib.md5(code.encode('utf-8',errors='ignore')).hexdigest())
                if key not in seen:
                    seen.add(key)
                    writer.writerow(rec)
                    written += 1

            # Find func_after matches
            for m in func_after_re.finditer(buf):
                code = m.group(1)
                start = max(0, m.start()-1000)
                context = buf[start:m.start()]
                commit_m = commit_re.search(context)
                commit_id = commit_m.group(1) if commit_m else f'row_regex_{m.start()}'
                rec = {
                    'id': f'{commit_id}_rafter_{m.start()}',
                    'label': 'seguro',
                    'language': 'unknown',
                    'code': code,
                }
                key = (rec['label'], hashlib.md5(code.encode('utf-8',errors='ignore')).hexdigest())
                if key not in seen:
                    seen.add(key)
                    writer.writerow(rec)
                    written += 1

    return written


def convert_dataset(raw_path: pathlib.Path = DEFAULT_RAW_PATH, output_path: pathlib.Path = DEFAULT_OUTPUT_PATH) -> pathlib.Path:
    if not raw_path.exists():
        raise FileNotFoundError(f"No se encontró el dataset original en {raw_path}")

    suffix = raw_path.suffix.lower()

    # Preparar salida
    output_path.parent.mkdir(parents=True, exist_ok=True)
    import csv
    import hashlib
    import sys
    # Raise CSV field size limit to handle very large code blobs in CSV fields
    try:
        csv.field_size_limit(10 * 1024 * 1024)  # 10MB
    except Exception:
        try:
            csv.field_size_limit(sys.maxsize)
        except Exception:
            pass

    seen = set()
    # If the output already exists with content, load existing keys and append (resume).
    header_bytes = len("id,label,language,code\n".encode('utf-8'))
    if output_path.exists() and output_path.stat().st_size > header_bytes:
        # Populate seen keys from existing output to avoid duplication
        with output_path.open('r', encoding='utf-8', newline='') as exist_f:
            rdr = csv.DictReader(exist_f)
            for r in rdr:
                code = r.get('code', '')
                key = (r.get('label', ''), hashlib.md5(code.encode('utf-8', errors='ignore')).hexdigest())
                seen.add(key)
        out_mode = 'a'
        write_header = False
    else:
        out_mode = 'w'
        write_header = True

    written = len(seen)
    with output_path.open(out_mode, encoding='utf-8', newline='') as out_f:
        writer = csv.DictWriter(out_f, fieldnames=["id", "label", "language", "code"])
        if write_header:
            writer.writeheader()

        if suffix in {".jsonl"}:
            iterator = _iter_jsonl(raw_path)
        elif suffix in {".json"}:
            # Try JSONL fast path: check first kilobyte for a '[' indicating array
            head = raw_path.open('r',encoding='utf-8',errors='ignore').read(1024)
            if head.lstrip().startswith('['):
                iterator = _iter_json_array(raw_path)
            else:
                iterator = _iter_jsonl(raw_path)
        else:
            # CSV path: stream rows using pandas in chunks
            # Read header first to detect available columns and only load necessary ones for speed
            cols = pd.read_csv(raw_path, nrows=0, engine='python').columns.tolist()
            needed_cols = [c for c in ['func_before', 'func_after', 'vul', 'commit_id', 'lang'] if c in cols]

            chunk_no = 0
            if needed_cols:
                try:
                    for chunk in pd.read_csv(raw_path, chunksize=1000, engine='python', usecols=needed_cols):
                        rows = chunk.to_dict(orient='records')
                        recs = _build_records_from_dicts(rows, chunk_no)
                        for rec in recs:
                            code = rec.get('code', '')
                            if not code:
                                continue
                            key = (rec.get('label', ''), hashlib.md5(code.encode('utf-8', errors='ignore')).hexdigest())
                            if key in seen:
                                continue
                            seen.add(key)
                            writer.writerow(rec)
                            written += 1
                            if written % 10000 == 0:
                                print(f'Conversión parcial: {written} registros escritos')
                        chunk_no += 1
                except csv.Error as e:
                    print(f'CSV parser failed with {e}; falling back to regex-based streaming extractor')
                    written = _regex_stream_extractor(raw_path, writer, seen, written)
            else:
                # fallback: read full rows if fields aren't present
                for chunk in pd.read_csv(raw_path, chunksize=1000, engine='python'):
                    rows = chunk.to_dict(orient='records')
                    recs = _build_records_from_dicts(rows, chunk_no)
                    for rec in recs:
                        code = rec.get('code', '')
                        if not code:
                            continue
                        key = (rec.get('label', ''), hashlib.md5(code.encode('utf-8', errors='ignore')).hexdigest())
                        if key in seen:
                            continue
                        seen.add(key)
                        writer.writerow(rec)
                        written += 1
                        if written % 10000 == 0:
                            print(f'Conversión parcial: {written} registros escritos')
                    chunk_no += 1
            return output_path

        # If JSON/JSONL iterator
        for idx, obj in enumerate(iterator):
            # Build records from the object (same logic as _build_records)
            # _build_records expects a DataFrame; implement inline to avoid overhead
            lang = _safe_str(obj.get("lang"))
            lang = _normalize_language(lang)
            commit_id = _safe_str(obj.get("commit_id")) or f"row{idx}"
            base_id = f"{commit_id}_{idx}"

            vul_flag = _safe_str(obj.get("vul"))
            is_vulnerable = vul_flag not in {"", "0", "false", "False"}

            func_before = _safe_str(obj.get("func_before")).strip()
            func_after = _safe_str(obj.get("func_after")).strip()

            if func_before:
                rec = {
                    "id": f"{base_id}_before",
                    "label": "vulnerable" if is_vulnerable else "seguro",
                    "language": lang,
                    "code": func_before,
                }
                code = rec['code']
                key = (rec['label'], hashlib.md5(code.encode('utf-8',errors='ignore')).hexdigest())
                if key not in seen:
                    seen.add(key)
                    writer.writerow(rec)
                    written += 1
                    if written % 10000 == 0:
                        print(f'Conversión parcial: {written} registros escritos')

            if func_after:
                rec = {
                    "id": f"{base_id}_after",
                    "label": "seguro",
                    "language": lang,
                    "code": func_after,
                }
                code = rec['code']
                key = (rec['label'], hashlib.md5(code.encode('utf-8',errors='ignore')).hexdigest())
                if key not in seen:
                    seen.add(key)
                    writer.writerow(rec)
                    written += 1
                    if written % 10000 == 0:
                        print(f'Conversión parcial: {written} registros escritos')

        # Fallback: if nothing was written, try a regex-based streaming extractor
        if written == 0:
            print('No se extrajeron registros con el parser JSON; usando extractor alternativo (regex)')
            import re
            # Patterns
            func_before_re = re.compile(r'"func_before"\s*:\s*"((?:[^"\\]|\\.)*)"', re.DOTALL)
            func_after_re = re.compile(r'"func_after"\s*:\s*"((?:[^"\\]|\\.)*)"', re.DOTALL)
            vul_re = re.compile(r'"vul"\s*:\s*"([^"\\]*)"')
            commit_re = re.compile(r'"commit_id"\s*:\s*"([^"\\]*)"')

            buf = ''
            with _open_text(raw_path) as f:
                while True:
                    chunk = f.read(2_000_000)
                    if not chunk:
                        break
                    buf += chunk
                    # Keep buffer limited
                    if len(buf) > 5_000_000:
                        buf = buf[-3_000_000:]

                    # Find all func_before matches
                    for m in func_before_re.finditer(buf):
                        code = m.group(1)
                        # find nearest vul and commit_id before the match
                        start = max(0, m.start()-1000)
                        context = buf[start:m.start()]
                        vul_m = vul_re.search(context)
                        commit_m = commit_re.search(context)
                        vul_flag = vul_m.group(1) if vul_m else ''
                        commit_id = commit_m.group(1) if commit_m else f'row_regex_{m.start()}'
                        is_vulnerable = vul_flag not in ('', '0', 'false', 'False')
                        rec = {
                            'id': f'{commit_id}_rbefore_{m.start()}',
                            'label': 'vulnerable' if is_vulnerable else 'seguro',
                            'language': 'unknown',
                            'code': code,
                        }
                        key = (rec['label'], hashlib.md5(code.encode('utf-8',errors='ignore')).hexdigest())
                        if key not in seen:
                            seen.add(key)
                            writer.writerow(rec)
                            written += 1
                            if written % 10000 == 0:
                                print(f'Conversión parcial: {written} registros escritos')

                    # Find func_after matches
                    for m in func_after_re.finditer(buf):
                        code = m.group(1)
                        start = max(0, m.start()-1000)
                        context = buf[start:m.start()]
                        commit_m = commit_re.search(context)
                        commit_id = commit_m.group(1) if commit_m else f'row_regex_{m.start()}'
                        rec = {
                            'id': f'{commit_id}_rafter_{m.start()}',
                            'label': 'seguro',
                            'language': 'unknown',
                            'code': code,
                        }
                        key = (rec['label'], hashlib.md5(code.encode('utf-8',errors='ignore')).hexdigest())
                        if key not in seen:
                            seen.add(key)
                            writer.writerow(rec)
                            written += 1
                            if written % 10000 == 0:
                                print(f'Conversión parcial: {written} registros escritos')

    # If we wrote nothing at all, raise to notify caller
    if written == 0:
        print('Conversión completada pero no se extrajeron registros.')
    else:
        print(f'Conversión completada: {written} registros escritos en {output_path}')

    return output_path


def main():
    parser = argparse.ArgumentParser(description="Convierte MSR_data_cleaned.csv al formato del pipeline")
    parser.add_argument("--input", type=pathlib.Path, default=DEFAULT_RAW_PATH, help="Ruta al CSV original")
    parser.add_argument(
        "--output", type=pathlib.Path, default=DEFAULT_OUTPUT_PATH, help="Ruta de salida para el CSV convertido"
    )
    args = parser.parse_args()

    output = convert_dataset(args.input, args.output)
    print(f"Dataset convertido guardado en {output}")


if __name__ == "__main__":
    main()
