from __future__ import annotations

import re
import zipfile
from datetime import datetime
from pathlib import Path, PurePosixPath


TASK_DATASET_DIRS = {
    "detect": "detection",
    "segment": "segmentation",
}


def _newest_data_yaml(directory: Path) -> Path:
    candidates = [path for path in directory.rglob("data.yaml") if path.is_file()]
    if not candidates:
        raise FileNotFoundError(
            f"В папке датасета не найден data.yaml: {directory}\n"
            "Распакуйте ZIP с сайта сюда либо передайте --data путь к ZIP, папке или data.yaml."
        )
    return max(candidates, key=lambda path: path.stat().st_mtime_ns)


def _safe_extract_zip(archive_path: Path, destination: Path) -> None:
    destination_root = destination.resolve()
    with zipfile.ZipFile(archive_path) as archive:
        for member in archive.infolist():
            normalized = member.filename.replace("\\", "/")
            parts = PurePosixPath(normalized).parts
            if not parts or normalized.startswith("/") or ".." in parts:
                raise ValueError(f"Небезопасный путь в ZIP: {member.filename}")
            target = destination.joinpath(*parts).resolve()
            if target != destination_root and destination_root not in target.parents:
                raise ValueError(f"Файл ZIP выходит за папку датасета: {member.filename}")
        archive.extractall(destination)


def _archive_destination(dataset_root: Path, archive_path: Path) -> Path:
    safe_stem = re.sub(r"[^0-9A-Za-zА-Яа-я._-]+", "_", archive_path.stem).strip("._")
    return dataset_root / "imports" / (safe_stem or "dataset")


def _make_yaml_portable(data_yaml: Path) -> Path:
    """Исправляет старые экспорты с зависимым `path: .` и исключает test."""
    text = data_yaml.read_text(encoding="utf-8-sig")
    portable = re.sub(
        r"(?mi)^\s*path\s*:\s*(['\"]?)\.\1\s*(?:#.*)?\r?\n?",
        "",
        text,
    )
    portable = re.sub(r"(?mi)^\s*test\s*:.*\r?\n?", "", portable)
    if portable == text:
        return data_yaml.resolve()

    prepared = data_yaml.with_name("data.local.yaml")
    prepared.write_text(portable, encoding="utf-8", newline="\n")
    return prepared.resolve()


def resolve_dataset_yaml(
    project_root: Path,
    task: str,
    source: str | Path | None = None,
) -> Path:
    """Находит датасет или безопасно распаковывает ZIP в dataset/<task>/imports/."""
    if task not in TASK_DATASET_DIRS:
        raise ValueError(f"Неизвестная задача обучения: {task}")

    dataset_root = project_root / "dataset" / TASK_DATASET_DIRS[task]
    dataset_root.mkdir(parents=True, exist_ok=True)

    if source is None:
        return _make_yaml_portable(_newest_data_yaml(dataset_root))

    source_path = Path(source).expanduser()
    if not source_path.is_absolute():
        source_path = (Path.cwd() / source_path).resolve()
    if not source_path.exists():
        raise FileNotFoundError(f"Датасет не найден: {source_path}")

    if source_path.is_dir():
        return _make_yaml_portable(_newest_data_yaml(source_path))

    if source_path.suffix.lower() in {".yaml", ".yml"}:
        return _make_yaml_portable(source_path)

    if source_path.suffix.lower() != ".zip":
        raise ValueError(
            "Параметр --data должен указывать на ZIP, папку датасета или data.yaml."
        )

    destination = _archive_destination(dataset_root, source_path)
    if destination.exists():
        try:
            return _make_yaml_portable(_newest_data_yaml(destination))
        except FileNotFoundError as exc:
            raise FileExistsError(
                f"Папка импорта уже существует, но data.yaml в ней нет: {destination}"
            ) from exc

    destination.mkdir(parents=True)
    try:
        _safe_extract_zip(source_path, destination)
        return _make_yaml_portable(_newest_data_yaml(destination))
    except Exception:
        if destination.exists() and not any(destination.iterdir()):
            destination.rmdir()
        raise


def build_run_name(model: str, requested_name: str | None = None) -> str:
    label = requested_name or Path(str(model).replace("\\", "/")).stem or "train"
    safe_label = re.sub(r"[^0-9A-Za-zА-Яа-я._-]+", "_", label).strip("._")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"{timestamp}_{safe_label or 'train'}"


def print_training_paths(data_yaml: Path, run_dir: Path) -> None:
    print("\n" + "=" * 70)
    print(f"Датасет:             {data_yaml}")
    print(f"Папка этого запуска: {run_dir}")
    print(f"Лучшая модель:       {run_dir / 'weights' / 'best.pt'}")
    print(f"Последняя модель:    {run_dir / 'weights' / 'last.pt'}")
    print("=" * 70)
