from __future__ import annotations

import re
import shutil
import tempfile
import zipfile
from datetime import datetime
from pathlib import Path, PurePosixPath


TASK_DATASET_DIRS = {
    "detect": "detection",
    "segment": "segmentation",
}


def _yaml_task(path: Path) -> str | None:
    name_match = re.fullmatch(r"data\.(detect|segment)\.ya?ml", path.name, re.IGNORECASE)
    if name_match:
        return name_match.group(1).lower()

    try:
        text = path.read_text(encoding="utf-8-sig")
    except (OSError, UnicodeError):
        return None
    task_match = re.search(
        r"(?mi)^\s*task\s*:\s*['\"]?(detect|segment)['\"]?\s*(?:#.*)?$",
        text,
    )
    return task_match.group(1).lower() if task_match else None


def _newest_data_yaml(directory: Path, task: str | None = None) -> Path:
    candidates = []
    for pattern in ("*.yaml", "*.yml"):
        for path in directory.rglob(pattern):
            if not path.is_file() or ".prepared" in path.parts:
                continue
            if not re.fullmatch(
                r"data(?:\.(?:detect|segment))?\.ya?ml",
                path.name,
                re.IGNORECASE,
            ):
                continue
            yaml_task = _yaml_task(path)
            if task is not None and yaml_task not in {None, task}:
                continue
            candidates.append(path)
    if not candidates:
        raise FileNotFoundError(
            f"В папке датасета не найден YAML для задачи {task or 'YOLO'}: {directory}\n"
            "Поддерживаются data.yaml, data.detect.yaml и data.segment.yaml. "
            "Распакуйте ZIP с сайта сюда либо передайте --data путь к ZIP, папке или YAML."
        )
    return max(
        candidates,
        key=lambda path: (
            int(_yaml_task(path) == task),
            path.stat().st_mtime_ns,
        ),
    )


def _link_or_copy_tree(source: Path, destination: Path) -> None:
    for source_file in source.rglob("*"):
        if not source_file.is_file():
            continue
        target = destination / source_file.relative_to(source)
        target.parent.mkdir(parents=True, exist_ok=True)
        try:
            target.hardlink_to(source_file)
        except OSError:
            shutil.copy2(source_file, target)


def _project_export_yaml(data_yaml: Path, task: str) -> Path:
    """Готовит YOLO-вид полного экспорта без изменения распакованной структуры."""
    export_root = data_yaml.parent
    image_root = export_root / "images"
    task_label_root = export_root / "labels" / task
    if not (
        (export_root / "project.json").is_file()
        and (image_root / "train").is_dir()
        and (task_label_root / "train").is_dir()
    ):
        return data_yaml

    prepared_parent = export_root / ".prepared"
    prepared_parent.mkdir(parents=True, exist_ok=True)
    prepared = prepared_parent / task
    temporary = Path(tempfile.mkdtemp(prefix=f".{task}-", dir=prepared_parent))
    try:
        splits = ["train"]
        val_images = image_root / "val"
        val_labels = task_label_root / "val"
        val_has_images = val_images.is_dir() and any(path.is_file() for path in val_images.rglob("*"))
        if val_has_images and val_labels.is_dir():
            splits.append("val")
        for split in splits:
            _link_or_copy_tree(image_root / split, temporary / "images" / split)
            _link_or_copy_tree(
                task_label_root / split,
                temporary / "labels" / split,
            )

        text = data_yaml.read_text(encoding="utf-8-sig")
        text = re.sub(
            r"(?mi)^\s*(?:path|train|val|test)\s*:.*\r?\n?",
            "",
            text,
        ).lstrip()
        val_path = "images/val" if "val" in splits else "images/train"
        (temporary / "data.yaml").write_text(
            f"train: images/train\nval: {val_path}\n{text}",
            encoding="utf-8",
            newline="\n",
        )
        if prepared.exists():
            shutil.rmtree(prepared)
        temporary.replace(prepared)
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    return (prepared / "data.yaml").resolve()


def _prepare_dataset_yaml(data_yaml: Path, task: str) -> Path:
    portable = _make_yaml_portable(data_yaml)
    return _project_export_yaml(portable, task)


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
    """Находит датасет в dataset/ или распаковывает ZIP в dataset/<task>/imports/."""
    if task not in TASK_DATASET_DIRS:
        raise ValueError(f"Неизвестная задача обучения: {task}")

    shared_dataset_root = project_root / "dataset"
    dataset_root = shared_dataset_root / TASK_DATASET_DIRS[task]
    dataset_root.mkdir(parents=True, exist_ok=True)

    if source is None:
        try:
            data_yaml = _newest_data_yaml(dataset_root, task)
        except FileNotFoundError:
            data_yaml = _newest_data_yaml(shared_dataset_root, task)
        return _prepare_dataset_yaml(data_yaml, task)

    source_path = Path(source).expanduser()
    if not source_path.is_absolute():
        source_path = (Path.cwd() / source_path).resolve()
    if not source_path.exists():
        raise FileNotFoundError(f"Датасет не найден: {source_path}")

    if source_path.is_dir():
        return _prepare_dataset_yaml(_newest_data_yaml(source_path, task), task)

    if source_path.suffix.lower() in {".yaml", ".yml"}:
        return _prepare_dataset_yaml(source_path, task)

    if source_path.suffix.lower() != ".zip":
        raise ValueError(
            "Параметр --data должен указывать на ZIP, папку датасета или data.yaml."
        )

    destination = _archive_destination(dataset_root, source_path)
    if destination.exists():
        try:
            return _prepare_dataset_yaml(_newest_data_yaml(destination, task), task)
        except FileNotFoundError as exc:
            raise FileExistsError(
                f"Папка импорта уже существует, но YAML датасета в ней нет: {destination}"
            ) from exc

    destination.mkdir(parents=True)
    try:
        _safe_extract_zip(source_path, destination)
        return _prepare_dataset_yaml(_newest_data_yaml(destination, task), task)
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
