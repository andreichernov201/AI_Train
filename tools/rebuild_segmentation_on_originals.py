#!/usr/bin/env python3
"""Пересобирает датасет YOLO-seg на исходных фотографиях RailGallery.

В старой папке ``yolo-seg`` нужны две группы:

* размеченные изображения с обычными именами;
* размеченные ``*_aug_1``. Их исходные кропы находятся среди изображений
  уникального размера в ``dataset/segmentation/images/train``.

Экспериментальные ``*_aug_0`` пропускаются. Скрипт находит кропы в исходной
коллекции, переносит преобразования и полигоны, затем объединяет кропы одной
фотографии в общий файл разметки.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import shutil
import sys
import tempfile
from collections import Counter
from dataclasses import dataclass, field, replace
from pathlib import Path

import cv2
import numpy as np


IMAGE_SUFFIXES = {".bmp", ".jpeg", ".jpg", ".png", ".tif", ".tiff", ".webp"}
SPLITS = ("train", "val", "test")
SPLIT_PRIORITY = {"train": 0, "val": 1, "test": 2}
MIN_PHOTOMETRIC_SCORE = 0.90


class RebuildError(RuntimeError):
    """Ошибка входных данных, сопоставления или записи результата."""


@dataclass(frozen=True)
class Features:
    points: np.ndarray
    descriptors: np.ndarray


@dataclass(frozen=True)
class OriginalImage:
    path: Path
    width: int
    height: int
    features: Features


@dataclass(frozen=True)
class SourceSample:
    split: str
    image_path: Path
    label_path: Path
    kind: str
    reference_path: Path
    source_to_reference: np.ndarray


@dataclass(frozen=True)
class Alignment:
    original: OriginalImage
    matrix: np.ndarray
    inliers: int
    inlier_ratio: float
    median_error: float
    coverage: float
    corners_inside: bool
    photometric_score: float


@dataclass
class Polygon:
    class_id: str
    points: np.ndarray
    source: str


@dataclass
class OutputItem:
    original: OriginalImage
    split: str
    polygons: list[Polygon] = field(default_factory=list)
    source_count: int = 0
    duplicate_count: int = 0


@dataclass
class OriginalCatalog:
    images: list[OriginalImage]
    matcher: cv2.FlannBasedMatcher
    index_by_path: dict[Path, int]


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Transfer non-augmented and *_aug_1 YOLO-seg polygons to original "
            "RailGallery photos; *_aug_0 is ignored."
        )
    )
    parser.add_argument(
        "--yolo-seg",
        type=Path,
        default=Path("yolo-seg"),
        help="Legacy annotated dataset with <split>/images and <split>/labels.",
    )
    parser.add_argument(
        "--reference-crops",
        type=Path,
        default=Path("dataset/segmentation/images/train"),
        help="Unaugmented reference crops corresponding to *_aug_1.",
    )
    parser.add_argument(
        "--originals",
        type=Path,
        action="append",
        default=None,
        help="Originals directory; repeat the option to search several directories.",
    )
    parser.add_argument(
        "--flat-originals",
        type=Path,
        action="append",
        default=None,
        help="Originals directory searched only at its top level; repeat as needed.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("dataset/segmentation_originals"),
        help="Destination YOLO dataset (created transactionally).",
    )
    parser.add_argument(
        "--feature-size",
        type=int,
        default=720,
        help="Maximum side used for SIFT indexing (default: 720).",
    )
    parser.add_argument(
        "--refine-candidates",
        type=int,
        default=64,
        help="Number of globally shortlisted originals to verify (default: 64).",
    )
    parser.add_argument(
        "--min-inliers",
        type=int,
        default=8,
        help="Minimum geometrically consistent feature matches (default: 8).",
    )
    parser.add_argument(
        "--min-inlier-ratio",
        type=float,
        default=0.10,
        help="Minimum RANSAC inlier ratio (default: 0.10).",
    )
    parser.add_argument(
        "--dedupe-iou",
        type=float,
        default=0.985,
        help="IoU at which same-class polygons are treated as duplicates.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace the output only after the entire rebuild succeeds.",
    )
    return parser.parse_args(argv)


def image_files(directory: Path) -> list[Path]:
    if not directory.is_dir():
        raise RebuildError(f"Directory does not exist: {directory}")
    return sorted(
        path
        for path in directory.rglob("*")
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
    )


def read_color(path: Path) -> np.ndarray:
    """Читает пути Windows с Unicode и учитывает ориентацию EXIF."""
    try:
        encoded = np.fromfile(path, dtype=np.uint8)
    except OSError as exc:
        raise RebuildError(f"Cannot read image {path}: {exc}") from exc
    image = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
    if image is None:
        raise RebuildError(f"Unsupported or damaged image: {path}")
    return image


def image_size(path: Path) -> tuple[int, int]:
    image = read_color(path)
    height, width = image.shape[:2]
    return width, height


def extract_features(
    image: np.ndarray,
    feature_size: int,
    max_features: int = 1000,
) -> Features | None:
    height, width = image.shape[:2]
    scale = min(1.0, feature_size / max(width, height))
    if scale < 1.0:
        image = cv2.resize(
            image,
            (max(2, round(width * scale)), max(2, round(height * scale))),
            interpolation=cv2.INTER_AREA,
        )
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    sift = cv2.SIFT_create(nfeatures=max_features)
    keypoints, descriptors = sift.detectAndCompute(gray, None)
    if descriptors is None or len(keypoints) < 3:
        return None
    points = np.float32([keypoint.pt for keypoint in keypoints]) / scale
    return Features(points, np.asarray(descriptors, dtype=np.float32))


def load_original_catalog(
    directories: list[Path], flat_directories: list[Path], feature_size: int
) -> OriginalCatalog:
    paths: list[Path] = []
    for directory in directories:
        paths.extend(image_files(directory))
    for directory in flat_directories:
        paths.extend(
            sorted(
                path
                for path in directory.iterdir()
                if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
            )
        )
    if not paths:
        raise RebuildError("No original images found")

    originals: list[OriginalImage] = []
    skipped: list[Path] = []
    for number, path in enumerate(paths, start=1):
        image = read_color(path)
        height, width = image.shape[:2]
        features = extract_features(image, feature_size)
        if features is None:
            skipped.append(path)
        else:
            originals.append(OriginalImage(path.resolve(), width, height, features))
        if number % 50 == 0 or number == len(paths):
            print(f"Indexed originals: {number}/{len(paths)}", file=sys.stderr, flush=True)

    if not originals:
        raise RebuildError("No usable visual features found in originals directories")
    if skipped:
        print(
            f"Warning: skipped {len(skipped)} featureless original(s)",
            file=sys.stderr,
            flush=True,
        )

    matcher = cv2.FlannBasedMatcher({"algorithm": 1, "trees": 5}, {"checks": 96})
    matcher.add([item.features.descriptors for item in originals])
    matcher.train()
    return OriginalCatalog(
        originals,
        matcher,
        {item.path: index for index, item in enumerate(originals)},
    )


def affine_matrix(matrix: np.ndarray) -> np.ndarray:
    result = np.eye(3, dtype=np.float64)
    result[:2, :] = matrix
    return result


def transform_points(points: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    transformed = cv2.perspectiveTransform(
        np.asarray(points, dtype=np.float32).reshape(1, -1, 2),
        np.asarray(matrix, dtype=np.float64),
    )[0]
    return np.asarray(transformed, dtype=np.float64)


def feature_pairs(query: Features, target: Features, ratio: float = 0.82) -> list[cv2.DMatch]:
    pairs = cv2.BFMatcher(cv2.NORM_L2).knnMatch(
        query.descriptors, target.descriptors, k=2
    )
    return [
        pair[0]
        for pair in pairs
        if len(pair) == 2 and pair[0].distance < ratio * pair[1].distance
    ]


def estimate_original_alignment(
    query_features: Features,
    query_size: tuple[int, int],
    original: OriginalImage,
) -> Alignment | None:
    good = feature_pairs(query_features, original.features)
    if len(good) < 3:
        return None
    source = np.float32([query_features.points[match.queryIdx] for match in good])
    destination = np.float32(
        [original.features.points[match.trainIdx] for match in good]
    )
    estimated, mask = cv2.estimateAffinePartial2D(
        source,
        destination,
        method=cv2.RANSAC,
        ransacReprojThreshold=5.0,
        maxIters=5000,
        confidence=0.997,
        refineIters=30,
    )
    if estimated is None or mask is None:
        return None
    matrix = affine_matrix(estimated)
    inlier_mask = mask.ravel().astype(bool)
    inliers = int(inlier_mask.sum())
    predicted = transform_points(source, matrix)
    errors = np.linalg.norm(predicted - destination, axis=1)
    median_error = float(np.median(errors[inlier_mask])) if inliers else math.inf

    query_width, query_height = query_size
    corners = np.float32(
        [[0, 0], [query_width, 0], [query_width, query_height], [0, query_height]]
    )
    mapped = transform_points(corners, matrix)
    tolerance = max(15.0, 0.02 * max(original.width, original.height))
    corners_inside = bool(
        np.all(mapped[:, 0] >= -tolerance)
        and np.all(mapped[:, 1] >= -tolerance)
        and np.all(mapped[:, 0] <= original.width + tolerance)
        and np.all(mapped[:, 1] <= original.height + tolerance)
    )
    determinant = abs(float(np.linalg.det(matrix[:2, :2])))
    coverage = determinant * query_width * query_height / (
        original.width * original.height
    )
    return Alignment(
        original=original,
        matrix=matrix,
        inliers=inliers,
        inlier_ratio=inliers / len(good),
        median_error=median_error,
        coverage=coverage,
        corners_inside=corners_inside,
        photometric_score=0.0,
    )


def photometric_similarity(
    query_image: np.ndarray,
    alignment: Alignment,
) -> float:
    try:
        inverse = np.linalg.inv(alignment.matrix)
    except np.linalg.LinAlgError:
        return -1.0

    height, width = query_image.shape[:2]
    original_image = read_color(alignment.original.path)
    warped = cv2.warpPerspective(
        original_image,
        inverse,
        (width, height),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
    )
    valid_mask = cv2.warpPerspective(
        np.ones(original_image.shape[:2], dtype=np.uint8),
        inverse,
        (width, height),
        flags=cv2.INTER_NEAREST,
    ) > 0
    if np.count_nonzero(valid_mask) < 0.90 * valid_mask.size:
        return -1.0

    left = cv2.GaussianBlur(cv2.cvtColor(query_image, cv2.COLOR_BGR2GRAY), (0, 0), 1)
    right = cv2.GaussianBlur(cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY), (0, 0), 1)
    left_values = left[valid_mask].astype(np.float64)
    right_values = right[valid_mask].astype(np.float64)
    left_values -= left_values.mean()
    right_values -= right_values.mean()
    denominator = np.linalg.norm(left_values) * np.linalg.norm(right_values)
    if denominator <= 1e-9:
        return -1.0
    return float(np.dot(left_values, right_values) / denominator)

def reliable(alignment: Alignment, min_inliers: int, min_ratio: float) -> bool:
    return bool(
        alignment.inliers >= min_inliers
        and alignment.inlier_ratio >= min_ratio
        and alignment.median_error <= 6.0
        and alignment.corners_inside
        and 0.05 <= abs(np.linalg.det(alignment.matrix[:2, :2])) <= 25.0
        and alignment.photometric_score >= MIN_PHOTOMETRIC_SCORE
    )


def alignment_rank(alignment: Alignment) -> tuple[float, int, float, float]:
    return (
        alignment.photometric_score,
        alignment.inliers,
        alignment.inlier_ratio,
        -alignment.median_error,
    )


def find_original(
    reference_path: Path,
    catalog: OriginalCatalog,
    feature_size: int,
    refine_candidates: int,
    min_inliers: int,
    min_ratio: float,
    known_originals: set[Path],
) -> Alignment:
    image = read_color(reference_path)
    height, width = image.shape[:2]
    query = extract_features(image, feature_size, max_features=1400)
    if query is None:
        raise RebuildError(f"Not enough visual features in: {reference_path}")

    known_matches: list[Alignment] = []
    for path in known_originals:
        index = catalog.index_by_path[path]
        candidate = estimate_original_alignment(query, (width, height), catalog.images[index])
        if candidate is not None:
            candidate = replace(
                candidate, photometric_score=photometric_similarity(image, candidate)
            )
            if reliable(candidate, min_inliers, min_ratio):
                known_matches.append(candidate)
    if known_matches:
        return max(known_matches, key=alignment_rank)

    votes: Counter[int] = Counter()
    for pair in catalog.matcher.knnMatch(query.descriptors, k=2):
        if len(pair) == 2 and pair[0].distance < 0.82 * pair[1].distance:
            votes[pair[0].imgIdx] += 1
    if not votes:
        raise RebuildError(f"No original candidates found for: {reference_path}")

    checked: list[Alignment] = []
    for index, _votes in votes.most_common(refine_candidates):
        candidate = estimate_original_alignment(query, (width, height), catalog.images[index])
        if candidate is not None:
            checked.append(
                replace(
                    candidate,
                    photometric_score=photometric_similarity(image, candidate),
                )
            )
    if not checked:
        raise RebuildError(f"No geometric match found for: {reference_path}")

    best = max(checked, key=alignment_rank)
    if not reliable(best, min_inliers, min_ratio):
        raise RebuildError(
            f"Unreliable match for {reference_path}: {best.original.path.name}, "
            f"inliers={best.inliers}, ratio={best.inlier_ratio:.3f}, "
            f"error={best.median_error:.2f}, photo={best.photometric_score:.3f}, "
            f"inside={best.corners_inside}"
        )
    return best


def estimate_augmented_transform(
    augmented_path: Path,
    reference_path: Path,
    feature_size: int,
) -> np.ndarray:
    augmented = read_color(augmented_path)
    reference = read_color(reference_path)
    height, width = augmented.shape[:2]
    if reference.shape[:2] != (height, width):
        raise RebuildError(
            f"aug_1/reference size mismatch: {augmented_path} and {reference_path}"
        )

    reference_features = extract_features(reference, feature_size, max_features=1800)
    if reference_features is None:
        raise RebuildError(f"Not enough features in aug_1 reference: {reference_path}")

    attempts: list[tuple[int, float, float, np.ndarray]] = []
    for flipped in (False, True):
        working = augmented[:, ::-1] if flipped else augmented
        source_features = extract_features(working, feature_size, max_features=1800)
        if source_features is None:
            continue
        good = feature_pairs(source_features, reference_features, ratio=0.84)
        if len(good) < 4:
            continue
        source = np.float32([source_features.points[item.queryIdx] for item in good])
        destination = np.float32(
            [reference_features.points[item.trainIdx] for item in good]
        )
        homography, mask = cv2.findHomography(
            source,
            destination,
            method=cv2.RANSAC,
            ransacReprojThreshold=5.0,
            maxIters=5000,
            confidence=0.997,
        )
        if homography is None or mask is None:
            continue
        inlier_mask = mask.ravel().astype(bool)
        inliers = int(inlier_mask.sum())
        predicted = transform_points(source, homography)
        errors = np.linalg.norm(predicted - destination, axis=1)
        error = float(np.median(errors[inlier_mask])) if inliers else math.inf
        if flipped:
            flip = np.array(
                [[-1.0, 0.0, width - 1.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
                dtype=np.float64,
            )
            homography = homography @ flip
        attempts.append((inliers, inliers / len(good), -error, homography))

    if not attempts:
        raise RebuildError(
            f"Cannot align augmented image {augmented_path} to {reference_path}"
        )
    inliers, ratio, negative_error, matrix = max(
        attempts, key=lambda item: item[:3]
    )
    if inliers < 8 or ratio < 0.20 or -negative_error > 6.0:
        raise RebuildError(
            f"Unreliable aug_1 alignment {augmented_path} -> {reference_path}: "
            f"inliers={inliers}, ratio={ratio:.3f}, error={-negative_error:.2f}"
        )
    return np.asarray(matrix, dtype=np.float64)


def collect_sources(
    yolo_seg: Path,
    reference_crops: Path,
    feature_size: int,
) -> tuple[list[SourceSample], int]:
    references = image_files(reference_crops)
    by_size: dict[tuple[int, int], Path] = {}
    duplicate_sizes: set[tuple[int, int]] = set()
    for path in references:
        size = image_size(path)
        if size in by_size:
            duplicate_sizes.add(size)
        by_size[size] = path.resolve()
    if duplicate_sizes:
        details = ", ".join(f"{w}x{h}" for w, h in sorted(duplicate_sizes))
        raise RebuildError(f"Reference crop sizes are not unique: {details}")

    samples: list[SourceSample] = []
    excluded_aug0 = 0
    used_references: set[Path] = set()
    for split in SPLITS:
        images_dir = yolo_seg / split / "images"
        labels_dir = yolo_seg / split / "labels"
        if not images_dir.is_dir() or not labels_dir.is_dir():
            continue
        for image_path in image_files(images_dir):
            stem = image_path.stem
            label_path = labels_dir / f"{stem}.txt"
            if re.search(r"_aug_0$", stem, flags=re.IGNORECASE):
                excluded_aug0 += 1
                continue
            if not label_path.is_file():
                raise RebuildError(f"Missing label for {image_path}: {label_path}")

            if re.search(r"_aug_1$", stem, flags=re.IGNORECASE):
                size = image_size(image_path)
                reference = by_size.get(size)
                if reference is None:
                    raise RebuildError(
                        f"No uniquely-sized unaugmented crop for {image_path} ({size[0]}x{size[1]})"
                    )
                if reference in used_references:
                    raise RebuildError(
                        f"Reference crop matched more than once: {reference}"
                    )
                used_references.add(reference)
                source_to_reference = estimate_augmented_transform(
                    image_path, reference, feature_size
                )
                kind = "aug_1"
            elif "_aug_" in stem.lower():
                raise RebuildError(f"Unknown augmentation suffix: {image_path.name}")
            else:
                reference = image_path.resolve()
                source_to_reference = np.eye(3, dtype=np.float64)
                kind = "base"

            samples.append(
                SourceSample(
                    split=split,
                    image_path=image_path.resolve(),
                    label_path=label_path.resolve(),
                    kind=kind,
                    reference_path=reference,
                    source_to_reference=source_to_reference,
                )
            )

    if not samples:
        raise RebuildError(f"No usable labels found in: {yolo_seg}")
    aug1_count = sum(sample.kind == "aug_1" for sample in samples)
    if aug1_count != len(references):
        raise RebuildError(
            f"Expected one aug_1 image per reference crop, got {aug1_count} aug_1 and "
            f"{len(references)} references"
        )
    return sorted(samples, key=lambda item: (SPLIT_PRIORITY[item.split], item.image_path.name)), excluded_aug0


def parse_polygons(sample: SourceSample, source_to_original: np.ndarray) -> list[Polygon]:
    image = read_color(sample.image_path)
    height, width = image.shape[:2]
    output: list[Polygon] = []
    for line_number, raw_line in enumerate(
        sample.label_path.read_text(encoding="utf-8-sig").splitlines(), start=1
    ):
        line = raw_line.strip()
        if not line:
            continue
        parts = line.split()
        if len(parts) < 7 or len(parts) % 2 == 0:
            raise RebuildError(
                f"Invalid YOLO-seg polygon at {sample.label_path}:{line_number}"
            )
        try:
            coordinates = np.asarray([float(value) for value in parts[1:]], dtype=np.float64)
        except ValueError as exc:
            raise RebuildError(
                f"Non-numeric coordinate at {sample.label_path}:{line_number}"
            ) from exc
        if np.any(coordinates < 0.0) or np.any(coordinates > 1.0):
            raise RebuildError(
                f"Coordinate outside [0, 1] at {sample.label_path}:{line_number}"
            )
        points = coordinates.reshape(-1, 2)
        points[:, 0] *= width
        points[:, 1] *= height
        mapped = transform_points(points, source_to_original)
        if not np.all(np.isfinite(mapped)):
            raise RebuildError(
                f"Non-finite transformed polygon at {sample.label_path}:{line_number}"
            )
        output.append(
            Polygon(parts[0], mapped, f"{sample.label_path}:{line_number}")
        )
    return output


def polygon_area(points: np.ndarray) -> float:
    return abs(float(cv2.contourArea(np.asarray(points, dtype=np.float32))))


def polygon_iou(left: np.ndarray, right: np.ndarray) -> float:
    all_points = np.vstack((left, right))
    minimum = np.floor(all_points.min(axis=0)).astype(int) - 2
    maximum = np.ceil(all_points.max(axis=0)).astype(int) + 2
    width, height = (maximum - minimum + 1).tolist()
    if width <= 0 or height <= 0:
        return 0.0
    left_mask = np.zeros((height, width), dtype=np.uint8)
    right_mask = np.zeros_like(left_mask)
    left_points = np.rint(left - minimum).astype(np.int32)
    right_points = np.rint(right - minimum).astype(np.int32)
    cv2.fillPoly(left_mask, [left_points], 1)
    cv2.fillPoly(right_mask, [right_points], 1)
    intersection = int(np.count_nonzero(left_mask & right_mask))
    union = int(np.count_nonzero(left_mask | right_mask))
    return intersection / union if union else 0.0


def add_polygon(item: OutputItem, polygon: Polygon, dedupe_iou: float) -> None:
    for index, existing in enumerate(item.polygons):
        if existing.class_id != polygon.class_id:
            continue
        if polygon_iou(existing.points, polygon.points) < dedupe_iou:
            continue
        if polygon_area(polygon.points) > polygon_area(existing.points):
            item.polygons[index] = polygon
        item.duplicate_count += 1
        return
    item.polygons.append(polygon)


def output_split(current: str, incoming: str) -> str:
    return min((current, incoming), key=lambda split: SPLIT_PRIORITY[split])


def classification(alignment: Alignment) -> str:
    return "original" if 0.90 <= alignment.coverage <= 1.10 else "crop"


def yaml_text(output: Path) -> str:
    try:
        configured = output.resolve().relative_to(Path.cwd().resolve()).as_posix()
    except ValueError:
        configured = output.resolve().as_posix()
    return (
        f"path: {configured}\n"
        "train: images/train\n"
        "val: images/val\n\n"
        "nc: 1\n"
        "names:\n"
        "  0: body\n"
    )


def write_dataset(
    output: Path,
    items: dict[Path, OutputItem],
    rows: list[dict[str, str]],
    summary: dict[str, object],
    overwrite: bool,
) -> None:
    output = output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists() and not overwrite:
        raise RebuildError(f"Output exists: {output} (use --overwrite to replace it)")

    temporary = Path(tempfile.mkdtemp(prefix=f".{output.name}-", dir=output.parent))
    filenames: dict[str, Path] = {}
    try:
        for split in SPLITS:
            (temporary / "images" / split).mkdir(parents=True, exist_ok=True)
            (temporary / "labels" / split).mkdir(parents=True, exist_ok=True)

        for original_path, item in sorted(items.items(), key=lambda pair: pair[0].name):
            key = original_path.name.casefold()
            collision = filenames.get(key)
            if collision is not None and collision != original_path:
                raise RebuildError(
                    f"Original filename collision: {collision} and {original_path}"
                )
            filenames[key] = original_path
            image_target = temporary / "images" / item.split / original_path.name
            label_target = (
                temporary / "labels" / item.split / original_path.with_suffix(".txt").name
            )
            shutil.copy2(original_path, image_target)
            lines: list[str] = []
            for polygon in item.polygons:
                points = polygon.points.copy()
                points[:, 0] = np.clip(points[:, 0], 0.0, item.original.width)
                points[:, 1] = np.clip(points[:, 1], 0.0, item.original.height)
                normalized = points / np.array(
                    [item.original.width, item.original.height], dtype=np.float64
                )
                values = " ".join(f"{value:.8f}" for value in normalized.ravel())
                lines.append(f"{polygon.class_id} {values}")
            label_target.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")

        (temporary / "data.yaml").write_text(yaml_text(output), encoding="utf-8")
        with (temporary / "matches.csv").open("w", newline="", encoding="utf-8-sig") as stream:
            fieldnames = [
                "status",
                "error",
                "split",
                "source",
                "source_kind",
                "reference_crop",
                "original",
                "reference_geometry",
                "coverage",
                "inliers",
                "inlier_ratio",
                "median_error",
                "photometric_score",
                "reused_original",
            ]
            writer = csv.DictWriter(stream, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
        (temporary / "summary.json").write_text(
            json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )

        if output.exists():
            shutil.rmtree(output)
        temporary.replace(output)
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise


def rebuild(args: argparse.Namespace) -> dict[str, object]:
    if args.feature_size < 256:
        raise RebuildError("--feature-size must be at least 256")
    if args.refine_candidates < 1:
        raise RebuildError("--refine-candidates must be at least 1")
    if args.min_inliers < 3:
        raise RebuildError("--min-inliers must be at least 3")
    if not 0.0 <= args.min_inlier_ratio <= 1.0:
        raise RebuildError("--min-inlier-ratio must be between 0 and 1")
    if not 0.0 <= args.dedupe_iou <= 1.0:
        raise RebuildError("--dedupe-iou must be between 0 and 1")

    yolo_seg = args.yolo_seg.resolve()
    references = args.reference_crops.resolve()
    original_args = args.originals or [
        Path(r"C:\Users\ROG\Desktop\railgallery_photos"),
        Path(r"C:\Users\ROG\Desktop\railgallery_photos_segmentation"),
        Path(r"C:\Users\ROG\Desktop\Новая папка (2)"),
    ]
    flat_original_args = args.flat_originals or [
        Path(r"C:\Users\ROG\Desktop\Новая папка (3)"),
    ]
    originals = [path.resolve() for path in original_args]
    flat_originals = [path.resolve() for path in flat_original_args]
    output = args.output.resolve()
    if output in {yolo_seg, references} or output in originals or output in flat_originals:
        raise RebuildError("Output must differ from every input directory")

    samples, excluded_aug0 = collect_sources(yolo_seg, references, args.feature_size)
    base_count = sum(sample.kind == "base" for sample in samples)
    aug1_count = sum(sample.kind == "aug_1" for sample in samples)
    print(
        f"Sources: {base_count} base + {aug1_count} aug_1; excluded {excluded_aug0} aug_0",
        file=sys.stderr,
        flush=True,
    )
    catalog = load_original_catalog(originals, flat_originals, args.feature_size)

    outputs: dict[Path, OutputItem] = {}
    known_originals: set[Path] = set()
    rows: list[dict[str, str]] = []
    geometry_counts: Counter[str] = Counter()
    unmatched_sources: list[str] = []
    match_cache: dict[Path, Alignment] = {}

    for number, sample in enumerate(samples, start=1):
        reused = False
        alignment = match_cache.get(sample.reference_path)
        if alignment is None:
            before = set(known_originals)
            try:
                alignment = find_original(
                    sample.reference_path,
                    catalog,
                    args.feature_size,
                    args.refine_candidates,
                    args.min_inliers,
                    args.min_inlier_ratio,
                    known_originals,
                )
            except RebuildError as exc:
                unmatched_sources.append(str(sample.image_path))
                rows.append(
                    {
                        "status": "unmatched",
                        "error": str(exc),
                        "split": sample.split,
                        "source": str(sample.image_path),
                        "source_kind": sample.kind,
                        "reference_crop": str(sample.reference_path),
                        "original": "",
                        "reference_geometry": "",
                        "coverage": "",
                        "inliers": "",
                        "inlier_ratio": "",
                        "median_error": "",
                        "photometric_score": "",
                        "reused_original": "false",
                    }
                )
                print(
                    f"Unmatched {number}/{len(samples)}: {sample.image_path.name}: {exc}",
                    file=sys.stderr,
                    flush=True,
                )
                continue
            reused = alignment.original.path in before
            match_cache[sample.reference_path] = alignment
        else:
            reused = True

        original_path = alignment.original.path
        known_originals.add(original_path)
        item = outputs.get(original_path)
        if item is None:
            item = OutputItem(alignment.original, sample.split)
            outputs[original_path] = item
        else:
            item.split = output_split(item.split, sample.split)
        item.source_count += 1

        source_to_original = alignment.matrix @ sample.source_to_reference
        for polygon in parse_polygons(sample, source_to_original):
            add_polygon(item, polygon, args.dedupe_iou)

        geometry = classification(alignment)
        geometry_counts[geometry] += 1
        rows.append(
            {
                "status": "matched",
                "error": "",
                "split": sample.split,
                "source": str(sample.image_path),
                "source_kind": sample.kind,
                "reference_crop": str(sample.reference_path),
                "original": str(original_path),
                "reference_geometry": geometry,
                "coverage": f"{alignment.coverage:.8f}",
                "inliers": str(alignment.inliers),
                "inlier_ratio": f"{alignment.inlier_ratio:.8f}",
                "median_error": f"{alignment.median_error:.4f}",
                "photometric_score": f"{alignment.photometric_score:.8f}",
                "reused_original": str(reused).lower(),
            }
        )
        print(
            f"Matched {number}/{len(samples)}: {sample.image_path.name} -> "
            f"{original_path.name} ({geometry}, inliers={alignment.inliers}, "
            f"coverage={alignment.coverage:.3f}, photo={alignment.photometric_score:.3f})",
            file=sys.stderr,
            flush=True,
        )

    if not outputs:
        raise RebuildError("No source image matched any originals collection")
    polygon_count = sum(len(item.polygons) for item in outputs.values())
    duplicate_count = sum(item.duplicate_count for item in outputs.values())
    split_counts = Counter(item.split for item in outputs.values())
    summary: dict[str, object] = {
        "included_base_images": base_count,
        "included_aug_1_images": aug1_count,
        "excluded_aug_0_images": excluded_aug0,
        "source_images": len(samples),
        "matched_source_images": len(samples) - len(unmatched_sources),
        "unmatched_source_images": unmatched_sources,
        "unique_originals": len(outputs),
        "output_polygons": polygon_count,
        "deduplicated_polygons": duplicate_count,
        "source_reference_geometry": dict(sorted(geometry_counts.items())),
        "output_splits": dict(sorted(split_counts.items())),
    }
    write_dataset(output, outputs, rows, summary, args.overwrite)
    return summary


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        summary = rebuild(args)
    except RebuildError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"Output: {args.output.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
