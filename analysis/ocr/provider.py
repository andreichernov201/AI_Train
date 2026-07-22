from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np

from ..models import OCRResult


class OCRProvider(ABC):
    @abstractmethod
    def recognize(self, crop: np.ndarray, preprocessing: str = "original") -> OCRResult:
        raise NotImplementedError

    @abstractmethod
    def status(self) -> dict[str, object]:
        raise NotImplementedError
