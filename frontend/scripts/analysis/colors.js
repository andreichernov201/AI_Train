import { COLORS } from "../app/colors.js";
import {
  DETECTION_CLASS_NAMES,
  SEGMENTATION_CLASS_NAMES,
} from "../app/train-classes.js";

function colorsByClass(classNames) {
  return Object.freeze(
    Object.fromEntries(
      classNames.map((name, classId) => [name, COLORS[classId % COLORS.length]])
    )
  );
}

export const ANALYSIS_DETECTION_COLORS = colorsByClass(DETECTION_CLASS_NAMES);
export const ANALYSIS_SEGMENTATION_COLORS = colorsByClass(SEGMENTATION_CLASS_NAMES);

export function analysisDetectionColor(className) {
  return ANALYSIS_DETECTION_COLORS[className] || COLORS[0];
}

export function analysisSegmentationColor(className) {
  return ANALYSIS_SEGMENTATION_COLORS[className] || COLORS[0];
}

export const ANALYSIS_MASK_STYLE = Object.freeze({
  fillAlpha: 0.24,
  strokeAlpha: 0.65,
  lineWidth: 1.5,
});
