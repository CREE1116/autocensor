'use strict';

// Canonical labels. Every detection model maps its own class names onto these,
// so thresholds and censoring options are set once regardless of which models
// are running.
const CLASSES = [
  'nipple',
  'vagina',
  'penis',
  'testicles',
  'anus',
  'pubic hair',
  'breast',
  'cum',
  'cross-section',
  'x-ray',
  'female face',
  'male face',
];

const KO = {
  nipple: '유두/유륜',
  vagina: '여성기',
  penis: '남성기',
  testicles: '고환',
  anus: '항문',
  'pubic hair': '음모',
  breast: '가슴 전체',
  cum: '체액',
  'cross-section': '단면도',
  'x-ray': '투시도',
  'female face': '여성 얼굴',
  'male face': '남성 얼굴',
};

// `expand` scales the segmentation contour about its centre. Genital contours
// already cover the whole part, so they need only a small margin.
//
// `nipple` is the exception: no model here has an areola class, and an areola is
// a colour region whose size relative to the nipple varies per drawing. It gets
// a small contour margin plus `areola`, which finds the real boundary from the
// image itself. `fallbackExpand` applies only when that fit is rejected.
const DEFAULTS = {
  nipple: {
    enabled: true,
    threshold: 0.25,
    expand: 1.2,
    areola: {
      // 'ray'  - cast rays outward from the nipple, fit an ellipse to the edges
      // 'flood'- region-grow across non-skin pixels
      // 'off'  - contour only, plus `expand`
      method: 'off',
      maxScale: 5,
      tolerance: 0.45,
      minContrast: 14,
      rays: 180,
      smoothWindow: 9,
      minValidRays: 0.4,
      minGrowRatio: 1,
      inflate: 1.05,
      // flood-only guards
      leakLimit: 0.75,
      minRoundness: 0.45,
      fallbackExpand: 2.6,
    },
  },
  vagina: { enabled: true, threshold: 0.28, expand: 1.25 },
  penis: { enabled: true, threshold: 0.3, expand: 1.15 },
  testicles: { enabled: true, threshold: 0.3, expand: 1.15 },
  anus: { enabled: false, threshold: 0.3, expand: 1.4 },
  'pubic hair': { enabled: false, threshold: 0.35, expand: 1.1 },
  breast: { enabled: false, threshold: 0.4, expand: 1.0 },
  cum: { enabled: false, threshold: 0.35, expand: 1.0 },
  'cross-section': { enabled: false, threshold: 0.35, expand: 1.0 },
  'x-ray': { enabled: false, threshold: 0.35, expand: 1.0 },
  'female face': { enabled: false, threshold: 0.4, expand: 1.0 },
  'male face': { enabled: false, threshold: 0.4, expand: 1.0 },
};

function defaultLabelConfig() {
  return JSON.parse(JSON.stringify(DEFAULTS));
}

module.exports = { CLASSES, KO, DEFAULTS, defaultLabelConfig };
