'use strict';

/**
 * Clean, high-performance anime NSFW segmentation models.
 * `anime-medium` (1280px) is the primary recommended model covering 7 classes.
 * `anime-xl` (1280px) is the highest-capacity model for maximum detail and recall.
 * `anime-nano` (640px) is a lightweight fast alternative.
 *
 * `map` translates a model's own class names onto canonical labels in labels.js.
 */
const MODELS = {
  'anime-medium': {
    file: 'nsfw-anime-medium-x1280.onnx',
    size: 1280,
    task: 'segment',
    label: 'anime-medium (1280px, 정밀 · 추천)',
    classes: ['anus', 'nipple', 'penis', 'vagina', 'female face', 'male face', 'pubic hair'],
    map: {
      anus: 'anus',
      nipple: 'nipple',
      penis: 'penis',
      vagina: 'vagina',
      'female face': 'female face',
      'male face': 'male face',
      'pubic hair': 'pubic hair',
    },
  },
  'anime-xl': {
    file: 'nsfw-anime-xl-x1280.onnx',
    size: 1280,
    task: 'segment',
    label: 'anime-xl (1280px, 초대형 최고성능)',
    classes: ['anus', 'nipple', 'penis', 'vagina', 'female face', 'male face', 'pubic hair'],
    map: {
      anus: 'anus',
      nipple: 'nipple',
      penis: 'penis',
      vagina: 'vagina',
      'female face': 'female face',
      'male face': 'male face',
      'pubic hair': 'pubic hair',
    },
  },
  'anime-nano': {
    file: 'nsfw-anime-nano-x640.onnx',
    size: 640,
    task: 'segment',
    label: 'anime-nano (640px, 빠름)',
    classes: ['anus', 'nipple', 'penis', 'vagina', 'female face', 'male face', 'pubic hair'],
    map: {
      anus: 'anus',
      nipple: 'nipple',
      penis: 'penis',
      vagina: 'vagina',
      'female face': 'female face',
      'male face': 'male face',
      'pubic hair': 'pubic hair',
    },
  },
};

const DEFAULT_ENSEMBLE = ['anime-medium'];

module.exports = { MODELS, DEFAULT_ENSEMBLE };
