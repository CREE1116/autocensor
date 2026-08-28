'use strict';

const { contextBridge, ipcRenderer } = require('electron');

contextBridge.exposeInMainWorld('api', {
  meta: () => ipcRenderer.invoke('meta'),
  pickFolder: (title) => ipcRenderer.invoke('pick-folder', title),
  pickImage: () => ipcRenderer.invoke('pick-image'),
  openPath: (p) => ipcRenderer.invoke('open-path', p),
  preview: (payload) => ipcRenderer.invoke('preview', payload),
  batchStart: (opts) => ipcRenderer.invoke('batch-start', opts),
  batchCancel: () => ipcRenderer.invoke('batch-cancel'),
  onBatchEvent: (cb) => ipcRenderer.on('batch-event', (_e, ev) => cb(ev)),

  reviewLoad: (dir) => ipcRenderer.invoke('review-load', dir),
  reviewThumb: (payload) => ipcRenderer.invoke('review-thumb', payload),
  reviewOpen: (payload) => ipcRenderer.invoke('review-open', payload),
  reviewApply: (payload) => ipcRenderer.invoke('review-apply', payload),
  reviewDest: (file) => ipcRenderer.invoke('review-dest', file),
  datasetAdd: (payload) => ipcRenderer.invoke('dataset-add', payload),
  datasetStats: (dir) => ipcRenderer.invoke('dataset-stats', dir),

  modelsList: () => ipcRenderer.invoke('models-list'),
  trainCheck: (paths) => ipcRenderer.invoke('train-check', paths),
  trainDatasetReady: (dir) => ipcRenderer.invoke('train-dataset-ready', dir),
  pickWeights: () => ipcRenderer.invoke('pick-weights'),
  trainStart: (opts) => ipcRenderer.invoke('train-start', opts),
  trainCancel: () => ipcRenderer.invoke('train-cancel'),
  onTrainEvent: (cb) => ipcRenderer.on('train-event', (_e, ev) => cb(ev)),

  modelsStatus: () => ipcRenderer.invoke('models-status'),
  modelDownload: (payload) => ipcRenderer.invoke('model-download', payload),
  modelDownloadCancel: () => ipcRenderer.invoke('model-download-cancel'),
  modelDelete: (key) => ipcRenderer.invoke('model-delete', key),
  onModelDownloadProgress: (cb) =>
    ipcRenderer.on('model-download-progress', (_e, ev) => cb(ev)),
});
