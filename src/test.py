from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import _init_paths

import os
import json
import cv2
import numpy as np
import time
from progress.bar import Bar
import torch

from external.nms import soft_nms
from opts import opts
from logger import Logger
from utils.utils import AverageMeter
from datasets.dataset_factory import dataset_factory
from detectors.detector_factory import detector_factory

class PrefetchDataset(torch.utils.data.Dataset):
  def __init__(self, opt, dataset, pre_process_func):
    self.images = dataset.images
    self.load_image_func = dataset.coco.loadImgs
    self.img_dir = dataset.img_dir
    self.pre_process_func = pre_process_func
    self.opt = opt
  
  def __getitem__(self, index):
    img_id = self.images[index]
    img_info = self.load_image_func(ids=[img_id])[0]
    img_path = os.path.join(self.img_dir, img_info['file_name'])
    image = cv2.imread(img_path)
    images, meta = {}, {}
    for scale in opt.test_scales:
      if opt.task == 'ddd':
        images[scale], meta[scale] = self.pre_process_func(
          image, scale, img_info['calib'])
      else:
        images[scale], meta[scale] = self.pre_process_func(image, scale)
    return img_id, {'images': images, 'image': image, 'meta': meta}

  def __len__(self):
    return len(self.images)

def prefetch_test(opt):
  os.environ['CUDA_VISIBLE_DEVICES'] = opt.gpus_str

  # opt.dataset 'coco'
  Dataset = dataset_factory[opt.dataset]
  opt = opts().update_dataset_info_and_set_heads(opt, Dataset)
  print(opt)
  Logger(opt)
  Detector = detector_factory[opt.task]
  
  split = 'val' if not opt.trainval else 'test'
  dataset = Dataset(opt, split)
  detector = Detector(opt)

  data_loader = torch.utils.data.DataLoader(
    PrefetchDataset(opt, dataset, detector.pre_process), 
    batch_size=1, shuffle=False, num_workers=0, pin_memory=True)

  results = {}
  num_iters = len(dataset)
  bar = Bar('{}'.format(opt.exp_id), max=num_iters)
  time_stats = ['tot', 'load', 'pre', 'net', 'dec', 'post', 'merge']
  avg_time_stats = {t: AverageMeter() for t in time_stats}
  for ind, (img_id, pre_processed_images) in enumerate(data_loader):
    # ind 0
    # img_id tensor([397133])
    
    ## test
    # pre_processed_images.keys() dict_keys(['images', 'image', 'meta'])
    # pre_processed_images['images'].keys() dict_keys([1.0])
    # pre_processed_images['images'][1.0].shape torch.Size([1, 1, 3, 512, 768])
    # pre_processed_images['image'].shape torch.Size([1, 427, 640, 3]) always true image size
    # pre_processed_images['meta'] 
    # {1.0: {'c': tensor([[320., 213.]]), 's': tensor([[768., 512.]]), 'out_height': tensor([128]), 'out_width': tensor([192])}}

    ## test_flip
    # pre_processed_images.keys() dict_keys(['images', 'image', 'meta'])
    # pre_processed_images['images'].keys() dict_keys([1.0])
    # pre_processed_images['images'][1.0].shape torch.Size([1, 2, 3, 512, 768])
    # pre_processed_images['image'].shape torch.Size([1, 427, 640, 3]) always true image size
    # pre_processed_images['meta'] 
    # {1.0: {'c': tensor([[320., 213.]]), 's': tensor([[768., 512.]]), 'out_height': tensor([128]), 'out_width': tensor([192])}}    

    ## test_scale
    # pre_processed_images.keys() dict_keys(['images', 'image', 'meta'])
    # pre_processed_images['images'].keys() dict_keys([0.5, 0.75, 1.0, 1.25, 1.5])
    # pre_processed_images['images'][1.0].shape torch.Size([1, 2, 3, 512, 768])
    # pre_processed_images['image'].shape torch.Size([1, 427, 640, 3]) always true image size
    # pre_processed_images['meta'] 
    # {0.5: {'c': tensor([[160., 106.]]), 's': tensor([[384., 256.]]), 'out_height': tensor([64]), 'out_width': tensor([96])}, 0.75: {'c': tensor([[240., 160.]]), 's': tensor([[512., 384.]]), 'out_height': tensor([96]), 'out_width': tensor([128])}, 1.0: {'c': tensor([[320., 213.]]), 's': tensor([[768., 512.]]), 'out_height': tensor([128]), 'out_width': tensor([192])}, 1.25: {'c': tensor([[400., 266.]]), 's': tensor([[896., 640.]]), 'out_height': tensor([160]), 'out_width': tensor([224])}, 1.5: {'c': tensor([[480., 320.]]), 's': tensor([[1024.,  768.]]), 'out_height': tensor([192]), 'out_width': tensor([256])}}

    ret = detector.run(pre_processed_images)
    ## test_scale
    # ret.keys()
    # dict_keys(['results', 'tot', 'load', 'pre', 'net', 'dec', 'post', 'merge'])
    # ret['tot'] 0.4539330005645752
    # ret['dec'] 0.004953622817993164
    # ...pre always time
    # ret['results'].keys()
    # always whatever flip scale or none dict_keys([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80])
    # ret['results'][1].shape (0, 5)
    # ret['results'][2].shape (0, 5)
    # ret['results'][70].shape (4, 5)

    results[img_id.numpy().astype(np.int32)[0]] = ret['results']
    Bar.suffix = '[{0}/{1}]|Tot: {total:} |ETA: {eta:} '.format(
                   ind, num_iters, total=bar.elapsed_td, eta=bar.eta_td)
    for t in avg_time_stats:
      avg_time_stats[t].update(ret[t])
      Bar.suffix = Bar.suffix + '|{} {tm.val:.3f}s ({tm.avg:.3f}s) '.format(
        t, tm = avg_time_stats[t])
    bar.next()
  bar.finish()
  dataset.run_eval(results, opt.save_dir)

def test(opt):
  os.environ['CUDA_VISIBLE_DEVICES'] = opt.gpus_str

  Dataset = dataset_factory[opt.dataset]
  opt = opts().update_dataset_info_and_set_heads(opt, Dataset)
  print(opt)
  Logger(opt)
  Detector = detector_factory[opt.task]
  
  split = 'val' if not opt.trainval else 'test'
  dataset = Dataset(opt, split)
  detector = Detector(opt)

  results = {}
  num_iters = len(dataset)
  bar = Bar('{}'.format(opt.exp_id), max=num_iters)
  time_stats = ['tot', 'load', 'pre', 'net', 'dec', 'post', 'merge']
  avg_time_stats = {t: AverageMeter() for t in time_stats}
  for ind in range(num_iters):
    import pdb
    pdb.set_trace()
    img_id = dataset.images[ind]
    img_info = dataset.coco.loadImgs(ids=[img_id])[0]
    img_path = os.path.join(dataset.img_dir, img_info['file_name'])

    if opt.task == 'ddd':
      ret = detector.run(img_path, img_info['calib'])
    else:
      ret = detector.run(img_path)
    
    results[img_id] = ret['results']

    Bar.suffix = '[{0}/{1}]|Tot: {total:} |ETA: {eta:} '.format(
                   ind, num_iters, total=bar.elapsed_td, eta=bar.eta_td)
    for t in avg_time_stats:
      avg_time_stats[t].update(ret[t])
      Bar.suffix = Bar.suffix + '|{} {:.3f} '.format(t, avg_time_stats[t].avg)
    bar.next()
  bar.finish()
  dataset.run_eval(results, opt.save_dir)

if __name__ == '__main__':
  opt = opts().parse()
  if opt.not_prefetch_test:
    test(opt)
  else:
    prefetch_test(opt)