import torch
from mmengine.registry import HOOKS
from mmengine.hooks import Hook
from mmengine.logging import MMLogger


@HOOKS.register_module()
class ChangeStrategyHook(Hook):

    def __init__(self, change_epoch=[], change_strategy=[]):
        self.epoch = 0
        self.change_epoch = change_epoch
        self.change_strategy = change_strategy
        self.logger = MMLogger.get_current_instance()

    def before_train_epoch(self, runner) -> None:
        self.epoch = runner.epoch + 1
        self.logger.info('Epoch %d, change_epoch: %s, change_strategy: %s' % (self.epoch, self.change_epoch, self.change_strategy))
        for i, epoch in enumerate(self.change_epoch):
            if self.epoch == epoch:
                if self.change_strategy[i] == 'remove_GTSample':
                    self.remove_GTSample(runner)
                elif self.change_strategy[i] == 'remove_DN':
                    self.remove_DN(runner)

    def remove_GTSample(self, runner):
        for i, transform in enumerate(runner.train_dataloader.dataset.dataset.pipeline.transforms):
            if transform.__class__.__name__ == 'UnifiedObjectSample':
                runner.train_dataloader.dataset.dataset.pipeline.transforms.pop(i)
                self.logger.info('Remove UnifiedObjectSample (transform %d of pipeline)' % i)
                break
        else:
            self.logger.info('UnifiedObjectSample not found in pipeline')

    def remove_DN(self, runner):
        if hasattr(runner.model, 'module'):
            if runner.model.module.pts_bbox_head.with_dn:
                runner.model.module.pts_bbox_head.with_dn = False
                self.logger.info('Remove DN in pts_bbox_head(module)')
            else:
                self.logger.info('Cannot remove DN')
        else:
            if runner.model.pts_bbox_head.with_dn:
                runner.model.pts_bbox_head.with_dn = False
                self.logger.info('Remove DN in pts_bbox_head')
            else:
                self.logger.info('Cannot remove DN')