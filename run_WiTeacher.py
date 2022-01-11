import sys
import logging

import torch

import main
from mean_teacher.cli import parse_dict_args
from mean_teacher.run_context import RunContext


LOG = logging.getLogger('runner')


def parameters():
    defaults = {
        # Technical details
        'workers': 2,
        'checkpoint_epochs': 100,

        # Data
        'dataset': 'SignFi',#or DeepSeg
        'train_subdir': 'train',
        'eval_subdir': 'test',

        # Data sampling
        'base_batch_size': 100,
        'base_labeled_batch_size': 99,

        # Architecture
        'arch': 'cifar_shakeshake26',

        # Costs
        'consistency_type': 'mse',
        'consistency_rampup': 5,
        'consistency': 99.8,
        'logit_distance_cost': .01,
        'weight_decay': 2e-4,

        # Optimization
        'lr_rampup': 0,
        'base_lr': 0.05,
        'nesterov': True,


    }


    # 1000 labels:
    for data_seed in range(10, 11):
        yield {
            **defaults,
            'title': '1000-label SignFi',
            'n_labels': 1000,
            'data_seed': data_seed,
            'epochs': 100,
            'lr_rampdown_epochs': 610,
            'ema_decay': 0.99,
        }


def run(title, base_batch_size, base_labeled_batch_size, base_lr, n_labels, data_seed, **kwargs):
    LOG.info('run title: %s, data seed: %d', title, data_seed)

    ngpu = torch.cuda.device_count()
    assert ngpu > 0, "Expecting at least one GPU, found none."

    adapted_args = {
        'batch_size': base_batch_size * ngpu,
        'labeled_batch_size': base_labeled_batch_size * ngpu,
        'lr': base_lr * ngpu,
        'labels': './data-local/labels/signfi/00.txt',
    }
    context = RunContext(__file__, "{}_{}".format(n_labels, data_seed))
    main.args = parse_dict_args(**adapted_args, **kwargs)
    main.main(context)


if __name__ == "__main__":
    for run_params in parameters():
         run(**run_params)


