# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This software may be used and distributed in accordance with
# the terms of the FAIR Noncommercial Research License.

import os
from dataclasses import dataclass
from typing import Tuple, TypedDict

import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn

from eupe.configs import EupeSetupArgs, setup_config
from eupe.models import build_model_for_eval


REPO_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


@dataclass
class ModelConfig:
    config_file: str | None = None
    pretrained_weights: str | None = None
    eupe_hub: str | None = None


class BaseModelContext(TypedDict):
    """
    An object that contains the context of a model (autocast, description, ...)
    """

    autocast_dtype: torch.dtype  # default could be torch.float


def load_model_and_context(model_config: ModelConfig, output_dir: str) -> tuple[torch.nn.Module, BaseModelContext]:
    if model_config.eupe_hub is not None:
        assert model_config.config_file is None
        model = torch.hub.load(
            REPO_DIR,
            model_config.eupe_hub,
            source="local",
            weights=model_config.pretrained_weights,
        )
        base_model_context = BaseModelContext(autocast_dtype=torch.float)
    else:
        model, base_model_context = setup_and_build_model(
            config_file=model_config.config_file,
            pretrained_weights=model_config.pretrained_weights,
            output_dir=output_dir,
        )

    model.cuda()
    model.eval()
    return model, base_model_context


def get_autocast_dtype(config):
    teacher_dtype_str = config.compute_precision.param_dtype
    if teacher_dtype_str == "bf16":
        return torch.bfloat16
    else:
        return torch.float


def setup_and_build_model(
    config_file: str,
    pretrained_weights: str | None = None,
    output_dir: str = "",
    opts: list | None = None,
    **ignored_kwargs,
) -> Tuple[nn.Module, BaseModelContext]:
    cudnn.benchmark = True
    del ignored_kwargs
    setup_args = EupeSetupArgs(
        config_file=config_file,
        pretrained_weights=pretrained_weights,
        output_dir=output_dir,
        opts=opts or [],
    )
    config = setup_config(setup_args, strict_cfg=False)
    model = build_model_for_eval(config, setup_args.pretrained_weights)
    autocast_dtype = get_autocast_dtype(config)
    return model, BaseModelContext(autocast_dtype=autocast_dtype)
