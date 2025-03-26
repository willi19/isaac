# import sapien.core as sapien
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Dict, Any, Tuple
from typing import Union

import numpy as np
import yaml

from dex_robot.retargeting.filter import LPFilter
from dex_robot.retargeting.seq_retarget import SeqRetargeting
from dex_robot.retargeting.optimizer import Optimizer
from dex_robot.utils.file_io_prev import rsc_path


@dataclass  # This is decorator for dataclass
# dataclass automatically generates special methods like __init__, __repr__, __eq__, __hash__ etc.
# for variable that is not initialized, it will use the default value
# default value can be set by using the assignment operator (=)
# target_link_human_indices is set to False by default


class RetargetingConfig:
    type: str
    urdf_path: str
    objective_fn_args: Optional[Dict[str, Any]] = None

    # Low pass filter
    low_pass_alpha: float = 0.1

    _DEFAULT_URDF_DIR = rsc_path
    target_joint_names: Optional[List[str]] = None

    def __post_init__(self):
        # Retargeting type check
        self.type = self.type.lower()

        if self.type == "position":
            if "target_link_names" not in self.objective_fn_args:
                raise ValueError(f"Position retargeting requires: target_link_names")

        # URDF path check
        urdf_path = Path(self.urdf_path)
        if not urdf_path.is_absolute():
            urdf_path = self._DEFAULT_URDF_DIR / urdf_path
            urdf_path = urdf_path.absolute()
        if not urdf_path.exists():
            raise ValueError(f"URDF path {urdf_path} does not exist")
        self.urdf_path = str(urdf_path)

    @classmethod
    def set_default_urdf_dir(cls, urdf_dir: Union[str, Path]):
        path = Path(urdf_dir)
        if not path.exists():
            raise ValueError(f"URDF dir {urdf_dir} not exists.")
        cls._DEFAULT_URDF_DIR = urdf_dir

    def set_default_target_joint_names(self, target_joint_names: List[str]):
        self.target_joint_names = target_joint_names

    @classmethod
    def load_from_file(
        cls, config_path: Union[str, Path], override: Optional[Dict] = None
    ):
        path = Path(config_path)
        if not path.is_absolute():
            path = "config/retargeting" / path
            path = path.absolute()
        with path.open("r") as f:
            yaml_config = yaml.load(f, Loader=yaml.FullLoader)
            cfg = yaml_config["retargeting"]
            return cls.from_dict(cfg, override)

    @classmethod
    def from_dict(cls, cfg: Dict[str, Any], override: Optional[Dict] = None):
        if override is not None:
            for key, value in override.items():
                cfg[key] = value
        config = RetargetingConfig(**cfg)
        return config

    def build(self) -> SeqRetargeting:
        optimizer = Optimizer(
            self.urdf_path,
            self.target_joint_names,
            self.type,
            self.objective_fn_args,
        )

        if 0 <= self.low_pass_alpha <= 1:
            lp_filter = LPFilter(self.low_pass_alpha)
        else:
            lp_filter = None

        retargeting = SeqRetargeting(
            optimizer,
            lp_filter=lp_filter,
        )
        return retargeting
