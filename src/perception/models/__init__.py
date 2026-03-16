REGISTRY = {}

from .joint_obs import JointObsModel
REGISTRY["joint_obs"] = JointObsModel

from .state import State
REGISTRY["state"] = State

from .masked_joint_obs import MaskedJointObsModel
REGISTRY["masked_joint_obs"] = MaskedJointObsModel

from .maro import MARO
REGISTRY["maro"] = MARO

