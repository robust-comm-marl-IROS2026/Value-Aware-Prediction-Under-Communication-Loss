REGISTRY = {}

from .basic_controller import BasicMAC
from .non_shared_controller import NonSharedMAC
from .basic_attention_controller import BasicAttentionMAC
from .non_shared_attention_controller import NonSharedAttentionMAC
from .maddpg_controller import MADDPGMAC

REGISTRY["basic_mac"] = BasicMAC
REGISTRY["non_shared_mac"] = NonSharedMAC
REGISTRY["basic_attention_mac"] = BasicAttentionMAC
REGISTRY["non_shared_attention_mac"] = NonSharedAttentionMAC
REGISTRY["maddpg_mac"] = MADDPGMAC