from hptf.uer.targets.mlm_target import MlmTarget
from hptf.uer.targets.lm_target import LmTarget
from hptf.uer.targets.bert_target import BertTarget
from hptf.uer.targets.cls_target import ClsTarget
from hptf.uer.targets.bilm_target import BilmTarget
from hptf.uer.targets.albert_target import AlbertTarget
from hptf.uer.targets.seq2seq_target import Seq2seqTarget
from hptf.uer.targets.t5_target import T5Target
from hptf.uer.targets.prefixlm_target import PrefixlmTarget
from hptf.uer.targets.nsp_target import NspTarget

str2target = {"bert": BertTarget, "mlm": MlmTarget, "lm": LmTarget,
              "bilm": BilmTarget, "albert": AlbertTarget, "seq2seq": Seq2seqTarget,
              "t5": T5Target, "cls": ClsTarget, "prefixlm": PrefixlmTarget, "nsp":NspTarget}

__all__ = ["BertTarget", "MlmTarget", "LmTarget", "BilmTarget", "AlbertTarget",
           "Seq2seqTarget", "T5Target", "ClsTarget", "PrefixlmTarget",
           "str2target","NspTarget"]
