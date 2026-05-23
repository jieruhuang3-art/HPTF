from hptf.uer.encoders.transformer_encoder import TransformerEncoder
from hptf.uer.encoders.rnn_encoder import RnnEncoder
from hptf.uer.encoders.rnn_encoder import LstmEncoder
from hptf.uer.encoders.rnn_encoder import GruEncoder
from hptf.uer.encoders.rnn_encoder import BirnnEncoder
from hptf.uer.encoders.rnn_encoder import BilstmEncoder
from hptf.uer.encoders.rnn_encoder import BigruEncoder
from hptf.uer.encoders.cnn_encoder import GatedcnnEncoder


str2encoder = {"transformer": TransformerEncoder, "rnn": RnnEncoder, "lstm": LstmEncoder,
               "gru": GruEncoder, "birnn": BirnnEncoder, "bilstm": BilstmEncoder, "bigru": BigruEncoder,
               "gatedcnn": GatedcnnEncoder}

__all__ = ["TransformerEncoder", "RnnEncoder", "LstmEncoder", "GruEncoder", "BirnnEncoder",
           "BilstmEncoder", "BigruEncoder", "GatedcnnEncoder", "str2encoder"]
