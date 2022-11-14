import torch
import torch.nn as nn


# from layers.masking import TriangularCausalMask, ProbMask
# from layers.Embed import
from layers.Transformer_EncDec import Decoder, DecoderLayer, Encoder, EncoderLayer, ConvLayer
from layers.SelfAttention_Family import ProbAttention, AttentionLayer
from layers.Embed import DataEmbedding, PureDataEmbedding, CateEmbedding
from layers.RevIn import RevIN

class Model(nn.Module):
    """
    Informer with Propspare attention in O(LlogL) complexity
    """
    def __init__(self, configs):
        super(Model, self).__init__()
        self.pred_len = configs.pred_len
        self.output_attention = configs.output_attention

        # Embedding
        self.enc_embedding = DataEmbedding(configs.enc_in, configs.d_model, configs.embed, configs.freq,
                                           configs.dropout)
        self.dec_embedding = DataEmbedding(configs.dec_in, configs.d_model, configs.embed, configs.freq,
                                           configs.dropout)

        # Encoder
        self.encoder = Encoder(
            [
                EncoderLayer(
                    AttentionLayer(
                        ProbAttention(False, configs.factor, attention_dropout=configs.dropout,
                                      output_attention=configs.output_attention),
                        configs.d_model, configs.n_heads),
                    configs.d_model,
                    configs.d_ff,
                    dropout=configs.dropout,
                    activation=configs.activation
                ) for l in range(configs.e_layers)
            ],
            [
                ConvLayer(
                    configs.d_model
                ) for l in range(configs.e_layers - 1)
            ] if configs.distil else None,
            norm_layer=torch.nn.LayerNorm(configs.d_model)
        )
        # Decoder
        self.decoder = Decoder(
            [
                DecoderLayer(
                    AttentionLayer(
                        ProbAttention(True, configs.factor, attention_dropout=configs.dropout, output_attention=False),
                        configs.d_model, configs.n_heads),
                    AttentionLayer(
                        ProbAttention(False, configs.factor, attention_dropout=configs.dropout, output_attention=False),
                        configs.d_model, configs.n_heads),
                    configs.d_model,
                    configs.d_ff,
                    dropout=configs.dropout,
                    activation=configs.activation,
                )
                for l in range(configs.d_layers)
            ],
            norm_layer=torch.nn.LayerNorm(configs.d_model),
            projection=nn.Linear(configs.d_model, configs.c_out, bias=True)
        )

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec,
                enc_self_mask=None, dec_self_mask=None, dec_enc_mask=None):

        enc_out = self.enc_embedding(x_enc, x_mark_enc)
        enc_out, attns = self.encoder(enc_out, attn_mask=enc_self_mask)

        dec_out = self.dec_embedding(x_dec, x_mark_dec)
        dec_out = self.decoder(dec_out, enc_out, x_mask=dec_self_mask, cross_mask=dec_enc_mask)

        if self.output_attention:
            return dec_out[:, -self.pred_len:, :], attns
        else:
            return dec_out[:, -self.pred_len:, :]  # [B, L, D]


class ModifiedInformerModel(nn.Module):
    def __init__(self, configs):
        super(ModifiedInformerModel, self).__init__()
        self.encoder_length = configs["encoder_length"]
        self.pred_len = configs["pred_length"]
        self.output_attention = False
        self.dropout = configs["dropout"]
        self.d_model = configs["model_dimension"]
        self.d_data = configs["data_dimension"]
        self.cate_num_list = configs["cardinality"]
        self.scale_factor = configs["scale_factor"]
        self.n_heads = configs["head_num"]
        self.d_ff = configs["ff_dimension"]
        self.activation = configs["activated_function"]
        self.e_layers = configs["encoder_layers"]
        self.d_layers = configs["decoder_layers"]

        self.c_out = configs["output_dimension"]
        try:
            self.distil = configs["distillation"]
        except:
            self.distil = False

        try:
            self.use_revIn = configs["revIn"]
            self.revINLayer = RevIN(num_features=1)
        except:
            self.use_revIn = False

        # Embedding
        self.enc_embedding = PureDataEmbedding(c_in=self.d_data, d_model=self.d_model, dropout=self.dropout)
        self.dec_embedding = PureDataEmbedding(c_in=self.d_data, d_model=self.d_model, dropout=self.dropout)

        self.item_embedding = CateEmbedding(d_model=self.d_model, total_num_list=self.cate_num_list,)

        # Encoder
        self.encoder = Encoder(
            [
                EncoderLayer(
                    AttentionLayer(
                        ProbAttention(False, self.scale_factor, attention_dropout=self.dropout,
                                      output_attention=self.output_attention),
                        self.d_model, self.n_heads),
                    self.d_model,
                    self.d_ff,
                    dropout=self.dropout,
                    activation=self.activation
                ) for _ in range(self.e_layers)
            ],
            [
                ConvLayer(
                    self.d_model
                ) for _ in range(self.e_layers - 1)
            ] if self.distil else None,
            norm_layer=torch.nn.LayerNorm(self.d_model)
        )
        # Decoder
        self.decoder = Decoder(
            [
                DecoderLayer(
                    AttentionLayer(
                        ProbAttention(True, self.scale_factor, attention_dropout=self.dropout, output_attention=False),
                        self.d_model, self.n_heads),
                    AttentionLayer(
                        ProbAttention(False, self.scale_factor, attention_dropout=self.dropout, output_attention=False),
                        self.d_model, self.n_heads),
                    self.d_model,
                    self.d_ff,
                    dropout=self.dropout,
                    activation=self.activation,
                )
                for l in range(self.d_layers)
            ],
            norm_layer=torch.nn.LayerNorm(self.d_model),
            projection=nn.Linear(self.d_model, self.c_out, bias=True)
        )

    def forward(self, x_enc, x_mark_enc, x_dec, enc_self_mask=None, dec_self_mask=None, dec_enc_mask=None):

        if self.use_revIn:
            x_enc = self.revINLayer(x_enc, 'norm')

        item_embedding = self.item_embedding(x_mark_enc).unsqueeze(1).expand(-1, self.encoder_length, -1)

        seq_embedding = self.enc_embedding(x_enc)

        enc_out = seq_embedding + item_embedding
        enc_out, attns = self.encoder(enc_out, attn_mask=enc_self_mask)

        dec_out = self.dec_embedding(x_dec)

        dec_out = self.decoder(dec_out, enc_out, x_mask=dec_self_mask, cross_mask=dec_enc_mask)
        dec_out = dec_out + torch.cat([x_enc, x_enc[:, :self.pred_len, :]], dim=1)

        if self.use_revIn:
            dec_out =self.revINLayer(dec_out, 'denorm')

        if self.output_attention:
            return dec_out[:, -self.pred_len:, :], attns
        else:
            return dec_out[:, -self.pred_len:, :]

