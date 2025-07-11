import torch
import torch.nn as nn
from pangaea.encoders.base import Encoder
from logging import Logger
import sys
import os

sys.path.append(os.path.abspath("/home/sghiassi/pangaea-bench/AnySat"))
from AnySat.hubconf import AnySat



class AnySat_Encoder(Encoder):
    """
    Wrapper for the pretrained AnySat model from Hugging Face.
    This encoder is compatible with the same input/output format as Prithvi_Encoder.
    """

    def __init__(
        self,
        encoder_weights: str,
        input_bands: dict[str, list[str]],
        input_size: int,
        output_dim: int,
        output_layers: int | list[int],
        download_url: str = "",
        patch_size: int = 24,        
        output_type: str = "patch",
        output_modality: str = None,
        embed_dim: int = 768,
        multi_temporal: bool = False,
        multi_temporal_output: bool = False,
        pyramid_output: bool = False,
    ):
        super().__init__(
            model_name="AnySat",
            encoder_weights=encoder_weights,
            input_bands=input_bands,
            input_size=input_size,
            embed_dim=embed_dim,
            output_layers=output_layers,
            output_dim=output_dim,
            multi_temporal=multi_temporal,
            multi_temporal_output=multi_temporal_output,
            pyramid_output=pyramid_output,
            download_url=download_url,
        )

        self.patch_size = patch_size
        self.output_type = output_type
        self.output_modality = output_modality
        # Load pretrained AnySat model
        # self.model = torch.hub.load('gastruc/anysat', 'anysat', pretrained=True, flash_attn=False)
        # self.model.patch_size = self.patch_size
        self.model = AnySat.from_pretrained('base', flash_attn=False)
        # Freeze model if needed
        self._frozen = False

    def forward(self, image: dict):
        """
        Forward pass through the AnySat encoder.
        Accepts a dictionary of inputs (e.g., {"s2": tensor, "s2_dates": tensor}).
        Returns a list of feature maps from specified output layers.
        """

        if "optical" in image.keys():
            # replace that to s2
            data = {}
            data["hls"] = image["optical"]
            # expand dim to [batch, time, channels, height, width]
            data["hls"] = data["hls"].unsqueeze(1)  # add time dimension
            data["hls_dates"] = image["hls_dates"]
        else: 
            data = image
        
        if data["hls_dates"].shape[0] == 0:
            data["hls_dates"] = torch.zeros(data["hls"].shape[0], dtype=torch.int64).to(data["hls"].device)
        
        # colapse time dimension if not multi-temporal
        # if self.multi_temporal and image["optical"].ndim == 5:
        if self.output_type == "dense" and self.output_modality is not None:
            features = self.model(data, patch_size=self.patch_size, output=self.output_type, output_modality=self.output_modality)
        else:
            features = self.model(data, scale=30, patch_size=self.patch_size, output=self.output_type)
        
        if not isinstance(features, list):

            features = [features]
        
        # Change the dimension order of feature data to [batch, channels, height, width]
        features = [f.permute(0, 3, 1, 2).contiguous() for f in features]

        return features

        # features = self.model(image, patch_size=self.patch_size)
        # if not isinstance(features, list):
        #     features = [features]
        # return features

    def freeze(self):
        """
        Freeze the encoder weights.
        """
        for param in self.model.parameters():
            param.requires_grad = False
        self._frozen = True

    
    def load_encoder_weights(self, logger: Logger) -> None:
        logger.info("AnySat model loaded via torch.hub; no additional weights to load.")


# import torch
# from logging import Logger
# from pathlib import Path

# from pangaea.encoders.base import Encoder
# from pangaea.encoders.pos_embed import get_3d_sincos_pos_embed


# class AnySat_Encoder(Encoder):
#     """
#     Paper: https://arxiv.org/pdf/2412.14123"
#     huggingface: https://huggingface.co/g-astruc/AnySat
    
#     Attributes:
#         patch_size (int): The size of each patch.  (in m, must be a multiple of 10): adjust according to the scale of your tiles and GPU memory. In general, avoid having more than 1024 patches per tile.
#         output_type (str): 'tile', 'patch, 'dense' or 'all'.
#         scale (int): The scale of the input data, default is 10.
#         output_modality (str): The modality of the output.
#     methods:
#         __init__(self, encoder_weights: str | Path, input_bands: dict[str, list[str]], input_size: int, output_layers: int | list[int], patch_size=10, output_type='tile', scale=10):
#             Initializes the AnySat_Encoder with the given parameters.
#         load_encoder_weights(self, logger: Logger) -> None:
#             Loads the encoder weights from a pretrained model and handles any missing or incompatible shapes.
#         freeze(self):
#             Freezes the parameters of the encoder to prevent them from being updated during training.
#         initialize_weights(self):
#             Initializes the weights of the encoder, including the positional embeddings and patch embeddings.
#         forward(self, image):
#             Performs the forward pass of the encoder, embedding the input patches, adding positional embeddings, and applying the Transformer blocks.

#     """
#     def __init__(
#         self,
#         encoder_weights: str | Path,
#         input_bands: dict[str, list[str]],
#         input_size: int,
#         output_layers: int | list[int],
#         download_url: str,
#         patch_size=10,
#         output_type='tile',
#         scale=10,
#     ):
#         super().__init__(
#             encoder_weights=encoder_weights,
#             input_bands=input_bands,
#             input_size=input_size,
#             output_layers=output_layers,
#             download_url=download_url
#         )
#         self.patch_size = patch_size
#         self.output_type = output_type
#         self.scale = scale