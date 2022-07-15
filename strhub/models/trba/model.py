import torch.nn as nn

from strhub.models.modules import BidirectionalLSTM
from .feature_extraction import ResNet_FeatureExtractor
from .prediction import Attention
from .transformation import TPS_SpatialTransformerNetwork


class TRBA(nn.Module):

    def __init__(self, img_h, img_w, num_class, num_fiducial=20, input_channel=3, output_channel=512, hidden_size=256,
                 use_ctc=False):
        super().__init__()
        """ Transformation """
        self.Transformation = TPS_SpatialTransformerNetwork(
            F=num_fiducial, I_size=(img_h, img_w), I_r_size=(img_h, img_w),
            I_channel_num=input_channel)

        """ FeatureExtraction """
        self.FeatureExtraction = ResNet_FeatureExtractor(input_channel, output_channel)
        self.FeatureExtraction_output = output_channel
        self.AdaptiveAvgPool = nn.AdaptiveAvgPool2d((None, 1))  # Transform final (imgH/16-1) -> 1

        """ Sequence modeling"""
        self.SequenceModeling = nn.Sequential(
            BidirectionalLSTM(self.FeatureExtraction_output, hidden_size, hidden_size),
            BidirectionalLSTM(hidden_size, hidden_size, hidden_size))
        self.SequenceModeling_output = hidden_size

        """ Prediction """
        if use_ctc:
            self.Prediction = nn.Linear(self.SequenceModeling_output, num_class)
        else:
            self.Prediction = Attention(self.SequenceModeling_output, hidden_size, num_class)

    def forward(self, image, max_label_length, text=None):
        """ Transformation stage """
        image = self.Transformation(image)

        """ Feature extraction stage """
        visual_feature = self.FeatureExtraction(image)
        visual_feature = visual_feature.permute(0, 3, 1, 2)  # [b, c, h, w] -> [b, w, c, h]
        visual_feature = self.AdaptiveAvgPool(visual_feature)  # [b, w, c, h] -> [b, w, c, 1]
        visual_feature = visual_feature.squeeze(3)  # [b, w, c, 1] -> [b, w, c]

        """ Sequence modeling stage """
        contextual_feature = self.SequenceModeling(visual_feature)  # [b, num_steps, hidden_size]

        """ Prediction stage """
        if isinstance(self.Prediction, Attention):
            prediction = self.Prediction(contextual_feature.contiguous(), text, max_label_length)
        else:
            prediction = self.Prediction(contextual_feature.contiguous())  # CTC

        return prediction  # [b, num_steps, num_class]
