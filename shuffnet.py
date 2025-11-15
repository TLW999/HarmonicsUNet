import math
import os
import sys
sys.path.append("/root/BirdsongRecognition")
from transformers.utils import ModelOutput

# os.environ['CUDA_VISIBLE_DEVICES'] = '0'  # 仅使用ID为0的GPU

from safetensors.torch import load_model


import numpy as np
from wandb import Config

from hugging_face.eca_resnet import eca_resnet50
from hugging_face.sa_resnet import sa_resnet50, sa_resnet152
from ResNet50_1d.ResNet50_1d import resnet50_1d as ResNet50_1d
from ResNet50_1d.sa_resnet50_1d import sa_resnet50 as sa_resnet50_1d
from kymatio.torch import Scattering1D, Scattering2D
import torchaudio


from rsnet import drsn

from torch import nn
from torchaudio.prototype.transforms import ChromaSpectrogram, BarkSpectrogram
from torchaudio.transforms import MelSpectrogram, AmplitudeToDB
from transformers.modeling_outputs import SequenceClassifierOutput

import torch
from CRNN import crnn,CR
from hugging_face.AudioDataSet import ImageDataset
from hugging_face.custom_trainer import custom_trainer
import warnings
from PGN import timespec,unet,leaf

from TCN import TCN_BiGRU_Attention

warnings.filterwarnings('ignore')
from ShuffleNetV2 import network
from wav2Encode import AudioEncoder
from ResNet50_1d.ResNet50_1d import resnet50_1d

from ResNet50 import resnet50
from ResNet50 import res2net
from AudioEnAttention import AudioAttention
from ScatAttention import ScatAttention
from loss import FocalLoss

import matplotlib.pyplot as plt

plt.rcParams['font.family'] = 'JetBrains Mono'  # 设置为你正在使用的字体

def add_gaussian_noise(input, mean=0., stddev=0.05):
    """
    向输入添加高斯噪声。

    参数:
    - input (Tensor): 输入数据。
    - mean (float): 噪声的平均值。
    - stddev (float): 噪声的标准差。

    返回:
    - 带噪声的输入数据。
    """
    if input.is_cuda:
        noise = torch.randn(input.size()).cuda() * stddev + mean
    else:
        noise = torch.randn(input.size()) * stddev + mean
    return input + noise

#数据增强
def mixup_data(x, y, alpha=0.4, use_cuda=True):


    """
    参数：
    x:输入数据
    y:对应的标签
    alpha:用于控制生成数据混合的比例
    """

    #生成混合比例
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1

    batch_size = x.size()[0] #获取输入数据的批次大小
    if use_cuda:
        index = torch.randperm(batch_size).cuda() #index生成一个随机排列的索引，用于在x中选择不同的样本
    else:
        index = torch.randperm(batch_size)

    mixed_x = lam * x + (1 - lam) * x[index, :] #生成的混合样本，通过将样本x和通过随机索引选择的样本x[index]按照比例lam和1-lam进行线性组合
    y_a, y_b = y, y[index] #y_a原始标签。y_b随机选择的标签y[index]
    return mixed_x, y_a, y_b, lam


class shuffle_net(nn.Module):
    def __init__(self, num_classes, audio_length, sr):
        super().__init__()
        self.num_classes = num_classes
        #频谱转换
        self.spectrogram_transform= MelSpectrogram(n_fft=2048, hop_length=1024, sample_rate=16000, n_mels=128)
        # self.bark_spectrogram_transform = BarkSpectrogram(n_fft=2048, hop_length=512, sample_rate=16000, n_barks=128)
        # self.chroma_transform = ChromaSpectrogram(sample_rate=16000, n_fft=2048, hop_length=512, n_chroma=12)
        #模型
        #self.model=TP.BirdSoundRecognitionWithBiGRUAndTPGN(num_mel_bands=128,num_classes=num_classes)
        #self.model=TPGN.SFEBirdsrecognition(num_classes=num_classes)
        #self.model=PGN.BirdSoundRecognitionWithBiGRUAndTPGN(num_classes=num_classes,num_mel_bands=128)
        # self.model=TP.BirdSoundRecognitionWithBiGRUAndTPGN(num_classes=num_classes,num_mel_bands=128)
        # self.model=leaf.UNetSpectroFusion(num_classes)
        # self.model=timespec.UNetSpectroFusion(num_classes)
        #self.model = unet.UNetSpectroFusion(num_classes)
        #self.model=Timing.BirdSoundRecognitionWithBiGRUAndTPGN(num_classes=num_classes,num_mel_bands=128)
        #self.model=leaf.BioLeaf(out_channels=64,sample_rate=32000,fusion_type="attention",use_adaptive_pcen=True)
        # self.model=TCN_BiGRU_Attention.BirdsongRecognitionNet(num_classes=num_classes,num_mel_bands=128)

        # self.model = network.ShuffleNetV2(n_class=num_classes, model_size="2.0x")

        # self.model = resnet50.ResNet50(num_classes=num_classes)
        self.model = CR.CRNN_Basic(num_classes=num_classes)
        # self.model = crnn.CRNN()
        # self.model = sa_resnet50(num_class=num_classes)
        # self.AudioBlock = AudioBlock()
        # self.model = sa_resnet152(num_class=num_classes)
        # self.model = drsn.rsnet34(num_classes=num_classes)
        # self.audio_en1 = AudioEn(chunk_time=20, audio_time=audio_length)

        # self.model = AudioAttention(chunk_time=50, audio_time=audio_length)
        # self.model = ScatAttention(num_class=num_classes, audio_length=audio_length, sr=sr)

        # self.model = AudioTransformer(chunk_time=50, audio_time=audio_length, num_classes=num_classes)
        # self.audio_en = AudioEn1()

        # self.model = resnet50_1d(num_class=num_classes)
        # self.model = res2net.res2net50_26w_4s()
        # self.model = resnet50.ResNet50(num_classes=num_classes)
        # self.model = sa_resnet50_1d(num_class=num_classes)
        # self.pcen = BatchPCEN(input_size=128, trainable=False, skip_transpose=False)
        # 8 12
        self.scattering_list = []
        self.scattering_params = [(10, 4)]
        for J, Q in self.scattering_params:
            self.scattering_list.append(Scattering1D(J=J, shape=sr * audio_length, Q=Q, max_order=2))

    def waveform_to_features(self, waveform):

        mel_spectrogram, chroma, bark_spectrogram, wst_mel, wst = None, None, None, None, None
        feature_group = []
        mel_spectrogram = AmplitudeToDB()(self.spectrogram_transform.to(waveform.device)(waveform)).contiguous()
        # mel_spectrogram = self.spectrogram_transform.to(waveform.device)(waveform)
        # bark_spectrogram = AmplitudeToDB()(self.bark_spectrogram_transform.to(waveform.device)(waveform))
        # bark_spectrogram = self.pcen(self.bark_spectrogram_transform.to(waveform.device)(waveform))
        # bark_spectrogram = bark_spectrogram.expand(-1, 3, -1, -1)

        # for i in self.scattering_list:
        #     # feature_group.append(torch.mean(i.to(waveform.device)(waveform)[:, :, 1:, :], dim=-1))
        #     feature_group.append(i.to(waveform.device)(waveform)[:, :, 1:, :])
        # wst = torch.cat(feature_group, dim=-1)
        # wst = torch.log(wst + 1e-6)
        # print(wst.shape)
        return mel_spectrogram, chroma, bark_spectrogram, wst_mel, wst

    def forward(self, input_values, labels=None, sample_id=None):

        mixup = False
        targets_a, targets_b, lam, = None, None, None
        if self.training and mixup:
            # 加噪声
            # input_values = add_gaussian_noise(input_values, mean=0., stddev=0.1)
            # 进行数据增强
            input_values, targets_a, targets_b, lam = mixup_data(input_values, labels, alpha=0.4, use_cuda=input_values.is_cuda)
            mel_spectrogram, chroma, bark_spectrogram, wst_mel, wst = self.waveform_to_features(input_values)
        else:
            mel_spectrogram, chroma, bark_spectrogram, wst_mel, wst = self.waveform_to_features(input_values)
        loss = None
        logits = self.model(mel_spectrogram)
        if labels is not None:
            if self.training and mixup:

                loss_fct = nn.CrossEntropyLoss()
                loss = lam * loss_fct(logits, targets_a) + (1 - lam) * loss_fct(logits, targets_b)
            else:

                loss_fct = torch.nn.CrossEntropyLoss()
                # loss_fct = FocalLoss.FocalLoss(num_classes=self.num_classes)
                loss = loss_fct(logits, labels)
        return SequenceClassifierOutput(
            loss=loss,
            logits=(logits, sample_id),
            hidden_states=None
        )


class train(custom_trainer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.data_dir = "/dev/sda1/dataset/BirdsData_16000"
        self.csv_data_files = r"/dev/sda1/dataset/BirdsData_16000/birds_data.csv"
        # self.data_dir = "/dev/sda1/dataset/UrbanSound8K/audio"
        # self.csv_data_files = r"/dev/sda1/dataset/UrbanSound8K/metadata/new_UrbanSound8K_id.csv"
        # self.data_dir="/dev/sda1/dataset/ESC-50-master/audio"
        # self.csv_data_files="/dev/sda1/dataset/ESC-50-master/audio/esc50.csv"
        super().setup_data()

    def set_model(self):
        model = shuffle_net(self.class_num, self.audio_length, self.sr)
        return model

    def set_dataset(self):
        return ImageDataset

    def optimizer_scheduler_init(self):
        pretrained_params = self.model.parameters()
        optimizer = torch.optim.AdamW(pretrained_params, lr=self.lr)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer=optimizer, T_0=5, T_mult=2)
        # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer=optimizer, T_max=self.batch_size * self.epoch)
        return optimizer, scheduler


if "__main__" == __name__:
    os.environ["WANDB_API_KEY"] = "112469dae0d2b5d107c43a37378126757d11871d"
    os.environ["WANDB_MODE"] = "offline"
    project = "CRNN"
    name = "Bardsdata"

    t = train(class_num=20, audio_length=2, lr=0.0001 , batch_size=128, eval_batch_size=128,  #class_num=10,audio_length=4
              resume_from_checkpoint=False,dataloader_num_workers=6,
              epoch=200, is_wandb=True, is_confusion_matrix=True, seed=42, wandb_name=name,
              wandb_project=project, remove_imbalance=False, remove_imbalance_threshold=460, k_fold=True,
              official=False, save_total_limit=1, log_sample=False, sr=16000, output_dir="./ScEncode",k_fold_num=5)
    t.train()
