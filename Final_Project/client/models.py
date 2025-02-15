import torch
import torch.nn as nn
import torchaudio.transforms as T

class MaskedConv1d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1, bias=True):
        super().__init__()
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size, stride=stride,
                              padding=padding, dilation=dilation, groups=groups, bias=bias)
        self.padding = padding
        self.dilation = dilation
        self.stride = stride
        self.kernel_size = kernel_size

    def forward(self, x, length):
        max_length = x.size(2)
        length = torch.div(((length + 2 * self.padding - self.dilation * (self.kernel_size - 1) - 1).float() + self.stride),
                           self.stride, rounding_mode='floor').long()
        mask = torch.arange(max_length, device=x.device)[None, :] < length[:, None]
        x = x * mask.unsqueeze(1)
        x = self.conv(x)
        return x, length

class JasperBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, dropout=0.0, residual=False):
        super().__init__()
        self.mconv = nn.ModuleList([
            MaskedConv1d(in_channels, in_channels, kernel_size, stride=stride, padding=padding, dilation=dilation, groups=in_channels, bias=False),
            MaskedConv1d(in_channels, out_channels, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm1d(out_channels)
        ])
        self.res = None
        if residual:
            self.res = nn.ModuleList([
                nn.ModuleList([
                    MaskedConv1d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                    nn.BatchNorm1d(out_channels)
                ])
            ])
        self.mout = nn.Sequential(
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout)
        )

    def forward(self, x, length):
        residual = x
        res_length = length
        out = x
        out_length = length
        for layer in self.mconv:
            if isinstance(layer, MaskedConv1d):
                out, out_length = layer(out, out_length)
            else:
                out = layer(out)
        if self.res:
            for res_layer_list in self.res:
                res = residual
                for layer in res_layer_list:
                    if isinstance(layer, MaskedConv1d):
                        res, _ = layer(res, res_length)
                    else:
                        res = layer(res)
                residual = res
        if self.res is not None:
            out = out + residual
        out = self.mout(out)
        return out, out_length

class ConvASREncoder(nn.Module):
    def __init__(self, in_channels, blocks_params):
        super().__init__()
        layers = [JasperBlock(**params) for params in blocks_params]
        self.encoder = nn.Sequential(*layers)
        self.in_channels = in_channels

    def forward(self, x, length):
        for layer in self.encoder:
            x, length = layer(x, length)
        return x, length

class AudioToMFCCPreprocessor(nn.Module):
    def __init__(self, sample_rate=16000, n_mels=64, n_mfcc=64, n_fft=512, hop_length=160, f_min=0, f_max=8000):
        super().__init__()
        self.featurizer = T.MFCC(
            sample_rate=sample_rate,
            n_mfcc=n_mfcc,
            melkwargs={
                "n_fft": n_fft,
                "n_mels": n_mels,
                "hop_length": hop_length,
                "f_min": f_min,
                "f_max": f_max,
            },
        )

    def forward(self, x, length):
        with torch.no_grad():
            x = self.featurizer(x)
        return x, length

class ConvASRDecoderClassification(nn.Module):
    def __init__(self, in_features, num_classes):
        super().__init__()
        self.pooling = nn.AdaptiveAvgPool1d(output_size=1)
        self.decoder_layers = nn.Sequential(
            nn.Linear(in_features, num_classes)
        )

    def forward(self, x):
        x = self.pooling(x)
        x = x.squeeze(2)
        x = self.decoder_layers(x)
        return x

class TopKClassificationAccuracy(nn.Module):
    def __init__(self, k=(1,)):
        super().__init__()
        self.k = k

    def forward(self, logits, targets):
        with torch.no_grad():
            maxk = max(self.k)
            batch_size = targets.size(0)
            _, pred = logits.topk(maxk, 1, True, True)
            pred = pred.t()
            correct = pred.eq(targets.view(1, -1).expand_as(pred))
            res = []
            for k in self.k:
                correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
                res.append(correct_k.mul_(100.0 / batch_size))
            return res[0] if len(res) == 1 else res

class EncDecClassificationModel(nn.Module):
    def __init__(self, num_classes, sample_rate=16000, n_mels=64, n_mfcc=64, n_fft=512, hop_length=160, f_min=0, f_max=8000):
        super().__init__()
        self.preprocessor = AudioToMFCCPreprocessor(
            sample_rate=sample_rate, n_mels=n_mels, n_mfcc=n_mfcc,
            n_fft=n_fft, hop_length=hop_length, f_min=f_min, f_max=f_max)
        blocks_params = [
            {"in_channels": n_mfcc, "out_channels": 128, "kernel_size": 11, "stride": 1, "padding": 5, "dilation": 1, "dropout": 0.0, "residual": False},
            {"in_channels": 128, "out_channels": 64, "kernel_size": 13, "stride": 1, "padding": 6, "dilation": 1, "dropout": 0.0, "residual": True},
            {"in_channels": 64, "out_channels": 64, "kernel_size": 15, "stride": 1, "padding": 7, "dilation": 1, "dropout": 0.0, "residual": True},
            {"in_channels": 64, "out_channels": 64, "kernel_size": 17, "stride": 1, "padding": 8, "dilation": 1, "dropout": 0.0, "residual": True},
            {"in_channels": 64, "out_channels": 128, "kernel_size": 29, "stride": 1, "padding": 28, "dilation": 2, "dropout": 0.0, "residual": False},
            {"in_channels": 128, "out_channels": 128, "kernel_size": 1, "stride": 1, "padding": 0, "dilation": 1, "dropout": 0.0, "residual": False},
        ]
        self.encoder = ConvASREncoder(in_channels=n_mfcc, blocks_params=blocks_params)
        self.decoder = ConvASRDecoderClassification(in_features=128, num_classes=num_classes)
        self.loss = nn.CrossEntropyLoss()
        self._accuracy = TopKClassificationAccuracy()

    def forward(self, x, length, y=None):
        x, length = self.preprocessor(x, length)
        x, length = self.encoder(x, length)
        logits = self.decoder(x)
        if y is not None:
            loss = self.loss(logits, y)
            acc = self._accuracy(logits, y)
            return loss, acc, logits
        else:
            return logits

    def predict(self, x, length):
        with torch.no_grad():
            logits = self.forward(x, length)
            return torch.argmax(logits, dim=-1)
