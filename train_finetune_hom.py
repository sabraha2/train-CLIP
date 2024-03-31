import torch
from argparse import ArgumentParser
from pytorch_lightning import Trainer
from data.text_image_dm import TextImageDataModule
from models import CustomCLIPWrapper
from torchvision.models import resnet50
from transformers import AutoTokenizer, AutoModel
from homotopy import HomotopyCLIPModule


def main(hparams):
    torch.autograd.set_detect_anomaly(True)  # Enable anomaly detection to find the operations that produce NaNs

    img_encoder = resnet50(pretrained=True)
    img_encoder.fc = torch.nn.Linear(2048, 768)

    tokenizer = AutoTokenizer.from_pretrained("johngiorgi/declutr-sci-base")
    txt_encoder = AutoModel.from_pretrained("johngiorgi/declutr-sci-base")

    if hparams.minibatch_size < 1:
        hparams.minibatch_size = hparams.batch_size

    model = CustomCLIPWrapper(img_encoder, txt_encoder, hparams.minibatch_size, avg_word_embs=True)
    dm = TextImageDataModule.from_argparse_args(hparams, custom_tokenizer=tokenizer)
    hom_model = HomotopyCLIPModule(model, tokenizer=tokenizer)
    
    # Update Trainer initialization with gradient clipping and terminate_on_nan
    trainer = Trainer.from_argparse_args(
        hparams,
        precision=16,  # Consider using 32 to disable mixed precision if NaNs persist
        max_epochs=32,
        accelerator='gpu',
        gpus=1,
        logger=True,
        gradient_clip_val=1.0,  # Add gradient clipping
        terminate_on_nan=True,  # Stop training on NaNs
    )
    
    trainer.fit(hom_model, dm)


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--minibatch_size', type=int, default=0)
    parser = TextImageDataModule.add_argparse_args(parser)
    parser = Trainer.add_argparse_args(parser)
    args = parser.parse_args()

    main(args)
