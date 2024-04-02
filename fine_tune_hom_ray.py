import torch
from argparse import ArgumentParser
from pytorch_lightning import Trainer
from data.text_image_dm import TextImageDataModule
from models import CustomCLIPWrapper
from torchvision.models import resnet50
from transformers import AutoTokenizer, AutoModel
from ray import tune
from ray.tune.integration.pytorch_lightning import TuneReportCallback


def train_model(config, checkpoint_dir=None):
    img_encoder = resnet50(pretrained=True)
    img_encoder.fc = torch.nn.Linear(2048, 768)

    tokenizer = AutoTokenizer.from_pretrained("johngiorgi/declutr-sci-base")
    txt_encoder = AutoModel.from_pretrained("johngiorgi/declutr-sci-base")

    if config['minibatch_size'] < 1:
        config['minibatch_size'] = config['batch_size']

    model = CustomCLIPWrapper(img_encoder, txt_encoder, config['minibatch_size'], avg_word_embs=True)
    dm = TextImageDataModule.from_argparse_args(config, custom_tokenizer=tokenizer)
    
    trainer = Trainer(
        max_epochs=config['max_epochs'],
        gpus=1,
        precision=16,
        accelerator='gpu',
        logger=True,
        callbacks=[TuneReportCallback({"loss": "ptl/val_loss"})]
    )
    
    trainer.fit(model, dm)


def main(config):
    tune_config = {
        "max_epochs": config['max_epochs'],
        "minibatch_size": tune.choice([16, 32, 64]),  # Define the search space for minibatch_size
        "batch_size": config['batch_size'],  # Assuming batch_size is fixed
        # Add other hyperparameters to tune here
    }

    analysis = tune.run(
        train_model,
        config=tune_config,
        stop={"training_iteration": 10},
        resources_per_trial={"gpu": 1},
        checkpoint_at_end=True
    )

    best_trial = analysis.get_best_trial("loss", "min", "last")
    print("Best hyperparameters found were: ", best_trial.config)


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--max_epochs', type=int, default=10)
    parser.add_argument('--batch_size', type=int, default=32)
    # Add other hyperparameters to the argument parser here

    args, _ = parser.parse_known_args()

    main(vars(args))
