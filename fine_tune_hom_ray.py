import torch
from argparse import ArgumentParser
from pytorch_lightning import Trainer
from data.text_image_dm import TextImageDataModule
from models import CustomCLIPWrapper
from torchvision.models import resnet50
from transformers import AutoTokenizer, AutoModel
from ray import tune
from ray.tune.integration.pytorch_lightning import TuneReportCallback
from ray.tune.suggest.bayesopt import BayesOptSearch

class DictToObject:
    def __init__(self, **entries):
        self.__dict__.update(entries)

def train_model(config, checkpoint_dir=None):
    img_encoder = resnet50(pretrained=True)
    img_encoder.fc = torch.nn.Linear(2048, 768)

    tokenizer = AutoTokenizer.from_pretrained("johngiorgi/declutr-sci-base")
    txt_encoder = AutoModel.from_pretrained("johngiorgi/declutr-sci-base")

    if config['minibatch_size'] < 1:
        config['minibatch_size'] = config['batch_size']

    model = CustomCLIPWrapper(img_encoder, txt_encoder, config['minibatch_size'], avg_word_embs=True)
    config['folder'] = "/project01/cvrl/sabraha2/DSIAC_CLIP_DATA/"
    config_object = DictToObject(**config)
    dm = TextImageDataModule.from_argparse_args(config_object, custom_tokenizer=tokenizer)
    
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
        "max_epochs": tune.randint(5, 20),  # Uniform distribution for max_epochs
        "minibatch_size": tune.choice([16, 32, 64]),  
        "batch_size": config['batch_size'],
        "learning_rate": tune.loguniform(1e-5, 1e-3),  
        "weight_decay": tune.loguniform(1e-6, 1e-2),  
        "optimizer": tune.choice(["adam", "sgd"]),
        # Add other hyperparameters to tune here
    }

    analysis = tune.run(
        train_model,
        config=tune_config,
        search_alg=BayesOptSearch(),
        stop={"training_iteration": 10},
        resources_per_trial={"gpu": 1},
        num_samples=10,
    )

    best_trial = analysis.get_best_trial("loss", "min", "last")
    print("Best hyperparameters found were: ", best_trial.config)

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--max_epochs', type=int, default=10)
    parser.add_argument('--batch_size', type=int, default=32)

    args = parser.parse_args()

    main(vars(args))
