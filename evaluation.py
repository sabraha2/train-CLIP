import torch
from torchvision.models import resnet50
from transformers import AutoTokenizer, AutoModel
from models import CustomCLIPWrapper
from data.text_image_dm import TextImageDataModule
from argparse import ArgumentParser
from torchvision.transforms import Compose, Resize, ToTensor, Normalize
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.manifold import TSNE
import seaborn as sns
import pandas as pd

class EvaluationScript:
    def __init__(self, model_checkpoint_path, dataset_path, batch_size=32):
        self.model_checkpoint_path = model_checkpoint_path
        self.dataset_path = dataset_path
        self.batch_size = batch_size
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained("johngiorgi/declutr-sci-base")
        self.img_encoder, self.txt_encoder = self.create_encoders()
        self.model = self.load_model()
        self.model.to(self.device)

    def create_encoders(self):
        img_encoder = resnet50(pretrained=True)
        img_encoder.fc = torch.nn.Linear(2048, 768)
        txt_encoder = AutoModel.from_pretrained("johngiorgi/declutr-sci-base")
        return img_encoder, txt_encoder

    def load_model(self):
        # Initialize the image and text encoders
        img_encoder = resnet50(pretrained=True)
        img_encoder.fc = torch.nn.Linear(2048, 768)  # Adjust this according to your model's architecture
        
        txt_encoder = AutoModel.from_pretrained("johngiorgi/declutr-sci-base")

        # Load the CustomCLIPWrapper model from the checkpoint with the correct arguments
        # Assuming CustomCLIPWrapper has been adjusted to accept these parameters on initialization
        model = CustomCLIPWrapper.load_from_checkpoint(
            checkpoint_path=self.model_checkpoint_path,
            image_encoder=img_encoder,  # These are the additional arguments needed
            text_encoder=txt_encoder,
            minibatch_size=32,  # Ensure this matches your model's expected batch size
            avg_word_embs=True  # This should match how your model was originally configured
        )
        
        model.eval()
        return model

    def evaluate(self):
        # Initialize the data module
        data_module = TextImageDataModule(
            folder=self.dataset_path,
            batch_size=self.batch_size,
            image_size=224,
            resize_ratio=0.75,
            shuffle=False,
            custom_tokenizer=self.tokenizer
        )
        data_module.setup()

        test_loader = data_module.val_dataloader()

        image_embeddings, text_embeddings, labels = [], [], []

        for images, texts, label in test_loader:
            images, texts = images.to(self.device), texts.to(self.device)
            with torch.no_grad():
                image_embed = self.model.encode_image(images)
                text_embed = self.model.encode_text(texts)

            image_embeddings.append(image_embed.cpu().numpy())
            text_embeddings.append(text_embed.cpu().numpy())
            labels.extend(label.cpu().numpy())

        image_embeddings = np.concatenate(image_embeddings, axis=0)
        text_embeddings = np.concatenate(text_embeddings, axis=0)
        self.compute_metrics(image_embeddings, text_embeddings, labels)
        self.visualize_embeddings(image_embeddings, text_embeddings, labels)

    def compute_metrics(self, image_embeddings, text_embeddings, labels):
        # Assuming your labels are correctly aligned with the embeddings
        similarity = cosine_similarity(image_embeddings, text_embeddings)
        predicted_indices = np.argmax(similarity, axis=1)
        true_indices = np.arange(len(labels))

        accuracy = accuracy_score(true_indices, predicted_indices)
        precision, recall, f1, _ = precision_recall_fscore_support(true_indices, predicted_indices, average='macro')

        print(f"Accuracy: {accuracy:.4f}")
        print(f"Precision: {precision:.4f}, Recall: {recall:.4f}, F1 Score: {f1:.4f}")

    def visualize_embeddings(self, image_embeddings, text_embeddings, labels):
        tsne = TSNE(n_components=2, random_state=42)
        tsne_results = tsne.fit_transform(np.concatenate((image_embeddings, text_embeddings), axis=0))

        labels_double = ['Image']*len(image_embeddings) + ['Text']*len(text_embeddings)
        self.plot_tsne(tsne_results, labels_double, title='CLIP Embeddings t-SNE Visualization')

    @staticmethod
    def plot_tsne(tsne_results, labels, title):
        df_subset = pd.DataFrame()
        df_subset['tsne-2d-one'] = tsne_results[:, 0]
        df_subset['tsne-2d-two'] = tsne_results[:, 1]
        df_subset['label'] = labels

        plt.figure(figsize=(16,10))
        sns.scatterplot(
            x="tsne-2d-one", y="tsne-2d-two",
            hue="label",
            palette=sns.color_palette("hsv", 2),
            data=df_subset,
            legend="full",
            alpha=0.3
        )
        plt.title(title)
        plt.savefig('tsne_visualization.png')

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--model_checkpoint_path", type=str, required=True, help="Path to the model checkpoint file")
    parser.add_argument("--dataset_path", type=str, required=True, help="Path to the dataset")
    parser.add_argument("--batch_size", type=int, default=32, help="Evaluation batch size")
    args = parser.parse_args()

    eval_script = EvaluationScript(args.model_checkpoint_path, args.dataset_path, args.batch_size)
    eval_script.evaluate()
