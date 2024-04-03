import torch
from torchvision.models import resnet50
from transformers import AutoTokenizer, AutoModel
from models import CustomCLIPWrapper
from data.text_image_dm import TextImageDataModule
from argparse import ArgumentParser
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.manifold import TSNE
import seaborn as sns
import logging

class EvaluationScript:
    def __init__(self, model_checkpoint_path, test_dataset_path, batch_size=32):
        self.model_checkpoint_path = model_checkpoint_path
        self.test_dataset_path = test_dataset_path
        self.batch_size = batch_size
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained("johngiorgi/declutr-sci-base")
        self.img_encoder, self.txt_encoder = self.create_encoders()
        self.model = self.load_model()
        self.model.to(self.device)
        
        # Configure logging
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.INFO)
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        console_handler.setFormatter(formatter)
        self.logger.addHandler(console_handler)

    def create_encoders(self):
        img_encoder = resnet50(pretrained=True)
        img_encoder.fc = torch.nn.Linear(2048, 768)
        txt_encoder = AutoModel.from_pretrained("johngiorgi/declutr-sci-base")
        return img_encoder, txt_encoder

    def load_model(self):
        model = CustomCLIPWrapper.load_from_checkpoint(
            checkpoint_path=self.model_checkpoint_path,
            image_encoder=self.img_encoder,
            text_encoder=self.txt_encoder,
            minibatch_size=32,
            avg_word_embs=True
        )
        model.eval()
        return model

    def evaluate(self):
        self.logger.info("Starting evaluation...")
        
        data_module = TextImageDataModule(
            folder=self.test_dataset_path,
            test_folder=self.test_dataset_path,
            batch_size=self.batch_size,
            image_size=224,
            resize_ratio=0.75,
            shuffle=False,
            custom_tokenizer=self.tokenizer
        )
        data_module.setup(stage='test')
        
        test_loader = data_module.test_dataloader()
        
        self.logger.info("Test dataset loaded.")
        
        image_embeddings, text_embeddings = [], []

        for batch_idx, batch in enumerate(test_loader):
            self.logger.info(f"Processing batch {batch_idx + 1}...")
            images, texts = batch  # Directly unpacking the batch

            images = images.to(self.device)
            # Prepare text inputs properly for the encode_text method
            inputs = self.tokenizer(texts, return_tensors="pt", padding=True, truncation=True, max_length=512)
            inputs = {key: value.to(self.device) for key, value in inputs.items()}

            with torch.no_grad():
                image_embed = self.model.encode_image(images)
                text_embed = self.model.encode_text(inputs)

            image_embeddings.append(image_embed.cpu().numpy())
            text_embeddings.append(text_embed.cpu().numpy())

        self.logger.info("Data processing completed.")

        if len(image_embeddings) == 0 or len(text_embeddings) == 0:
            self.logger.error("No embeddings found.")
            return

        image_embeddings = np.concatenate(image_embeddings, axis=0)
        text_embeddings = np.concatenate(text_embeddings, axis=0)
        self.analyze_similarity_distribution(image_embeddings, text_embeddings)
        # Call the t-SNE visualization method
        self.visualize_tsne_embeddings(image_embeddings, text_embeddings)
        # Assuming you have adjusted data loading to include paths and descriptions
        # self.visualize_top_n_matches(image_embeddings, text_embeddings, image_paths, text_descriptions, n=5)
        self.logger.info("Evaluation completed.")
    
    def visualize_tsne_embeddings(self, image_embeddings, text_embeddings):
        """Visualize t-SNE plot of image and text embeddings."""
        self.logger.info("Generating t-SNE visualization...")
        
        # Combine image and text embeddings
        combined_embeddings = np.vstack((image_embeddings, text_embeddings))
        
        # Perform t-SNE
        tsne_results = TSNE(n_components=2, random_state=42).fit_transform(combined_embeddings)
        
        # Split back into image and text embeddings for plotting
        tsne_images, tsne_texts = tsne_results[:len(image_embeddings)], tsne_results[len(image_embeddings):]
        
        plt.figure(figsize=(12, 8))
        
        plt.scatter(tsne_images[:, 0], tsne_images[:, 1], c='blue', label='Images', alpha=0.5)
        plt.scatter(tsne_texts[:, 0], tsne_texts[:, 1], c='red', label='Texts', alpha=0.5)
        
        plt.legend()
        plt.title("t-SNE visualization of Image and Text Embeddings", fontsize=15)
        plt.xlabel("t-SNE 1", fontsize=12)
        plt.ylabel("t-SNE 2", fontsize=12)
        
        plt.grid(True, which='both', linestyle='--', linewidth=0.5)
        plt.tight_layout()
        plt.savefig('tsne_embeddings_visualization.png')
        plt.show()
    
    def visualize_top_n_matches(self, image_embeddings, text_embeddings, image_paths, text_descriptions, n=5):
        """
        Visualizes top-N text descriptions that are most similar to each image.
        Args:
        - image_embeddings: Embeddings of the images.
        - text_embeddings: Embeddings of the texts.
        - image_paths: Paths to the images for loading and displaying.
        - text_descriptions: Corresponding descriptions for each text embedding.
        - n: Number of top matches to visualize.
        """
        self.logger.info("Visualizing top-N text matches for images...")

        # Compute cosine similarity and get top N indices for texts per image
        similarity = cosine_similarity(image_embeddings, text_embeddings)
        top_n_indices = np.argsort(similarity, axis=1)[:, -n:][::-1]

        # Randomly select a few images to visualize
        selected_images = np.random.choice(len(image_embeddings), size=min(3, len(image_embeddings)), replace=False)

        for image_index in selected_images:
            image = plt.imread(image_paths[image_index])
            plt.figure(figsize=(5 + 3*n, 4))

            plt.subplot(1, n+1, 1)
            plt.imshow(image)
            plt.title("Image")
            plt.axis('off')

            for rank, text_index in enumerate(top_n_indices[image_index], start=1):
                plt.subplot(1, n+1, rank+1)
                text = text_descriptions[text_index]
                plt.text(0.5, 0.5, text, ha='center', va='center')
                plt.title(f"Match {rank}")
                plt.axis('off')

            plt.tight_layout()
            plt.savefig(f"top_{n}_matches_image_{image_index}.png")
            plt.show()



    def analyze_similarity_distribution(self, image_embeddings, text_embeddings):
        similarity = cosine_similarity(image_embeddings, text_embeddings)
        similarity_values = similarity.flatten()
        self.visualize_similarity_distribution(similarity_values)

    def visualize_similarity_distribution(self, similarity_values):
        plt.figure(figsize=(10, 6))
        sns.histplot(similarity_values, bins=50, kde=True, color='skyblue', edgecolor='black')
        plt.title('Distribution of Cosine Similarities', fontsize=16)
        plt.xlabel('Cosine Similarity', fontsize=14)
        plt.ylabel('Frequency', fontsize=14)
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.tight_layout()
        plt.savefig('similarity_distribution.png')
        plt.show()

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--model_checkpoint_path", type=str, required=True, help="Path to the model checkpoint file")
    parser.add_argument("--test_dataset_path", type=str, required=True, help="Path to the test dataset")
    parser.add_argument("--batch_size", type=int, default=32, help="Evaluation batch size")
    args = parser.parse_args()

    eval_script = EvaluationScript(args.model_checkpoint_path, args.test_dataset_path, args.batch_size)
    eval_script.evaluate()
