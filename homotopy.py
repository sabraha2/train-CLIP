import pytorch_lightning as pl
from torchvision.utils import make_grid
import matplotlib.pyplot as plt
import io
import torch
import torch.nn.functional as F
import math

class HomotopyCLIPModule(pl.LightningModule):
    def __init__(self, model, tokenizer, start_lr=1e-4, end_lr=1e-6, total_steps=1000, temperature=0.07):
        super().__init__()
        self.model = model
        self.tokenizer = tokenizer
        self.start_lr = start_lr
        self.end_lr = end_lr
        self.total_steps = total_steps
        self.current_step = 0
        self.temperature = temperature
        
        # Homotopy parameter initialization
        self.register_buffer("t", torch.tensor(0.0))

    def forward(self, images, texts):
        image_features = self.model.encode_image(images)
        text_features = self.model.encode_text(texts)
        return image_features, text_features

    def training_step(self, batch, batch_idx):
        images, texts = batch
        image_features, text_features = self.forward(images, texts)
        
        # Compute losses
        contrastive_loss = self.compute_contrastive_loss(image_features, text_features)
        similarity_score = self.compute_similarity_score(image_features, text_features)
        similarity_loss = -similarity_score  # Minimize negative similarity
        
        # Combined loss using homotopy parameter
        combined_loss = (1 - self.t) * contrastive_loss + self.t * similarity_loss

        # NaN detection
        if torch.isnan(combined_loss).any():
            self.log('train_loss', float('nan'), prog_bar=True)  # Log NaN loss
            raise pl.utilities.exceptions.CancelBatchException()  # Skip the rest of this training step
       
        # Compute similarities
        logits = torch.matmul(image_features, text_features.t())
        logits_t = logits.t()

        # Calculate image-to-text accuracy
        predictions_i2t = logits.argmax(dim=1)
        correct_i2t = predictions_i2t.eq(torch.arange(logits.size(0), device=self.device)).sum()
        
        # Calculate text-to-image accuracy
        predictions_t2i = logits_t.argmax(dim=1)
        correct_t2i = predictions_t2i.eq(torch.arange(logits_t.size(0), device=self.device)).sum()

        # Aggregate accuracies
        accuracy_i2t = correct_i2t.float() / logits.size(0)
        accuracy_t2i = correct_t2i.float() / logits_t.size(0)
        overall_accuracy = (accuracy_i2t + accuracy_t2i) / 2

        if self.trainer.global_step % 100 == 0:
            # Log images
            grid = make_grid(images)
            self.logger.experiment.add_image('training_images', grid, self.global_step)

            # Process texts to visualize as images in TensorBoard
            texts_to_log = self.process_texts_for_logging(texts)
            self.logger.experiment.add_image('training_texts', texts_to_log, self.global_step)
        
        # Update homotopy parameter
        self.update_homotopy_parameter()
        
        self.log("train_loss", combined_loss, prog_bar=True)
        self.log("train_accuracy", overall_accuracy, on_step=False, on_epoch=True, prog_bar=True)
        self.log("homotopy_t", self.t, prog_bar=True)

        return combined_loss
    
    def process_texts_for_logging(self, texts):
        # Turn text into a list of strings if it's tokenized
        if isinstance(texts, dict):
            texts = self.tokenizer.batch_decode(texts['input_ids'], skip_special_tokens=True)

        # Create a figure and a set of subplots
        fig, ax = plt.subplots()

        # Hide axes
        ax.axis('off')

        # Set the text at the center of the figure
        ax.text(0.5, 0.5, "\n".join(texts), fontsize=12, ha='center')

        # Save the plot to a buffer
        buf = io.BytesIO()
        plt.savefig(buf, format='jpeg')
        buf.seek(0)

        # Ensure to specify the format as 'jpeg' when reading the image
        text_image = plt.imread(buf, format='jpeg')
        plt.close(fig)

        # Convert the image into a torch tensor and permute the dimensions
        # Note: You'll need to adjust the tensor conversion as plt.imread returns a numpy array
        text_image_tensor = torch.from_numpy(text_image).permute(2, 0, 1).float() / 255.0  # Normalize the image
        text_image_tensor = text_image_tensor.unsqueeze(0)  # Add batch dimension

        return text_image_tensor


    def compute_contrastive_loss(self, image_features, text_features):
        """
        Compute the contrastive loss between image and text features using NT-Xent loss.
        """
        labels = torch.arange(text_features.size(0)).long().to(self.device)
        image_features = F.normalize(image_features, dim=1)
        text_features = F.normalize(text_features, dim=1)

        # Compute cosine similarity as dot product in normalized feature space
        logits = torch.matmul(image_features, text_features.t()) * self.temperature
        logits_max, _ = torch.max(logits, dim=1, keepdim=True)
        logits = logits - logits_max.detach()

        # Compute log_prob
        exp_logits = torch.exp(logits)
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))

        # Compute mean log-likelihood for positive
        mean_log_prob_pos = (F.log_softmax(logits, dim=1) * labels).sum(1)

        # NT-Xent loss
        loss = -mean_log_prob_pos.mean()
        return loss

    def compute_similarity_score(self, image_features, text_features):
        """
        Compute the similarity score as the mean cosine similarity between corresponding image and text features.
        """
        image_features = F.normalize(image_features, dim=1)
        text_features = F.normalize(text_features, dim=1)

        # Compute cosine similarity
        similarity = (image_features * text_features).sum(dim=1).mean()
        return similarity

    # def update_homotopy_parameter(self):
    #     # Calculate the total number of training steps across all epochs
    #     total_training_steps = self.total_steps * self.trainer.max_epochs
        
    #     # Calculate the current overall step across all epochs
    #     current_overall_step = self.trainer.global_step + self.trainer.current_epoch * self.total_steps
        
    #     # Calculate the fraction of training completed
    #     current_progress = current_overall_step / total_training_steps
        
    #     # Ensure the progress fraction remains within [0, 1]
    #     self.t = torch.tensor(max(0.0, min(current_progress, 1.0)), device=self.t.device)

    import math

    def update_homotopy_parameter(self):
        # Calculate the total number of training steps across all epochs
        total_training_steps = self.total_steps * self.trainer.max_epochs
        
        # Calculate the current overall step across all epochs
        current_overall_step = self.trainer.global_step + self.trainer.current_epoch * self.total_steps
        
        # Instead of linearly increasing 't', use a sigmoid function to get a smooth transition
        # Compute a 'progress' value that goes from -6 to 6 over the training steps, 
        # which corresponds to the steep part of the sigmoid function
        progress = (current_overall_step / total_training_steps - 0.5) * 12
        sigmoid_t = 1 / (1 + math.exp(-progress))
        
        # Update the homotopy parameter 't'
        self.t = torch.tensor(sigmoid_t, device=self.device)


    # def configure_optimizers(self):
    #     # Implement a learning rate scheduler that decreases from start_lr to end_lr
    #     optimizer = torch.optim.Adam(self.model.parameters(), lr=self.start_lr)
    #     scheduler = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=1.0, end_factor=self.end_lr/self.start_lr, total_iters=self.total_steps)
    #     return [optimizer], [scheduler]

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.start_lr)
        scheduler = {
            'scheduler': torch.optim.lr_scheduler.LinearLR(
                optimizer, start_factor=1.0, end_factor=self.end_lr/self.start_lr, total_iters=self.total_steps
            ),
            'interval': 'step',  # or 'epoch' depending on how you've scheduled it
            'frequency': 1,
        }
        return {
            'optimizer': optimizer,
            'lr_scheduler': scheduler,
            'gradient_clip_val': 1.0,  # add gradient clipping here
        }