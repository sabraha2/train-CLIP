import pytorch_lightning as pl
import torch
import torch.nn.functional as F

class HomotopyCLIPModule(pl.LightningModule):
    def __init__(self, model, start_lr=1e-4, end_lr=1e-6, total_steps=1000, temperature=0.07):
        super().__init__()
        self.model = model
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
        
        # Update homotopy parameter
        self.update_homotopy_parameter()
        
        self.log("train_loss", combined_loss, prog_bar=True)
        self.log("homotopy_t", self.t, prog_bar=True)
        return combined_loss

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

    def update_homotopy_parameter(self):
        # Update step and homotopy parameter 't'
        self.current_step += 1
        self.t = self.current_step / self.total_steps

    def configure_optimizers(self):
        # Implement a learning rate scheduler that decreases from start_lr to end_lr
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.start_lr)
        scheduler = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=1.0, end_factor=self.end_lr/self.start_lr, total_iters=self.total_steps)
        return [optimizer], [scheduler]
