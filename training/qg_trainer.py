from torch.utils.data import DataLoader, Dataset
from torch.optim import AdamW
import torch
from transformers import AutoTokenizer
from tqdm import tqdm



class Trainer():

    def __init__(
        self,
        device: str,
        epochs: int,
        learning_rate: float,
        model: torch.nn.Module,
        tokenizer: AutoTokenizer,
        save_dir: str,
        train_batch_size: int,
        train_set: Dataset,
        valid_batch_size: int,
        valid_set: Dataset,
    ) -> None:

        self.device = device
        self.epochs = epochs
        self.save_dir = save_dir
        self.train_loss = 0
        self.train_batch_size = train_batch_size
        self.valid_batch_size = valid_batch_size
        self.train_loader = DataLoader(
            train_set,
            batch_size=train_batch_size,
            shuffle=True
        )
        self.valid_loader = DataLoader(
            valid_set,
            batch_size=train_batch_size,
            shuffle=False
        )
        self.tokenizer = tokenizer
        self.model = model.to(self.device)
        self.optimizer = AdamW(self.model.parameters(), lr=learning_rate)

    def train(self)-> None:

        """
        Function to be called for training with the parameters passed from main function

        """
        self.best_valid_score = float("inf")
        for epoch in range(1, self.epochs+1):
            self.model.train()
            with tqdm(total=len(self.train_loader), unit="batches") as tepoch:
                tepoch.set_description(f"epoch {epoch}")

                for x in self.train_loader:

                    self.optimizer.zero_grad()
                    data = {key: value.to(self.device) for key, value in x.items()}
                    outputs = self.model(**data)
                    loss = outputs[0]
                    loss.backward()
                    self.optimizer.step()
                    tepoch.update(1)


            valid_loss = self.evaluate(self.valid_loader)
            if valid_loss < self.best_valid_score:
                print(
                    f"Validation loss decreased from {self.best_valid_score:.8f} to {valid_loss:.8f}. Saving.")
                self.best_valid_score = valid_loss
                self._save()
                    


    def evaluate(self, dataloader: DataLoader) -> float:
        self.model.eval()
        eval_loss = 0
        with tqdm(total=len(dataloader), unit="batches") as tepoch:
            tepoch.set_description("validation")
            for x in dataloader:
                data = {key: value.to(self.device) for key, value in x.items()}
                output = self.model(**data)
                eval_loss = output[0]
                tepoch.set_postfix({"valid_loss": eval_loss})
                tepoch.update(1)
        return eval_loss


    def _save(self) -> None:
        self.tokenizer.save_pretrained(self.save_dir)
        self.model.save_pretrained(self.save_dir)