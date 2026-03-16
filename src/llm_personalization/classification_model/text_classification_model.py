import torch
from transformers import AutoConfig, AutoModelForSequenceClassification, AutoTokenizer
from tqdm import tqdm
import gc

class TextClassificationModel:
    def __init__(self, num_classes: int, pooling: str = "mean", base_model: str = "answerdotai/ModernBERT-base", max_length: int | None = None):
        self.base_model = base_model
        self.num_classes = num_classes
        self.pooling = pooling
        self._max_length_override = max_length

    def _tokenize(self, texts: list[str], track_truncation: bool = False) -> dict:
        inputs = self.tokenizer(texts, return_tensors="pt", padding=True, truncation=True, max_length=self.max_length)
        if track_truncation:
            untrunc = self.tokenizer(texts, padding=False, truncation=False)
            lengths = [len(ids) for ids in untrunc["input_ids"]]
            truncated = [(l - self.max_length) for l in lengths if l > self.max_length]
            self._trunc_total += len(lengths)
            self._trunc_count += len(truncated)
            self._trunc_excess += sum(truncated)
        return inputs

    def _report_truncation(self):
        if self._trunc_count > 0:
            avg_excess = self._trunc_excess / self._trunc_count
            print(f"  [Truncation] {self._trunc_count}/{self._trunc_total} sequences truncated "
                  f"(max_length={self.max_length}, avg excess tokens={avg_excess:.0f})")
        self._trunc_total = self._trunc_count = self._trunc_excess = 0

    def forward(self, texts: list[str]) -> torch.Tensor:
        inputs = self._tokenize(texts)
        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
        return self.model(**inputs).logits

    def predict(self, texts: list[str], batch_size: int | None = None) -> list[int]:
        if self.model is None:
            raise ValueError("Model not loaded")
        self.model.eval()
        if batch_size is None:
            with torch.no_grad():
                logits = self.forward(texts)
                return logits.argmax(dim=-1).tolist()
        all_preds = []
        with torch.no_grad():
            for i in range(0, len(texts), batch_size):
                logits = self.forward(texts[i:i + batch_size])
                all_preds.extend(logits.argmax(dim=-1).tolist())
        return all_preds

    def train(self, texts: list[str], labels: list[int],
              val_texts: list[str] | None = None, val_labels: list[int] | None = None,
              batch_size: int = 8, grad_accum_steps: int = 8, epochs: int = 3, lr: float = 2e-5):
        if self.model is None:
            raise ValueError("Model not loaded")
        self.model.train()
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=lr)

        self._trunc_total = self._trunc_count = self._trunc_excess = 0

        for epoch in range(epochs):
            self.model.train()
            total_loss = 0
            indices = torch.randperm(len(texts)).tolist()

            optimizer.zero_grad()
            for step, i in enumerate(tqdm(range(0, len(texts), batch_size), desc=f"Epoch {epoch}")):
                batch_indices = indices[i:i + batch_size]
                batch_texts = [texts[j] for j in batch_indices]
                batch_labels = torch.tensor([labels[j] for j in batch_indices], device=self.model.device)

                inputs = self._tokenize(batch_texts, track_truncation=(epoch == 0))
                inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
                inputs["labels"] = batch_labels

                loss = self.model(**inputs).loss / grad_accum_steps
                loss.backward()
                total_loss += loss.item() * grad_accum_steps

                if (step + 1) % grad_accum_steps == 0 or i + batch_size >= len(texts):
                    optimizer.step()
                    optimizer.zero_grad()

            avg_train_loss = total_loss / (len(texts) / batch_size)
            if epoch == 0:
                self._report_truncation()
            
            if val_texts and val_labels:
                # Validation loss
                self.model.eval()
                with torch.no_grad():
                    val_inputs = self._tokenize(val_texts)
                    val_inputs = {k: v.to(self.model.device) for k, v in val_inputs.items()}
                    val_inputs["labels"] = torch.tensor(val_labels, device=self.model.device)
                    val_loss = self.model(**val_inputs).loss.item()
                
                # Validation accuracy
                val_preds = self.predict(val_texts)
                val_acc = sum(p == l for p, l in zip(val_preds, val_labels)) / len(val_labels)
                print(f"  Epoch {epoch}: train_loss={avg_train_loss:.4f}, val_loss={val_loss:.4f}, val_acc={val_acc:.4f}")
            else:
                print(f"  Epoch {epoch}: train_loss={avg_train_loss:.4f}")

    def save_to_file(self, path: str):
        if self.model is None:
            raise ValueError("Model not loaded")
        self.model.save_pretrained(path)
        self.tokenizer.save_pretrained(path)


    def load_from_file(self, path: str):
        self.model = AutoModelForSequenceClassification.from_pretrained(path)
        self.tokenizer = AutoTokenizer.from_pretrained(path)
        self.max_length = self._max_length_override or self.model.config.max_position_embeddings
        if torch.cuda.is_available():
            self.model = self.model.cuda()

    def load_untrained(self):
        self.tokenizer = AutoTokenizer.from_pretrained(self.base_model)
        config = AutoConfig.from_pretrained(self.base_model)
        config.classifier_pooling = self.pooling
        config.num_labels = self.num_classes
        self.max_length = self._max_length_override or config.max_position_embeddings
        self.model = AutoModelForSequenceClassification.from_pretrained(self.base_model, config=config)
        if torch.cuda.is_available():
            self.model = self.model.cuda()


    def unload(self) -> None:
        if self.model is None:
            return
        
        del self.model
        del self.tokenizer
        self.model = None
        self.tokenizer = None
        
        gc.collect()

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            for i in range(torch.cuda.device_count()):
                free, total = torch.cuda.mem_get_info(i)
                print(f"[TextClassificationModel] GPU {i}: {free/1024**3:.1f}/{total/1024**3:.1f} GiB free after unload")

        print("[TextClassificationModel] Model unloaded")
