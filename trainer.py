from torch.cuda.amp import autocast
from adversary import AdversaryForEmbedding


class Trainer:
    def __init__(self, model, scaler=None):
        self.model = model
        self.model_uw = model.module if hasattr(model, "module") else model
        if scaler is not None:
            self.fp16 = True
            self.scaler = scaler
        else:
            self.fp16 = False

    def step(self, inputs):
        if self.fp16:
            with autocast():
                outputs = self.model(**inputs)
        else:
            outputs = self.model(**inputs)
        loss = outputs[0].mean()
        
        if self.fp16:
            self.scaler.scale(loss).backward()
        else:
            loss.backward()

        return loss


class FreeLBTrainer:
    def __init__(self, model, scaler=None, adv_steps=2, adv_lr=1e-1, adv_max_norm=1e-1, adv_norm_type="fro", adv_init_var=0):
        self.model = model
        self.model_uw = model.module if hasattr(model, "module") else model
        self.word_embeddings = getattr(self.model_uw, self.model_uw.config.model_type).embeddings.word_embeddings
        self.adv_steps = adv_steps
        self.eadv = AdversaryForEmbedding(adv_lr,
                                          adv_max_norm,
                                          adv_norm_type)
        if scaler is not None:
            self.fp16 = True
            self.scaler = scaler
        else:
            self.fp16 = False

    def step(self, inputs):
        input_ids = inputs["input_ids"]
        inputs["input_ids"] = None
        inputs_embeds = self.word_embeddings(input_ids)
        self.eadv.init(inputs_embeds)
        
        for j in range(self.adv_steps):
            self.eadv.requires_grad()
            inputs["inputs_embeds"] = inputs_embeds + self.eadv.delta
            if self.fp16:
                with autocast():
                    outputs = self.model(**inputs)
            else:
                outputs = self.model(**inputs)
            loss = outputs[0].mean()

            loss = loss / self.adv_steps
            if self.fp16:
                self.scaler.scale(loss).backward()
            else:
                loss.backward()

            if j == self.adv_steps - 1:
                break

            self.eadv.update()
            inputs_embeds = self.model_uw.bert.embeddings.word_embeddings(input_ids)

        return loss
