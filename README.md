# GPT2WithExtraAttention

```python
import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM

class GPT2WithExtraAttention(nn.Module):
    def __init__(self, model_name="gpt2"):
        super().__init__()
        # Load GPT-2 with its LM head
        self.gpt2 = AutoModelForCausalLM.from_pretrained(model_name)

        # Freeze everything
        for p in self.gpt2.parameters():
            p.requires_grad = False

        hidden_size = self.gpt2.config.hidden_size
        num_heads = self.gpt2.config.n_head

        # Add one new trainable attention layer
        self.new_attention = nn.MultiheadAttention(
            embed_dim=hidden_size,
            num_heads=num_heads,
            batch_first=True
        )

    def forward(self, input_ids, attention_mask=None, labels=None):
        # Get hidden states from GPT-2 transformer
        outputs = self.gpt2.transformer(input_ids, attention_mask=attention_mask)
        hidden_states = outputs.last_hidden_state  # [batch, seq_len, hidden]

        # Apply the new attention layer
        attn_out, _ = self.new_attention(hidden_states, hidden_states, hidden_states)

        # Reuse GPT-2’s LM head
        logits = self.gpt2.lm_head(attn_out)

        loss = None
        if labels is not None:
            shift_logits = logits[:, :-1, :].contiguous()
            shift_labels = labels[:, 1:].contiguous()
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1)
            )

        return {"loss": loss, "logits": logits}

```
