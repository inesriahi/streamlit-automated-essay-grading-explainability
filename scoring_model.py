import torch
import torch.nn as nn
from transformers import AutoModel, AutoConfig
import shap
import numpy as np
# ====================================================
# CFG
# ====================================================
class CFG:
    num_workers=4
    path="./FPE_train_model/"
    model="microsoft/deberta-v3-xsmall"
    batch_size=1
    fc_dropout=0.0
    model_config={
        'attention_dropout':0.0,
        'attention_probs_dropout_prob':0.0,
        'hidden_dropout':0.0,
        'hidden_dropout_prob':0.0,
        #'layer_norm_eps':1e-7,
    }
    target_size=6
    target_cols=['cohesion', 'syntax', 'vocabulary', 'phraseology', 'grammar', 'conventions']
    max_len=512
    seed=42
    n_fold=5
    trn_fold=[0]
    train=True
    
from text_unidecode import unidecode
from typing import Dict, List, Tuple
import codecs

def replace_encoding_with_utf8(error: UnicodeError) -> Tuple[bytes, int]:
    return error.object[error.start : error.end].encode("utf-8"), error.end


def replace_decoding_with_cp1252(error: UnicodeError) -> Tuple[str, int]:
    return error.object[error.start : error.end].decode("cp1252"), error.end

# Register the encoding and decoding error handlers for `utf-8` and `cp1252`.
codecs.register_error("replace_encoding_with_utf8", replace_encoding_with_utf8)
codecs.register_error("replace_decoding_with_cp1252", replace_decoding_with_cp1252)

def resolve_encodings_and_normalize(text: str) -> str:
    """Resolve the encoding problems and normalize the abnormal characters."""
    text = (
        text.encode("raw_unicode_escape")
        .decode("utf-8", errors="replace_decoding_with_cp1252")
        .encode("cp1252", errors="replace_encoding_with_utf8")
        .decode("utf-8", errors="replace_decoding_with_cp1252")
    )
    return unidecode(text)



# ====================================================
# tokenizer
# ====================================================
from transformers.models.deberta_v2.tokenization_deberta_v2_fast import DebertaV2TokenizerFast

tokenizer = DebertaV2TokenizerFast.from_pretrained(f'{CFG.path}tokenizer/')
tokenizer.add_special_tokens(
    {"additional_special_tokens": ['[BR]']}
)
CFG.tokenizer = tokenizer

# ====================================================
# Model
# ====================================================
class MeanPooling(nn.Module):
    def __init__(self):
        super(MeanPooling, self).__init__()
        
    def forward(self, last_hidden_state, attention_mask):
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
        sum_embeddings = torch.sum(last_hidden_state * input_mask_expanded, 1)
        sum_mask = input_mask_expanded.sum(1)
        sum_mask = torch.clamp(sum_mask, min=1e-9)
        mean_embeddings = sum_embeddings / sum_mask
        return mean_embeddings

class CustomModel(nn.Module):
    def __init__(self, cfg, config_path=None, pretrained=False):
        super().__init__()
        self.cfg = cfg
        if config_path is None:
            self.config = AutoConfig.from_pretrained(cfg.model, output_hidden_states=True)
        else:
            self.config = torch.load(config_path)
        
        self.config.update(CFG.model_config)
        
        if pretrained:
            self.model = AutoModel.from_pretrained(cfg.model, config=self.config)
        else:
            self.model = AutoModel.from_config(self.config)
        self.model.gradient_checkpointing_enable()
        self.pool = MeanPooling()
        self.fc = nn.Linear(self.config.hidden_size, self.cfg.target_size)
        self._init_weights(self.fc)
        self.layer_norm1 = nn.LayerNorm(self.config.hidden_size)
        
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        
    def feature(self, inputs):
        outputs = self.model(**inputs)
        last_hidden_states = outputs[0]
        feature = self.pool(last_hidden_states, inputs['attention_mask'])
        return feature

    def forward(self, inputs):
        feature = self.feature(inputs)
        feature = self.layer_norm1(feature)
        output = self.fc(feature)
        return output
    
model = CustomModel(CFG, config_path=f'{CFG.path}config.pth', pretrained=False)
state = torch.load(
    f"{CFG.path}{CFG.model.replace('/', '-')}_fold0_best.pth",
    map_location=torch.device('cpu'),
)
model.load_state_dict(state['model'])

def inference_essay_raw_scores(essay, model=model, device='cpu'):
    essay_pred = essay.replace('\n', '[BR]')
    model.eval()
    model.to(device)

    tokenized = tokenizer(essay_pred, padding=True, truncation=True, return_tensors="pt")
    for k, v in tokenized.items():
        tokenized[k] = v.to(device)
    with torch.no_grad():
        y_preds = model(tokenized)

    return y_preds.to('cpu').numpy()
    
# this defines an explicit python function that takes a list of strings and outputs scores for each class
def predict_sentiment_scores(essays):
    inputs_ = [CFG.tokenizer.encode_plus(
        essay, 
        return_tensors='pt', 
        add_special_tokens=True,
        padding="do_not_pad",
        max_length=CFG.max_len,
        truncation=True
    ) for essay in essays]
    scores = []
    for inputs in inputs_:
        outputs = model(inputs).detach().cpu().numpy()
        scores.append(outputs)
    return np.vstack(scores)

def inference_essay(essay):
    result_array = inference_essay_raw_scores(essay)
    return dict(zip(CFG.target_cols, result_array[0]))

def shap_explainer(tokenizer=tokenizer, output_names=CFG.target_cols):
    return shap.Explainer(predict_sentiment_scores, tokenizer, output_names=output_names)

def get_shap_values(essay):
    explainer = shap_explainer()
    return explainer(essay)