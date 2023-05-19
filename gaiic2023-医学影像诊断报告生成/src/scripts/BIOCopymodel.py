from transformers.models.bart.modeling_bart import BartForConditionalGeneration,shift_tokens_right
from transformers.models.bart.configuration_bart import BartConfig
from transformers.modeling_outputs import Seq2SeqLMOutput, ModelOutput
from transformers.utils import logging
from typing import Optional, Union, Tuple, Dict, Any, List
from dataclasses import dataclass
import torch
import torch.nn as nn
import collections
from scripts.utils import Sparsemax

logger = logging.get_logger(__name__)

@dataclass
class Seq2SeqLMOutputWithBio(Seq2SeqLMOutput):
    enc_bio_loss: Optional[torch.FloatTensor] = None
    dec_bio_loss: Optional[torch.FloatTensor] = None
    src_input_ids: Optional[Tuple[torch.LongTensor]] = None
    src_acc: Optional[torch.FloatTensor] = None
    tgt_acc: Optional[torch.FloatTensor] = None
    state: torch.FloatTensor = None
    
class BartForConditionalGenerationWithBioCopy(BartForConditionalGeneration):
    def __init__(self, config: BartConfig):
        super().__init__(config)
        
        self.enc_bio_fc = nn.Linear(config.d_model, 3)
        self.dec_bio_fc = nn.Linear(config.d_model, 3)
        self.post_init()
        
    def _prepare_encoder_decoder_kwargs_for_generation(self, inputs_tensor: torch.Tensor, model_kwargs, model_input_name: Optional[str] = None) -> Dict[str, Any]:
        # 1. get encoder
        encoder = self.get_encoder()

        # 2. prepare encoder args and encoder kwargs from model kwargs
        irrelevant_prefix = ["decoder_", "cross_attn", "use_cache"]
        encoder_kwargs = {
            argument: value
            for argument, value in model_kwargs.items()
            if not any(argument.startswith(p) for p in irrelevant_prefix)
        }

        # 3. make sure that encoder returns `ModelOutput`
        model_input_name = model_input_name if model_input_name is not None else self.main_input_name
        encoder_kwargs["return_dict"] = True
        encoder_kwargs[model_input_name] = inputs_tensor
        model_kwargs["encoder_outputs"]: ModelOutput = encoder(**encoder_kwargs)
        model_kwargs["src_input_ids"] = inputs_tensor[:,1:]
        
        p = 2
        ngram = []
        for src in model_kwargs["src_input_ids"]:
            tokens = src[src!=0]
            ngram_dict = collections.defaultdict(set)
            for q in range(0, len(tokens)-p+1):
                ngram_ids = tokens[q:q+p].tolist()
                ngram_dict[ngram_ids[0]].add(ngram_ids[1])
            
            ngram.append(ngram_dict)
        model_kwargs["ngram"] = ngram
        model_kwargs["state"] = torch.LongTensor([0]*len(inputs_tensor)).to(device=inputs_tensor.device)

        return model_kwargs
    
    def _expand_inputs_for_generation(
        self,
        input_ids: torch.LongTensor,
        expand_size: int = 1,
        is_encoder_decoder: bool = False,
        attention_mask: Optional[torch.LongTensor] = None,
        encoder_outputs: Optional[ModelOutput] = None,
        **model_kwargs,
    ) -> Tuple[torch.LongTensor, Dict[str, Any]]:
        expanded_return_idx = (
            torch.arange(input_ids.shape[0]).view(-1, 1).repeat(1, expand_size).view(-1).to(input_ids.device)
        )
        input_ids = input_ids.index_select(0, expanded_return_idx)

        if "token_type_ids" in model_kwargs:
            token_type_ids = model_kwargs["token_type_ids"]
            model_kwargs["token_type_ids"] = token_type_ids.index_select(0, expanded_return_idx)

        if attention_mask is not None:
            model_kwargs["attention_mask"] = attention_mask.index_select(0, expanded_return_idx)

        if "src_input_ids" in model_kwargs:
            src_input_ids = model_kwargs["src_input_ids"]
            model_kwargs["src_input_ids"] = src_input_ids.index_select(0, expanded_return_idx)
            
        if "state" in model_kwargs:
            state = model_kwargs["state"]
            model_kwargs["state"] = state.index_select(0, expanded_return_idx)
            # model_kwargs["state"] = torch.LongTensor([0]*len(input_ids)).to(device=input_ids.device)
        
        if "ngram" in model_kwargs:
            ngram = [[ite] * expand_size for ite in model_kwargs["ngram"]]
            model_kwargs["ngram"] = sum(ngram,[])
        
        if is_encoder_decoder:
            if encoder_outputs is None:
                raise ValueError("If `is_encoder_decoder` is True, make sure that `encoder_outputs` is defined.")
            encoder_outputs["last_hidden_state"] = encoder_outputs.last_hidden_state.index_select(
                0, expanded_return_idx.to(encoder_outputs.last_hidden_state.device)
            )
            model_kwargs["encoder_outputs"] = encoder_outputs
        return input_ids, model_kwargs
    
    @staticmethod
    def _update_model_kwargs_for_generation(
        outputs: ModelOutput, model_kwargs: Dict[str, Any], is_encoder_decoder: bool = False
    ) -> Dict[str, Any]:
        # update past
        if "past_key_values" in outputs:
            model_kwargs["past"] = outputs.past_key_values
        elif "mems" in outputs:
            model_kwargs["past"] = outputs.mems
        elif "past_buckets_states" in outputs:
            model_kwargs["past"] = outputs.past_buckets_states
        else:
            model_kwargs["past"] = None

        # update token_type_ids with last value
        if "token_type_ids" in model_kwargs:
            token_type_ids = model_kwargs["token_type_ids"]
            model_kwargs["token_type_ids"] = torch.cat([token_type_ids, token_type_ids[:, -1].unsqueeze(-1)], dim=-1)

        # update attention mask
        if not is_encoder_decoder:
            if "attention_mask" in model_kwargs:
                attention_mask = model_kwargs["attention_mask"]
                model_kwargs["attention_mask"] = torch.cat(
                    [attention_mask, attention_mask.new_ones((attention_mask.shape[0], 1))], dim=-1
                )

        # update state
        if "state" in outputs:
            model_kwargs['state'] = outputs['state']
        
        return model_kwargs
    
    def prepare_inputs_for_generation(
        self,
        decoder_input_ids,
        past=None,
        attention_mask=None,
        head_mask=None,
        decoder_head_mask=None,
        cross_attn_head_mask=None,
        use_cache=None,
        encoder_outputs=None,
        **kwargs
    ):
        # cut decoder_input_ids if past is used
        if past is not None:
            decoder_input_ids = decoder_input_ids[:, -1:]
            
        return {
            "input_ids": None,  # encoder_outputs is defined. input_ids not needed
            "src_input_ids": kwargs["src_input_ids"],
            "ngram":kwargs["ngram"],
            "state":kwargs["state"],
            "encoder_outputs": encoder_outputs,
            "past_key_values": past,
            "decoder_input_ids": decoder_input_ids,
            "attention_mask": attention_mask,
            "head_mask": head_mask,
            "decoder_head_mask": decoder_head_mask,
            "cross_attn_head_mask": cross_attn_head_mask,
            "use_cache": use_cache,  # change this to avoid caching (presumably for debugging)
        }
    
    def forward(
        self,
        input_ids: torch.LongTensor = None,
        src_input_ids: Optional[torch.LongTensor] = None,
        ngram = None,
        state = None,
        src_labels: torch.LongTensor = None,
        tgt_labels: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        decoder_input_ids: Optional[torch.LongTensor] = None,
        decoder_attention_mask: Optional[torch.LongTensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        decoder_head_mask: Optional[torch.Tensor] = None,
        cross_attn_head_mask: Optional[torch.Tensor] = None,
        encoder_outputs: Optional[List[torch.FloatTensor]] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        decoder_inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, Seq2SeqLMOutputWithBio]:
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if labels is not None:
            if use_cache:
                logger.warning("The `use_cache` argument is changed to `False` since `labels` is provided.")
            use_cache = False
            if decoder_input_ids is None and decoder_inputs_embeds is None:
                decoder_input_ids = shift_tokens_right(
                    labels, self.config.pad_token_id, self.config.decoder_start_token_id
                )
        outputs = self.model(
            input_ids,
            attention_mask=attention_mask,
            decoder_input_ids=decoder_input_ids,
            encoder_outputs=encoder_outputs,
            decoder_attention_mask=decoder_attention_mask,
            head_mask=head_mask,
            decoder_head_mask=decoder_head_mask,
            cross_attn_head_mask=cross_attn_head_mask,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            decoder_inputs_embeds=decoder_inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        
        lm_logits = self.lm_head(outputs[0]) + self.final_logits_bias
        
        enc_bio_loss = None
        src_acc = None
        if src_labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            enc_last_state = outputs.encoder_last_hidden_state[:,:-1]
            enc_bio_logits = self.enc_bio_fc(enc_last_state)
            enc_bio_loss = loss_fct(enc_bio_logits.view(-1, 3), src_labels.view(-1))
            preds = torch.argmax(enc_bio_logits, dim=-1)
            src_acc = (preds == src_labels).sum()/(src_labels!=-100).sum()
            
        dec_bio_loss = None
        tgt_acc = None
        next_state = None
        if tgt_labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            dec_last_state = outputs.last_hidden_state
            dec_bio_logits = self.enc_bio_fc(dec_last_state)
            dec_bio_loss = loss_fct(dec_bio_logits.view(-1, 3), tgt_labels.view(-1)) # 错开一位
            preds = torch.argmax(dec_bio_logits, dim=-1)
            tgt_acc = (preds == tgt_labels).sum()/(tgt_labels!=-100).sum()
        else:# 推理阶段
            # 对 输出token分布进行mask--> logits mask
            dec_last_state = outputs.last_hidden_state[:,-1] # 当前隐藏向量 bs x d_model
            probs = nn.functional.softmax(self.enc_bio_fc(dec_last_state), -1) # bs x 3
            
            # 0，1，2 即 0 后面不能接2，2后面不能接1
            tag = torch.nonzero(state==0).squeeze(-1)
            if len(tag)>0:
                probs[tag, 2] *= -1
            tag = torch.nonzero(state==2).squeeze(-1)
            if len(tag)>0:
                probs[tag, 1] *= -1
            
            preds = torch.argmax(probs, dim=-1)
            
            next_state = preds
            new_lm_logits = torch.zeros_like(lm_logits) - 1000
            #0 不做处理
            select_sample_ids = torch.nonzero(preds==0).squeeze(-1)
            new_lm_logits[select_sample_ids,:,:] = lm_logits[select_sample_ids,:,:]
            #1 mask 不在 src 中的所有 tokens
            select_sample_ids = torch.nonzero(preds==1).squeeze(-1)
            if len(select_sample_ids)>0:
                for i in select_sample_ids:
                    src_inp = src_input_ids[i]
                    src_inp = src_inp[src_inp!=0]
                    new_lm_logits[i,-1,src_inp] = lm_logits[i,-1,src_inp]

            #2 mask 与前一个词不能组成ngram的所有token
            select_sample_ids = torch.nonzero(preds==2).squeeze(-1)
            if len(select_sample_ids)>0:
                for i in select_sample_ids:
                    select_ids = ngram[i][decoder_input_ids[i,-1].item()]
                    if len(select_ids)==0:continue
                    # select_ids = list(select_ids)
                    new_lm_logits[i,-1,list(select_ids)] = lm_logits[i,-1,list(select_ids)]
                    
                    # TODO add sparsemax
            lm_logits = new_lm_logits
        masked_lm_loss = None
        if labels is not None:
            # loss_fct = nn.CrossEntropyLoss()
            loss_fct = Sparsemax(k_sparse=10)
            masked_lm_loss = loss_fct(lm_logits.view(-1, self.config.vocab_size), labels.view(-1))

        if not return_dict:
            output = (lm_logits,) + outputs[1:]
            return ((masked_lm_loss,) + output) if masked_lm_loss is not None else output

        return Seq2SeqLMOutputWithBio(
            enc_bio_loss=enc_bio_loss,
            dec_bio_loss=dec_bio_loss,
            src_acc = src_acc,
            tgt_acc = tgt_acc,
            loss=masked_lm_loss,
            logits=lm_logits,
            state=next_state,
            past_key_values=outputs.past_key_values,
            decoder_hidden_states=outputs.decoder_hidden_states,
            decoder_attentions=outputs.decoder_attentions,
            cross_attentions=outputs.cross_attentions,
            encoder_last_hidden_state=outputs.encoder_last_hidden_state,
            encoder_hidden_states=outputs.encoder_hidden_states,
            encoder_attentions=outputs.encoder_attentions,
        )
        
        