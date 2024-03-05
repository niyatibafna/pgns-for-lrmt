debug = False
debug_pgen = False
debug_vis = False
SEED = 42

import warnings 
from typing import Optional, Tuple, Union, List
import torch
from torch import nn
import torch.nn.functional as F
from torch.nn import Embedding, NLLLoss, CrossEntropyLoss
# from numpy import ndarray

from transformers import AutoTokenizer, EncoderDecoderModel, EncoderDecoderConfig, PretrainedConfig, \
    PreTrainedModel, BertTokenizer, BertConfig, BertForMaskedLM, DataCollatorForSeq2Seq, Seq2SeqTrainer, Seq2SeqTrainingArguments
from transformers import AutoTokenizer, EncoderDecoderModel, PretrainedConfig, EncoderDecoderConfig, \
    PreTrainedModel, BertTokenizer, BertModel, BertConfig, BertForMaskedLM, DataCollatorForSeq2Seq, Seq2SeqTrainer, Seq2SeqTrainingArguments
from transformers.modeling_outputs import Seq2SeqLMOutput, BaseModelOutput
from transformers.models.bert.modeling_bert import BertEmbeddings, BertModel, \
    BertEncoder, BertPooler, BertForSequenceClassification,BertPreTrainedModel, \
    BaseModelOutputWithPoolingAndCrossAttentions
from transformers.models.encoder_decoder.modeling_encoder_decoder import shift_tokens_right, DEPRECATION_WARNING
# from transformers.generation_utils import LogitsProcessorList, StoppingCriteriaList

from torch.utils.tensorboard import SummaryWriter

from datasets import Dataset, load_dataset
from math import inf as INF
import numpy as np

from datasets import load_metric
from transformers import EvalPrediction
import evaluate


import argparse
import os, sys

from transformers.trainer_callback import TrainerControl, TrainerState
from transformers.training_args import TrainingArguments
sys.path.append("../")
# from utils.variables import LANGS
from utils import get_tokenizer
import logging
from dataclasses import dataclass
# LANGS = sorted(LANGS)
# print(LANGS)

from transformers.deepspeed import is_deepspeed_zero3_enabled
from transformers.integrations import TensorBoardCallback, is_tensorboard_available

import seaborn as sns
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties

from memory_profiler import profile

SEED = 42

#%%
@dataclass
class Seq2SeqLMOutputWithPGen(Seq2SeqLMOutput):
    '''Same as Seq2SeqLMOutput, but with p_gen'''
    p_gen: torch.FloatTensor = None


class EncoderDecoderModelPGN(EncoderDecoderModel):
    '''This class modifies EncoderDecoderModel to incorporate a Pointer-Generator Network (PGN)'''


    def __init__(
        self,
        config: Optional[PretrainedConfig] = None,
        encoder: Optional[PreTrainedModel] = None,
        decoder: Optional[PreTrainedModel] = None,
    ):
        self.log = {}
        self.increment = 0
        super().__init__(config, encoder, decoder)
        self.hidden_size = self.encoder.config.hidden_size
        self.generate_prob = nn.Linear(self.hidden_size * 3, 1)


    
    def print_grads(self):
        for param in self.model_lid.parameters():
            print(param.requires_grad)

    ##@profile
    def forward(
    self,
    input_ids: Optional[torch.LongTensor] = None,
    attention_mask: Optional[torch.FloatTensor] = None,
    decoder_input_ids: Optional[torch.LongTensor] = None,
    decoder_attention_mask: Optional[torch.BoolTensor] = None,
    encoder_outputs: Optional[Tuple[torch.FloatTensor]] = None,
    past_key_values: Tuple[Tuple[torch.FloatTensor]] = None,
    inputs_embeds: Optional[torch.FloatTensor] = None,
    decoder_inputs_embeds: Optional[torch.FloatTensor] = None,
    labels: Optional[torch.LongTensor] = None,
    use_cache: Optional[bool] = None,
    output_attentions: Optional[bool] = None,
    output_hidden_states: Optional[bool] = None,
    return_dict: Optional[bool] = None,
    encoder_input_ids: Optional[torch.LongTensor] = None,
    **kwargs,
) -> Union[Tuple, Seq2SeqLMOutput]:
        '''Returns:

        Examples:

        ```python
        >>> from transformers import EncoderDecoderModel, BertTokenizer
        >>> import torch

        >>> tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
        >>> model = EncoderDecoderModel.from_encoder_decoder_pretrained(
        ...     "bert-base-uncased", "bert-base-uncased"
        ... )  # initialize Bert2Bert from pre-trained checkpoints

        >>> # training
        >>> model.config.decoder_start_token_id = tokenizer.cls_token_id
        >>> model.config.pad_token_id = tokenizer.pad_token_id
        >>> model.config.vocab_size = model.config.decoder.vocab_size

        >>> input_ids = tokenizer("This is a really long text", return_tensors="pt").input_ids
        >>> labels = tokenizer("This is the corresponding summary", return_tensors="pt").input_ids
        >>> outputs = model(input_ids=input_ids, labels=labels)
        >>> loss, logits = outputs.loss, outputs.logits

        >>> # save and load from pretrained
        >>> model.save_pretrained("bert2bert")
        >>> model = EncoderDecoderModel.from_pretrained("bert2bert")

        >>> # generation
        >>> generated = model.generate(input_ids)
        '''

        global increment_for_tb, tb_writer
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        kwargs_encoder = {argument: value for argument, value in kwargs.items() if not argument.startswith("decoder_")}

        kwargs_decoder = {
            argument[len("decoder_") :]: value for argument, value in kwargs.items() if argument.startswith("decoder_")
        }

        if debug:
            if input_ids is not None:
                print(f"input_ids:{input_ids.shape}")
            else:
                print("input_ids: None")
            if inputs_embeds is not None:
                print(f"inputs_embeds:{inputs_embeds.shape}")
            else:
                print("inputs_embds:: None")

        if encoder_outputs is None:
            encoder_outputs = self.encoder(
                input_ids=input_ids,
                attention_mask=attention_mask,
                inputs_embeds=inputs_embeds,
                output_attentions = output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
                **kwargs_encoder,
            )
        elif isinstance(encoder_outputs, tuple):
            encoder_outputs = BaseModelOutput(*encoder_outputs)

        # Last hidden state of the encoder is the context
        encoder_hidden_states = encoder_outputs[0]

        # optionally project encoder_hidden_states
        if (
            self.encoder.config.hidden_size != self.decoder.config.hidden_size
            and self.decoder.config.cross_attention_hidden_size is None
        ):
            encoder_hidden_states = self.enc_to_dec_proj(encoder_hidden_states)

        if (labels is not None) and (decoder_input_ids is None and decoder_inputs_embeds is None):
            decoder_input_ids = shift_tokens_right(
                labels, self.config.pad_token_id, self.config.decoder_start_token_id
            )
            if decoder_attention_mask is None:
                decoder_attention_mask = decoder_input_ids.new_tensor(decoder_input_ids != self.config.pad_token_id)

        
        # Set output_attentions to True because we need cross attention weights.
        decoder_outputs = self.decoder(
            input_ids=decoder_input_ids,
            attention_mask=decoder_attention_mask,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=attention_mask,
            inputs_embeds=decoder_inputs_embeds,
            output_attentions=True,
            output_hidden_states=True,
            use_cache=use_cache,
            past_key_values=past_key_values,
            return_dict=True,
            **kwargs_decoder,
        )
        # S1: length of input sequence, S2: length of output sequence
        # decoder_inputs_embeds, decoder_last_hidden_state: (B, S2, H)
        decoder_inputs_embeds, decoder_last_hidden_state = decoder_outputs.hidden_states[0], decoder_outputs.hidden_states[-1]
        # cross_attention_weights (multihead): (B, H, S2, S1)
        cross_attention_weights = decoder_outputs.cross_attentions[-1]

        if debug:
            print(f"decoder_hidden_states: {len(decoder_outputs.hidden_states)}")
            print(f"embeddings ouputs: {decoder_outputs.hidden_states[0].shape}")
            print(f"first layer outputs: {decoder_outputs.hidden_states[1].shape}")

            # We want to see whether decoder_inputs_embeds contains anything
            # print(f"decoder_inputs_embeds: {decoder_inputs_embeds.shape}")
            print(f"encoder_hidden_states shape: {encoder_hidden_states.shape}")
            # print(f"Encoder attention shape: {encoder_outputs.attentions[-1].shape}")
            print(f"Decoder input_ids: {decoder_input_ids.shape}")
            print(f"Decoder outputs: {decoder_outputs[0].shape}")
            print(f"Decoder outputs: {type(decoder_outputs)}")
            print(f"Cross attention weights shape: {cross_attention_weights.shape}")
            # print(f"Decoder attention shape: {decoder_outputs.attentions[-1].shape}")

        

        # Calculating p_gen

        
        # (General case) We have to calculate p_gen
        # First, we collapse multihead attention into a single head
        ## (B, H, S2, S1) -> (B, S2, S1)
        attentions = cross_attention_weights.mean(dim=1)
        # Then, we find the context vector by taking a weighted sum of the encoder hidden states
        ## (B, S2, S1) x (B, S1, H) -> (B, S2, H)
        context_vector = torch.bmm(attentions, encoder_hidden_states)
        # Then, we concatenate the context vector with the decoder hidden states and the decoder input
        ## (B, S2, H) + (B, S2, H) + (B, S2, H) -> (B, S2, 3H)
        
        concat_vector = torch.cat([context_vector, decoder_last_hidden_state, decoder_inputs_embeds], dim = -1)
        # Then, we pass this through a linear layer to get the probability of generating the next word
        ## (B, S2, 3H) -> (B, S2, 1)
        ## We use sigmoid to get a probability
        p_gen = torch.sigmoid(self.generate_prob(concat_vector))

        if self.force_p_gen is not None:
            # We're forcing p_gen, initialize it directly
            p_gen = torch.ones(cross_attention_weights.shape[0], cross_attention_weights.shape[2], 1).to(cross_attention_weights.device) * self.force_p_gen
        
        
        # Note that copy_prob = 1 - generate_prob

        # Mixing probabilities from generate and copy mechanism
        ## Finding the logits from the copy mechanism
        ## For this, we just use the cross attention weights 
        ## Shape: (B, S2, S1)
        ## Then, we have to redistribute these over the vocabulary, so that we can mix them
        ## with the generate probabilities
        ## required_shape: (B, S2, V)
        ## attentions: (B, S2, S1) 
        ## input_ids: (B, S1) --> (B, S2, S1) (we just repeat in the second dimension)
        required_shape = (attentions.shape[0], attentions.shape[1], self.decoder.config.vocab_size)

        # The following condition will only be true during evaluation, when we use generate() with greedy_search()
        # Both of those functions have been modified to pass the input_ids to the encoder to the model forward function (this one)
        # as 'encoder_input_ids'
        if encoder_input_ids is not None and input_ids is None:
            input_ids = encoder_input_ids

        copy_logits = torch.zeros(required_shape, dtype=attentions.dtype).to(self.device).scatter(dim = -1, index = input_ids.unsqueeze(1).repeat(1, attentions.shape[1], 1).to(self.device), \
            src = attentions.to(self.device))

        if debug:
            # Print encoder outputs
            if encoder_outputs is not None:
                print(f"Encoder outputs: {encoder_outputs}")
            if encoder_input_ids is not None:
                print(f"Encoder input ids: {encoder_input_ids.shape}")
            # print(f"input_ids in kwargs in forward: {kwargs['encoder_input_ids']}")

            print(f"generate_prob shape: {p_gen.shape}")
            print(f"copy_logits shape: {copy_logits.shape}")
    
        ## Now, we mix the generate and copy probabilities
        generate_logits = decoder_outputs.logits if return_dict else decoder_outputs[0]
        # Normalizing and mixing probabilities
        mixed_probs = F.softmax(generate_logits, dim=-1) * p_gen + F.softmax(copy_logits, dim=-1) * (1 - p_gen)

        ### TODO Change this - sanity check to see what happens if we switch the generate and copy logits
        ### Over here: p_gen is conceptually p_copy
        # mixed_probs = F.softmax(generate_logits, dim=-1) * (1 - p_gen) + F.softmax(copy_logits, dim=-1) * p_gen 

        ## This is equivalent to doing:
        ## generate_probs = torch.softmax(generate_logits, dim = -1)
        ## copy_probs = torch.softmax(copy_logits, dim = -1)
        ## mixed_probs = generate_probs * p_gen + copy_probs * (1 - p_gen)

        # Compute loss independent from decoder (as some shift the logits inside them)
        # Pass log probabilities to NLLLoss
        loss = None
        if labels is not None:
            warnings.warn(DEPRECATION_WARNING, FutureWarning)
            loss_fct = NLLLoss()
            loss = loss_fct(torch.log(mixed_probs).reshape(-1, self.decoder.config.vocab_size), labels.view(-1))


        model_outputs = Seq2SeqLMOutputWithPGen(
            loss=loss,
            logits=torch.log(mixed_probs), # We pass log probabilities since these will become probabilities after softmax
            past_key_values=decoder_outputs.past_key_values,
            decoder_hidden_states=decoder_outputs.hidden_states,
            decoder_attentions=decoder_outputs.attentions,
            cross_attentions=decoder_outputs.cross_attentions,
            encoder_last_hidden_state=encoder_outputs.last_hidden_state,
            encoder_hidden_states=encoder_outputs.hidden_states,
            encoder_attentions=encoder_outputs.attentions,
            p_gen = p_gen,
        )
        
        if self.training: 
            # Once in a while, log visualizations.
            ## Do this every 100 batches
            if self.increment % 100 == 0:
                self.log["params"] = (input_ids, decoder_input_ids, \
                                    torch.clone(model_outputs.cross_attentions[-1].unsqueeze(0)), torch.clone(model_outputs.p_gen))
            self.increment += 1

        if not return_dict:
            if loss is not None:
                return (loss,) + decoder_outputs + encoder_outputs
            else:
                return decoder_outputs + encoder_outputs

        return model_outputs

    ##@profile    
    def generate(
        self,
        inputs: Optional[torch.Tensor] = None,
        **kwargs,
    ):
        if debug:
            print("In derived class for generate!")
            print("Inputs: ", inputs)
            print("kwargs: ", kwargs)
        return super().generate(inputs, **kwargs)



def init_tokenizer(TOKENIZER_INPATH, FILES = None, vocab_size = 16000):
    '''Note that if we are using a pretrained tokenizer,
    we should simply pass the HF key of the model as TOKENIZER_INPATH'''

    logging.info("Loading src tokenizer from {}".format(TOKENIZER_INPATH))
    tokenizer = get_tokenizer.train_or_load_tokenizer(TOKENIZER_INPATH,  \
        FILES = FILES, vocab_size = vocab_size)
    # Looks like HF MT doesn't support separate source and target tokenizers
    # logging.info("Loading tgt tokenizer from {}".format(TGT_TOKENIZER_INPATH))
    # tgt_tokenizer = get_tokenizer.train_or_load_tokenizer(TGT_TOKENIZER_INPATH, tokenizer)

    ### Optionally add language ID tokens
    # tokenizer = get_tokenizer.add_langid_tokens(tokenizer, LANGS)    
    return tokenizer

def init_models(ENC_DEC_MODELPATH, tokenizer, PT_CKPT = None, force_p_gen = None):
    '''Get seq2seq model'''

    # Initialize Seq2Seq model, input and output tokenizer, special hyperparameters
    if PT_CKPT:
        # First we check if there is some enc-dec checkpoint (along with cross-attention weights)
        logging.info("Loading encoder-decoder model from {}".format(PT_CKPT))
        model_enc_dec = EncoderDecoderModelPGN.from_pretrained(PT_CKPT)
    elif ENC_DEC_MODELPATH:
        # If not, we check if the encoder and decoder can be initalized separately from some model
        logging.info("Loading encoder-decoder model from {}".format(ENC_DEC_MODELPATH))
        model_enc_dec = EncoderDecoderModelPGN.from_encoder_decoder_pretrained(ENC_DEC_MODELPATH, ENC_DEC_MODELPATH)
    else:
        # If not, we initialize the encoder and decoder from scratch
        logging.info("Initializing encoder-decoder model from scratch")
        encoder_config = BertConfig(vocab_size=len(tokenizer), num_hidden_layers=6, num_attention_heads=4, hidden_size=512, intermediate_size=1024)
        decoder_config = BertConfig(vocab_size=len(tokenizer), num_hidden_layers=6, num_attention_heads=4, hidden_size=512, intermediate_size=1024)
        config = EncoderDecoderConfig.from_encoder_decoder_configs(encoder_config, decoder_config)
        model_enc_dec = EncoderDecoderModelPGN(config)

    ## Set model parameters
    # model_enc_dec.model_lid = model_lid
    # model_enc_dec.alpha = alpha
    # model_enc_dec.tau = tau
    # model_enc_dec.istauhard = istauhard
        
    model_enc_dec.force_p_gen = force_p_gen

    if model_enc_dec.encoder.config.vocab_size !=len(tokenizer):
        model_enc_dec.encoder.resize_token_embeddings(len(tokenizer))
    if model_enc_dec.decoder.config.vocab_size != len(tokenizer):
        model_enc_dec.decoder.resize_token_embeddings(len(tokenizer))

    # print("In init: model lid print info 1")
    # model_lid.print_info()
    # print("In init: model lid print info 2")
    # model_enc_dec.model_lid.print_info()
    
    model_enc_dec.config.decoder_start_token_id = tokenizer.cls_token_id
    model_enc_dec.config.pad_token_id = tokenizer.pad_token_id

    return model_enc_dec

def get_dataset_split(DATAFILE_L1, DATAFILE_L2, max_lines, tokenizer, max_length = 512):

    def preprocess_function(examples):
        # HF doesn't support separate source and target tokenizers, 
        # so we assume it's the same tokenizer for now.
        inputs = [ex for ex in examples["source"]]
        targets = [ex for ex in examples["target"]]
        model_inputs = tokenizer(
            inputs, text_target=targets, max_length=max_length, truncation=True
        )
        return model_inputs
    
    logging.info("Loading Datasets...")
    dataset = load_dataset("text", data_files={"source": [DATAFILE_L1], \
        "target": [DATAFILE_L2]})
    
    if debug:
        print(dataset)
        # print(dataset["source"])
        # print(dataset["target"])
        print("Example:")
        print(dataset["source"][0])
        print(dataset["target"][0])
    
    # Create a new dataset that has source and target as columns
    dataset = Dataset.from_dict({"source": dataset["source"]["text"], "target": dataset["target"]["text"]})
    
    dataset = dataset.select(range(min(max_lines, len(dataset))))

    if debug:
        print("Examples:")
        print(dataset[0])
    
    logging.info("STARTING TOKENIZING")
    dataset = dataset.map(preprocess_function, batched = True, \
                                        remove_columns=["source", "target"])
    
    logging.info("DONE TOKENIZING!")
    dataset = dataset.with_format("torch")
    return dataset

def get_mt_dataset(DATADIR_L1, DATADIR_L2, max_lines, tokenizer, max_length = 512):

    '''
    This function assumes that the datadir contains train, dev, and test splits, 
    called "train", "dev", and "test"
    '''
    # splits = {"train", "test", "dev"}

    def get_split(split):
        DATAFILE_L1 = os.path.join(DATADIR_L1, split)
        DATAFILE_L2 = os.path.join(DATADIR_L2, split)
        return get_dataset_split(DATAFILE_L1, DATAFILE_L2, max_lines, tokenizer, max_length = max_length)

    train_dataset = get_split("train")
    dev_dataset = get_split("dev")
    test_dataset = get_split("test")

    ## Split dataset into train, dev, and test if no splits are provided
    # dataset = dataset.train_test_split(test_size=0.2, seed=SEED)
    # train_dataset = dataset["train"]
    # devtest_dataset = dataset["test"]
    # devtest_dataset = devtest_dataset.train_test_split(test_size=0.5, seed=SEED)
    # dev_dataset = devtest_dataset["train"]
    # test_dataset = devtest_dataset["test"]

    # Log all sizes
    # logging.info("Length of dataset: {}".format(len(dataset)))
    logging.info("Length of train dataset: {}".format(len(train_dataset)))
    logging.info("Length of dev dataset: {}".format(len(dev_dataset)))
    logging.info("Length of test dataset: {}".format(len(test_dataset)))
    
    return train_dataset, dev_dataset, test_dataset

def compute_metrics(pred):
    '''Compute BLEU score'''
    global tokenizer, bleu

    # Get predictions
    predictions = pred.predictions
    labels = pred.label_ids
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)

    # Decode
    predictions = tokenizer.batch_decode(predictions, skip_special_tokens=True, clean_up_tokenization_spaces = True)
    labels = tokenizer.batch_decode(labels, skip_special_tokens=True, clean_up_tokenization_spaces = True)

    # Compute BLEU
    ## If length of predictions is 0, we return 0
    if sum([len(pred.split()) for pred in predictions]) == 0:
        return {'bleu': 0.0, \
                'brevity_penalty': 0, \
                'length_ratio': 0, 'translation_length': 0, 'reference_length': 0}
    bleu_metric = bleu.compute(predictions = predictions, references = labels)

    # Remove precisions
    bleu_metric = {k: v for k, v in bleu_metric.items() if k != "precisions"}

    return bleu_metric

#@profile
def visualization_of_cross_attentions_and_pgen(input_ids, decoder_input_ids, cross_attentions, p_gen, global_step):
    global tokenizer, increment_for_tb, tb_writer, script

    if debug:
        print(f"Shape of cross attention weights: {cross_attentions[0].shape}")
    # If we have a batch, we take the first element
    if input_ids.shape[0] > 1:
        cross_attention_weights = cross_attentions[-1][0].mean(dim=0)
    else:
        cross_attention_weights = cross_attentions[-1].squeeze(0).mean(dim=0)
    
    cross_attention_weights = cross_attention_weights.detach().cpu().numpy()
    # Transpose cross attention weights
    cross_attention_weights = cross_attention_weights.T
    
    input_ids = input_ids[0]
    decoder_input_ids = decoder_input_ids[0]
    input_tokens = tokenizer.convert_ids_to_tokens(input_ids)
    output_tokens = tokenizer.convert_ids_to_tokens(decoder_input_ids)
    # Get p_gen
    p_gen = p_gen[0]
    p_gen = p_gen.squeeze(1).detach().cpu().numpy()
    p_copy = 1 - p_gen
    
    if debug_vis:
        logging.info(f"Shape of p_copy: {p_copy.shape}")
        logging.info(f"Shape of input: {input_ids.shape}")
        logging.info(f"Shape of decoder input: {decoder_input_ids.shape}")
        logging.info(f"Shape of cross attention weights: {cross_attention_weights.shape}")

    # The decoder ids have been shifted right in decoder, le'ts shift them back
    output_tokens = output_tokens[1:]  # Now we're missing the last token, but it's a PAD token so we don't care

    # Get tokens to label x and y axes    
    
    # Remove PAD tokens from input and output tokens
    input_tokens = [token for token in input_tokens if token != tokenizer.pad_token]
    output_tokens = [token for token in output_tokens if token != tokenizer.pad_token]
    # Remove PAD tokens from cross attention weights
    
    cross_attention_weights = cross_attention_weights[:len(input_tokens), :len(output_tokens)]
    # Remove PAD tokens from p_gen
    
    p_copy = p_copy[:len(output_tokens)]

    # Don't want to show the CLS and SEP tokens
    input_tokens = input_tokens[1:-1]
    output_tokens = output_tokens[1:-1]
    cross_attention_weights = cross_attention_weights[1:-1, 1:-1]
    p_copy = p_copy[1:-1] 

    if debug_vis:
        logging.info(f"Number of input tokens: {len(input_tokens)}")
        logging.info(f"Number of output tokens: {len(output_tokens)}")
        logging.info(f"Number of p_copy tokens: {len(p_copy)}")
        logging.info(f"Shape of cross attention weights: {cross_attention_weights.shape}")

    # Log cross attention weights as a heatmap
    ## Here, we decide whether to use Devanagari (hi-mr) or Latin (for es-ca)
    if script == "dev":
        # All labels in Devanagari font  
        matplotlib.rcParams['font.family'] = 'Noto Serif Devanagari'
        # Reduce font size
    matplotlib.rcParams.update({'font.size': 22})
    # Prepare figure
    fig, ax = plt.subplots(2, 1, figsize=(40, 20))
    # Set up heatmap as first subplot
    sns.heatmap(cross_attention_weights, ax = ax[0], cbar=False)
    # Set up labels
    ax[0].set_xticks(ticks=range(len(output_tokens)), labels=output_tokens, rotation=60)
    ax[0].set_yticks(ticks=range(len(input_tokens)), labels=input_tokens, rotation=0)

    # Plot p_gen as a bar graph
    if debug_vis:
        logging.info(f"Shape of new p_copy going into bar chart: {p_copy.shape}")
        logging.info(f"length of labels: {len(output_tokens)}")
        logging.info(output_tokens)
        logging.info(f"p_copy: {p_copy}")
    ax[1].bar(range(len(output_tokens)), p_copy)
    # Label x axis as output tokens
    ax[1].set_xticks(ticks=range(len(output_tokens)), labels=output_tokens, rotation=60)

    # Label y axis as "p_copy"
    ax[1].set_ylabel("p_copy")


    # Save figure to file
    fig.savefig(f"{args.LOG_DIR}/vis/cross_attention_weights,p_copy_histogram_{global_step}.png")
    

class CustomTensorboardCallback(TensorBoardCallback):
    
    def on_log(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        '''We visualize things here'''
        model = kwargs.pop("model")
        (input_ids, decoder_input_ids, \
            cross_attns, p_gens) = model.log["params"]

        # print(f"In custom callback, something to visualize: {something_to_visualize}")
        visualization_of_cross_attentions_and_pgen(input_ids, decoder_input_ids, cross_attns, p_gens, state.global_step)
        
        # Log avg pgen to tensorboard
        mean_pgen = p_gens.mean()
        self.tb_writer.add_scalar("p_gen across training steps", mean_pgen, state.global_step)

#@profile
def main(args):
# TRAIN_FILES, DEV_FILES, 
# LID_MODELPATH, NUM_LID_LABELS, ENC_DEC_MODELPATH, alpha = 0.5, max_length = 512
# OUTPUT_DIR, LOG_DIR, epochs, batch_size

    global tokenizer
    global bleu

    global script
    if (args.ENC_DEC_MODELPATH and "spanish" in args.ENC_DEC_MODELPATH) or ("es" in args.DATADIR_L1):
        script = "lat"
    elif (args.ENC_DEC_MODELPATH and "hindi" in args.ENC_DEC_MODELPATH) or ("hi" in args.DATADIR_L1):
        script = "dev"
    else:
        script = "lat"

    # Get seq2seq model and tokenizer
    logging.info("Initializing tokenizer...")
    FILES = [os.path.join(args.DATADIR_L1, "train"),\
             os.path.join(args.DATADIR_L2, "train"),\
            os.path.join(args.DATADIR_L1, "dev"),\
            os.path.join(args.DATADIR_L2, "dev"),\
            os.path.join(args.DATADIR_L1, "test"),\
            os.path.join(args.DATADIR_L2, "test")]
    tokenizer = init_tokenizer(args.TOKENIZER_INPATH, \
                               FILES, args.vocab_size)

    logging.info("Initializing models...")
    model_enc_dec = init_models(args.ENC_DEC_MODELPATH, tokenizer, args.PT_CKPT)
    
    logging.info("Getting datasets...")
    # Get dataset splits, and preprocess them
    train_dataset, dev_dataset, test_dataset = \
    get_mt_dataset(args.DATADIR_L1, args.DATADIR_L2, max_lines=args.max_lines, tokenizer= tokenizer, max_length= args.max_length)
    
    # Instead of that, download some MT dataset from HF
    # tokenizer = AutoTokenizer.from_pretrained(args.ENC_DEC_MODELPATH)
    # train_dataset = load_dataset("wmt16", "de-en", split="train[:1%]").select(range(args.max_lines))
    # dev_dataset = load_dataset("wmt16", "de-en", split="test[:1%]").select(range(int(args.max_lines/9))) # This gives us a 9:1 ratio
    # train_dataset, dev_dataset = preprocess_wmt(train_dataset, dev_dataset, tokenizer, max_length = args.max_length)
    # test_dataset = dev_dataset
    # Print example
    logging.info(f"EXAMPLE: {train_dataset[0]} ")   

    data_collator = DataCollatorForSeq2Seq(tokenizer, model=model_enc_dec)
    
    # Metric
    bleu = evaluate.load("bleu")

    # Initialize Seq2SeqTrainer
    logging.info("Initializing trainer...")

    # if args.resume_from_checkpoint: train_steps = 1
    # else: train_steps = len(train_dataset) * args.epochs // args.batch_size

    training_args = Seq2SeqTrainingArguments(
    output_dir=args.OUTPUT_DIR,
    resume_from_checkpoint=args.resume_from_checkpoint,
    overwrite_output_dir=False, #use False for continuing training
    num_train_epochs=args.epochs,
    # max_steps=train_steps,
    per_device_train_batch_size=args.batch_size,
    per_device_eval_batch_size=args.batch_size,
    warmup_steps=500,
    weight_decay=0.01,
    logging_dir=args.LOG_DIR,
    predict_with_generate=True,
    generation_max_length=40, # defaults to model config max_length
    report_to="tensorboard",
    evaluation_strategy="epoch",
    logging_strategy="epoch",
    save_strategy="epoch",
    # eval_steps=2000, 
    # logging_steps=2000,
    # save_steps=2000, # For 15000 examples, this will save roughly every epoch with batch size 8
    load_best_model_at_end=True,
    save_total_limit=2
    )

    trainer = Seq2SeqTrainer(
        model=model_enc_dec,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=dev_dataset,
        tokenizer=tokenizer,
        data_collator= data_collator,
        compute_metrics=compute_metrics,
        callbacks=[CustomTensorboardCallback],
    )   

    
    logging.info("STARTING TRAINING")
    logging.info(f"CUDA: {torch.cuda.is_available()}")
    print(f"Model device: {model_enc_dec.device}")
    trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)

    logging.info("SAVING MODEL")
    model_enc_dec.save_pretrained(args.OUTPUT_DIR)

    # # Get performance and labels on test set
    if test_dataset:
        logging.info("STARTING EVALUATION")
        test_results = trainer.predict(test_dataset)
        test_metrics = test_results.metrics
        predictions = test_results.predictions
        labels = test_results.label_ids
        # Replace -100 with pad token since we can't decode those
        labels = np.where(labels != -100, labels, tokenizer.pad_token_id)

        # Decode into text
        inputs = tokenizer.batch_decode(test_dataset["input_ids"], skip_special_tokens=True)
        predictions = tokenizer.batch_decode(predictions, skip_special_tokens=True)
        labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

        # Log examples
        logging.info("Logging examples...")
        for i in range(len(predictions[:100])):
            logging.info("Example {}: ".format(i))
            logging.info("Input: {}".format(inputs[i]))
            logging.info("Prediction: {}".format(predictions[i]))
            logging.info("Label: {}".format(labels[i]))
        # Log metrics
        logging.info("Logging metrics...")
        logging.info("Test metrics: {}".format(test_metrics))


        logging.info("DONE EVALUATION")

        
        

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--DATADIR_L1", type=str, default=None)
    parser.add_argument("--DATADIR_L2", type=str, default=None)
    parser.add_argument("--ENC_DEC_MODELPATH", type=str, default=None, help="Path to encoder model to initalize encoder/decoder (separately)")
    parser.add_argument("--TOKENIZER_INPATH", type=str, default=None, help="Path to tokenizer - if self-trained, put path. If None, \
                        the tokenizer from the encoder model will be used")
    parser.add_argument("--PT_CKPT", type=str, default=None, help="Path to PGN checkpoint")
    parser.add_argument("--vocab_size", type=int, default = 16_000)
    parser.add_argument("--max_length", type=int, default = 512)
    parser.add_argument("--OUTPUT_DIR", type=str, default="output_dir", help="Path to save model")
    parser.add_argument("--LOG_DIR", type=str, default="logs", help="Path to save tensorboard logs")
    parser.add_argument("--epochs", type=int, default = 20)
    parser.add_argument("--batch_size", type=int, default = 16)
    parser.add_argument("--max_lines", type=int, default = INF)
    parser.add_argument("--resume_from_checkpoint", action="store_true", default=False, help="Resume training from args.OUTPUT_DIR")
    parser.add_argument("--force_p_gen", type=float, default = None)
    # Take any additional approach-related parameters

    args = parser.parse_args()

    logging.basicConfig(filename=f"{args.LOG_DIR}/log.txt", filemode="w", format="%(name)s - %(levelname)s - %(message)s", level=logging.INFO)

    os.makedirs(f"{args.LOG_DIR}/vis/", exist_ok=True)


    main(args)