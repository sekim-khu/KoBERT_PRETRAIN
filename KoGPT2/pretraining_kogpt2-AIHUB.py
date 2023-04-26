# %% [markdown]
# # Import

# %%
import os
import nltk
import json
from time import gmtime, strftime
# from nltk import sent_tokenize
from tqdm import tqdm
from itertools import chain

import multiprocessing

from datasets import load_dataset, load_from_disk, concatenate_datasets
from transformers import AutoTokenizer, AutoConfig, TrainingArguments, AutoModelForCausalLM, DataCollatorForLanguageModeling, Trainer
from accelerate import notebook_launcher

# %% [markdown]
# # Load Dataset

# %%
data_path = "../data"
original_train_datasets_path = data_path + "/original_datasets/TS1"
original_valid_datasets_path = data_path + "/original_datasets/VS1"

current_dir = os.getcwd()
current_time = strftime("%Y-%m-%d-%H:%M:%S", gmtime())
model_output_dir = './gpt2'+"_"+current_time
model_cache_dir = "./gpt2_cache"
raw_datasets_cache_name = ".raw_datasets_cache"
raw_datasets_cache_path = os.path.join(current_dir, raw_datasets_cache_name)

train_sentence_list_file_name = "train_sentence_list.txt"
valid_sentence_list_file_name = "valid_sentence_list.txt"
tokenizer_name = "tokenizer_aihub_news_gpt2"
tokenized_datasets_folder_name = ["gpt2_tokenized_datasets", "gpt2_tokenized_datasets_1", "gpt2_tokenized_datasets_2", "gpt2_tokenized_datasets_3"]
grouped_tokenized_datasets_folder_name = "gpt2_grouped_tokenized_datasets"

old_model_name = "gpt2"

# %%
num_proc = multiprocessing.cpu_count()

# %%
# raw_datasets = load_dataset('text', data_files={"train": os.path.join(data_path, train_sentence_list_file_name), "valid": os.path.join(data_path, valid_sentence_list_file_name)}, cache_dir=raw_datasets_cache_path)

# %% [markdown]
# # Tokenizing

# %% [markdown]
# ### from scratch

# %%
# def get_training_corpus(batch_size=10000):
#     for dataset in [raw_datasets['train'], raw_datasets['valid']]:
#         for start_idx in range(0, len(dataset), batch_size):
#             yield dataset[start_idx : start_idx + batch_size]["text"]
        
# old_tokenizer = AutoTokenizer.from_pretrained(old_model_name)
# tokenizer = old_tokenizer.train_new_from_iterator(get_training_corpus(), 51200)
# tokenizer.save_pretrained(tokenizer_name)

# %%
# raw_datasets_1 = raw_datasets.select(range(20000000))
# raw_datasets_2 = raw_datasets.select(range(20000000, 40000000))
# raw_datasets_3 = raw_datasets.select(range(40000000, len(raw_datasets)))

# %%
# def preprocess_texts(examples):
#     tokenized_inputs = tokenizer(
#        examples["text"], return_special_tokens_mask=True, truncation=True, max_length=512
#     )
#     return tokenized_inputs

# %%
# tokenized_datasets = raw_datasets.map(preprocess_texts, batched=True, remove_columns=["text"], num_proc=num_proc)
# tokenized_datasets.save_to_disk(os.path.join(current_dir, tokenized_datasets_folder_name[0]))

# %%
# tokenized_datasets_1 = raw_datasets_1.map(preprocess_texts, batched=True, remove_columns=["text"], num_proc=num_proc)
# tokenized_datasets_1.save_to_disk(os.path.join(current_dir, tokenized_datasets_folder_name[1]))

# %%
# tokenized_datasets_2 = raw_datasets_2.map(preprocess_texts, batched=True, remove_columns=["text"], num_proc=num_proc)
# tokenized_datasets_2.save_to_disk(os.path.join(current_dir, tokenized_datasets_folder_name[2]))

# %%
# tokenized_datasets_3 = raw_datasets_3.map(preprocess_texts, batched=True, remove_columns=["text"], num_proc=num_proc)
# tokenized_datasets_3.save_to_disk(os.path.join(current_dir, tokenized_datasets_folder_name[3]))

# %%
# tokenized_datasets = concatenate_datasets([tokenized_datasets_1, tokenized_datasets_2, tokenized_datasets_1])
# tokenized_datasets.save_to_disk(os.path.join(current_dir, tokenized_datasets_folder_name[0]))

# %%
# tokenized_datasets = load_from_disk(os.path.join(current_dir, tokenized_datasets_folder_name[0]))

# %% [markdown]
# ### load pretrained

# %%
tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

# %%
# tokenized_datasets_1 = load_from_disk(os.path.join(current_dir, tokenized_datasets_folder_name[1]))
# tokenized_datasets_2 = load_from_disk(os.path.join(current_dir, tokenized_datasets_folder_name[2]))
# tokenized_datasets_3 = load_from_disk(os.path.join(current_dir, tokenized_datasets_folder_name[3]))

# %%
# tokenized_datasets = load_from_disk(os.path.join(current_dir, tokenized_datasets_folder_name[0]))

# %% [markdown]
# # Grouping

# %% [markdown]
# ### from scratch

# %%
# # Main data processing function that will concatenate all texts from our dataset and generate chunks of
# # max_seq_length.
# model_max_length = 512
# def group_texts(examples):
#     # Concatenate all texts.
#     concatenated_examples = {k: list(chain(*examples[k])) for k in examples.keys()}
#     total_length = len(concatenated_examples[list(examples.keys())[0]])
#     # We drop the small remainder, we could add padding if the model supported it instead of this drop, you can
#     # customize this part to your needs.
#     if total_length >= model_max_length:
#         total_length = (total_length // model_max_length) * model_max_length
#     # Split by chunks of max_len.
#     result = {
#         k: [t[i : i + model_max_length] for i in range(0, total_length, model_max_length)]
#         for k, t in concatenated_examples.items()
#     }
#     return result

# %%
# tokenized_datasets = tokenized_datasets.map(group_texts, batched=True, num_proc=num_proc)
# # shuffle dataset
# tokenized_datasets = tokenized_datasets.shuffle(seed=34)

# print(f"the dataset contains in total {len(tokenized_datasets)*model_max_length} tokens")

# %%
# tokenized_datasets

# %%
# tokenized_datasets.save_to_disk(os.path.join(current_dir, grouped_tokenized_datasets_folder_name))

# %% [markdown]
# ### load pre-tokenized

# %%
tokenized_datasets = load_from_disk(os.path.join(current_dir, grouped_tokenized_datasets_folder_name))

# %% [markdown]
# # DDP Train

# %%
# os.environ [ "TF_FORCE_GPU_ALLOW_GROWTH" ] = "true"

# %%
model_config = AutoConfig.from_pretrained(old_model_name)
tokenizer.pad_token = tokenizer.eos_token

# %%
# DDP Train
def train_trainer_ddp():
    model = AutoModelForCausalLM.from_pretrained(old_model_name, config=model_config, cache_dir=model_cache_dir)
    model.resize_token_embeddings(len(tokenizer))

    training_args = TrainingArguments(
        output_dir = model_output_dir,
        logging_dir="runs/"+model_output_dir,
        do_train = True,
        do_eval = True,
        no_cuda = False,
        per_device_train_batch_size = 28,
        per_device_eval_batch_size = 28,        
        evaluation_strategy = "steps",
        eval_steps=1000,
        save_strategy="steps",
        save_steps=5000,
        logging_steps = 100,
        learning_rate = 5e-5,
        weight_decay = 0,
        adam_epsilon = 1e-8,
        max_grad_norm = 1.0,
        num_train_epochs = 10,
        disable_tqdm="false",
    )

    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, 
                                                mlm=False)

    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=tokenized_datasets['train'],
        eval_dataset=tokenized_datasets['valid'].select(range(100000))
    )   

    trainer.train()

notebook_launcher(train_trainer_ddp, args=(), num_processes=4)

# %%



