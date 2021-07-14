
local transformer_model = "roberta-base";
local transformer_dim = 1024;

{
  "dataset_reader":{
    "type": "basic_classifier_reader",
      "token_indexers": {
      "tokens": {
        "type": "pretrained_transformer",
        "model_name": transformer_model
      }
    },
    "tokenizer": {
      "type": "pretrained_transformer",
      "model_name": transformer_model
    }
  },
  "train_data_path": "train.txt",
  "validation_data_path": "dev.txt",
  "model": {
    "type": "basic_classifier_model",
    "text_field_embedder": {
      "token_embedders": {
        "tokens": {
          "type": "pretrained_transformer",
          "model_name": transformer_model
        }
      }
    },
    "seq2vec_encoder": {
       "type": "bert_pooler",
       "pretrained_model": transformer_model,
       "dropout": 0.1,
    },
    "namespace": "tags"
  },
  "data_loader": {
    "batch_sampler": {
      "type": "bucket",
      "sorting_keys": ["tokens"],
      "batch_size" : 8
    }
  },
  "trainer": {
    "cuda_device": 1,
    "num_epochs": 10,
    "validation_metric": "+accuracy",
    "learning_rate_scheduler": {
      "type": "slanted_triangular",
      "num_epochs": 10,
      "num_steps_per_epoch": 706,
      "cut_frac": 0.06
    },
    "optimizer": {
      "type": "huggingface_adamw",
      "lr": 2e-5,
      "weight_decay": 0.1,
    }
  }
}