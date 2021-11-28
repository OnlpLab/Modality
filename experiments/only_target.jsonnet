
{
  "dataset_reader": {
    "type": "only_target_reader",
  },
  "train_data_path": "train.txt",
  "validation_data_path": "dev.txt",
  "model": {
    "type": "basic_classifier_model",
    "text_field_embedder": {
      "token_embedders": {
        "tokens": {
          "type": "embedding",
          "embedding_dim": 300,
          "pretrained_file": "https://allennlp.s3.amazonaws.com/datasets/glove/glove.840B.300d.txt.gz",
          "trainable": false
        }
      }
    },
    "seq2vec_encoder": {
       "type": "lstm",
       "input_size": 300,
       "hidden_size": 512,
       "num_layers": 2
    }
  },
  "data_loader": {
    "batch_sampler": {
      "type": "bucket",
      "batch_size" : 32
    }
  },
  "trainer": {
      "cuda_device": 3,
    "num_epochs": 5,
    "patience": 1,
    "grad_norm": 5.0,
    "validation_metric": "+accuracy",
    "optimizer": {
      "type": "adam",
      "lr": 0.001
    }
  }
}