# Bert-Pytorch-Chinese-TextClassification
Pytorch Bert Finetune in Chinese Text Classification

### Step 1

Download the pretrained TensorFlow model:[chinese_L-12_H-768_A-12](https://storage.googleapis.com/bert_models/2018_11_03/chinese_L-12_H-768_A-12.zip)

### Step 2

Change the TensorFlow Pretrained Model into Pytorch

```shell
cd  convert_tf_to_pytorch
```

```shell
export BERT_BASE_DIR=/workspace/mnt/group/ocr/xieyufei/bert-tf-chinese/chinese_L-12_H-768_A-12

python3 convert_tf_checkpoint_to_pytorch.py \
  --tf_checkpoint_path $BERT_BASE_DIR/bert_model.ckpt \
  --bert_config_file $BERT_BASE_DIR/bert_config.json \
  --pytorch_dump_path $BERT_BASE_DIR/pytorch_model.bin
```

### Step 3

Download the Chinese News DataSet:[Train](https://pan.baidu.com/s/15rkzx-YRbP5XRNeapzYWLw) for 5w and [Dev](https://pan.baidu.com/s/1HuYTacgAQFqGAJ8FYXNqOw) for 5k

### Step 4

Just Train and Test

```shell
cd src
```

```shell
export GLUE_DIR=/workspace/mnt/group/ocr/xieyufei/bert-tf-chinese/glue_data
export BERT_BASE_DIR=/workspace/mnt/group/ocr/xieyufei/bert-tf-chinese/chinese_L-12_H-768_A-12/
export BERT_PYTORCH_DIR=/workspace/mnt/group/ocr/xieyufei/bert-tf-chinese/chinese_L-12_H-768_A-12/

python3 run_classifier_word.py \
  --task_name NEWS \
  --do_train \
  --do_eval \
  --data_dir $GLUE_DIR/SouGou/ \
  --vocab_file $BERT_BASE_DIR/vocab.txt \
  --bert_config_file $BERT_BASE_DIR/bert_config.json \
  --init_checkpoint $BERT_PYTORCH_DIR/pytorch_model.bin \
  --max_seq_length 256 \
  --train_batch_size 24 \
  --learning_rate 2e-5 \
  --num_train_epochs 50.0 \
  --output_dir ./newsAll_output/ \
  --local_rank 3
```

1个Epoch的结果如下：

```
eval_accuracy = 0.9742
eval_loss = 0.10202122390270234
global_step = 2084
loss = 0.15899521649851786
```



