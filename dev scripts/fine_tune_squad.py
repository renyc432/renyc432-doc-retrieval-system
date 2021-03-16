

path = 'C:\\Users\\roy79\\Desktop\\Research\\question answering system\\code\\'
os.chdir(path)


pip install -r albert/requirements.txt
python -m albert.run_squad_v2_revised \
  --albert_config_file=albert_config.json \
  --output_dir=./out/model_tuned \
  --train_file=SQuAD_train-v2.0.json \
  --predict_file=SQuAD_dev-v2.0.json \
  --spm_model_file=30k-clean.model \
  --do_lower_case=True \
  --max_seq_length=384 \
  --doc_stride=128 \
  --max_query_length=64 \
  --train_batch_size=8 \
  --predict_batch_size=8 \
  --learning_rate=5e-5 \
  --num_train_epochs=1.0 \
  --warmup_proportion=.1 \
  --save_checkpoints_steps=1000 \
  --n_best_size=20 \
  --max_answer_length=30 \
  --do_train=True \
  --do_predict=True \
  --train_feature_file=train_feature_file.tfrecord \
  --predict_feature_file=predict_feature_file.tfrecord \
  --predict_feature_left_file=predict_feature_left_file.tfrecord \
  --init_checkpoint=./albert_base/model.ckpt-best \
    
    
python -m albert.run_squad_v2_revised --albert_config_file=albert_config.json --output_dir=./out/model_tuned --train_file=SQuAD_train-v2.0.json --predict_file=SQuAD_dev-v2.0.json --spm_model_file=30k-clean.model --do_lower_case=True --max_seq_length=384 --doc_stride=128 --max_query_length=64 --train_batch_size=8 --predict_batch_size=16 --learning_rate=5e-5 --num_train_epochs=1.0 --warmup_proportion=.1 --save_checkpoints_steps=1000 --n_best_size=20 --max_answer_length=30 --do_train=True --do_predict=True --train_feature_file=train_feature_file.tfrecord --predict_feature_file=predict_feature_file.tfrecord --predict_feature_left_file=predict_feature_left_file.tfrecord --init_checkpoint=./albert_base/model.ckpt-best