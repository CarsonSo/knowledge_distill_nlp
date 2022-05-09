export OUTPUT_DIR=../../TextBrewer/output_model/conll_distill/student_raw_layer9
export BATCH_SIZE=32
export NUM_EPOCHS=20
export SAVE_STEPS=750
export SEED=42
export MAX_LENGTH=128
export BERT_MODEL_TEACHER=../../TextBrewer/output_model/conll_distill/teacher_raw
python run_ner_distill.py \
--data_dir ../../data/conll \
--model_type bert \
--model_name_or_path $BERT_MODEL_TEACHER \
--output_dir $OUTPUT_DIR \
--max_seq_length  $MAX_LENGTH \
--num_train_epochs $NUM_EPOCHS \
--per_gpu_train_batch_size $BATCH_SIZE \
--num_hidden_layers 9 \
--save_steps $SAVE_STEPS \
--learning_rate 1e-4 \
--warmup_steps 0.1 \
--seed $SEED \
--do_distill \
--do_train \
--do_eval \
--do_predict \
--overwrite_output_dir

