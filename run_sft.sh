
for block_id in {0..12}; do
  CUDA_VISIBLE_DEVICES=7 python trainerSFT.py --prefix_tag cudacoder_eval_4_turn.deepseek_r1.sft13 --epoch_id 0 \
        --block_id $(printf "%02d" $block_id) --input_tag cudacoder_eval_4_turn.qwen32b_000_$(printf "%02d" $block_id)/deepseek-reasoner-sft5 \
        --input_dir shared/re_kbeval/.inference/codeGenEval/
done[]
