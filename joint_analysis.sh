python3 joint_analysis.py \
    --mcq_processed_ids_json_file "./outputs/piqa_all_cat_ids.json" \
    --bin_processed_ids_json_file "./outputs/binpiqa_all_cat_ids.json" \
    --mcq_samples_csv ./files/piqa_survey_sampels_all.csv \
    --bin_samples_csv ./files/binpiqa_survey_sampels_all.csv \
    --mcq_qualt_dict "./outputs/piqa_qualt_sorted_dict.json" \
    --bin_qualt_dict "./outputs/binpiqa_qualt_sorted_dict.json" \
    --mcq_model_preds "../ai2_light_telin/outputs/physicaliqa_ablated/large_roberta_bz4_accu2" \
    --bin_model_preds "../ai2_light_telin/outputs/physicalbinqa/param_sweep_binqa_5050/bz4_accu1_v2_contd" \
    --mcq_data_jsonl ./files/dev.jsonl \
    --mcq_data_lst ./files/dev-labels.lst \
    --bin_data_jsonl ./files/bin-dev.jsonl \
    --bin_data_lst ./files/bin-dev-labels.lst \
    --verbose False \
    --num_examples_to_show 0 \
    --num_total_blocks 24 \
    # --start_block 22 \
    # --end_block 26 \
