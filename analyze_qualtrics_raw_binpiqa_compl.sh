python3 analyze_qualtrics_raw.py \
    --input_csv_files "files/BINPIQASurveyComplTeLinToDistributePart1.csv" \
    --task physicalbinqa \
    --samples_csv ./files/binpiqa_survey_sampels_complementary.csv \
    --data_jsonl ./files/bin-dev.jsonl \
    --data_lst ./files/bin-dev-labels.lst \
    --verbose False \
    --num_total_blocks 16 \
    # --start_block 22 \
    # --end_block 26 \
