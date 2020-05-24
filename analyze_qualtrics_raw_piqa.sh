python3 analyze_qualtrics_raw.py \
    --input_csv_files "files/PIQASurveyTeLinToDistributePart1.csv" \
                      "files/PIQASurveyTeLinToDistributePart2.csv" \
                      "files/PIQASurveyTeLinToDistributePart3.csv" \
    --task physicaliqa \
    --samples_csv ./files/piqa_survey_sampels_all.csv \
    --data_jsonl ./files/dev.jsonl \
    --data_lst ./files/dev-labels.lst \
    --verbose False \
    --num_total_blocks 24 \
    # --start_block 22 \
    # --end_block 26 \
