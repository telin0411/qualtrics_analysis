python3 analyze_qualtrics_raw.py \
    --input_csv_files "files/BINPIQASurveyComplTeLinToDistributePart1.csv" \
                      "files/BINPIQASurveyComplTeLinToDistributePart2.csv" \
                      "files/BINPIQASurveyComplTeLinToDistributePart3.csv" \
    --task physicalbinqa \
    --samples_csv ./files/binpiqa_survey_sampels_complementary.csv \
    --data_jsonl ./files/bin-dev.jsonl \
    --data_lst ./files/bin-dev-labels.lst \
    --verbose False \
    --num_total_blocks 17 \
    # --start_block 22 \
    # --end_block 26 \
