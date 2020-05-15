python3 analyze_qualtrics_raw.py \
    --input_csv_files "files/BINPIQASurveyTeLinToDistributePart1.csv" \
                      "files/BINPIQASurveyTeLinToDistributePart2.csv" \
                      "files/BINPIQASurveyTeLinToDistributePart3.csv" \
    --task physicalbinqa \
    --samples_csv ./files/binpiqa_survey_sampels.csv \
    --data_jsonl ./files/bin-dev.jsonl \
    --data_lst ./files/bin-dev-labels.lst \
    --verbose False \
    --num_total_blocks 20 \
    # --start_block 22 \
    # --end_block 26 \
    # --input_csv_files "files/BIN-PIQA-Survey-Te-Lin-To-Distribute-Part1_May 4, 2020_15.39.csv" \
