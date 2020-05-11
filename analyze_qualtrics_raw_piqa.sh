python3 analyze_qualtrics_raw.py \
    --input_csv_files "files/PIQASurveyTeLinToDistributePart1.csv" \
                      "files/PIQASurveyTeLinToDistributePart2.csv" \
    --task physicaliqa \
    --samples_csv ./files/piqa_survey_sampels_all.csv \
    --data_jsonl ./files/dev.jsonl \
    --data_lst ./files/dev-labels.lst \
    --verbose False \
    --num_total_blocks 24 \
    # --start_block 22 \
    # --end_block 26 \
    # --input_csv_files "files/PIQA-Survey-Te-Lin-To-Distribute-Part1_May 4, 2020_02.11.csv" \
    #                       "files/PIQA-Survey-Te-Lin-To-Distribute-Part2_May 4, 2020_02.14.csv" \
