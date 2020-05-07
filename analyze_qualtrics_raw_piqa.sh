python3 analyze_qualtrics_raw.py \
    --input_csv_files "files/PIQASurveyTeLinToDistributePart1.csv" \
                      "files/PIQASurveyTeLinToDistributePart2.csv" \
    --task physicaliqa \
    --samples_csv ./files/piqa_survey_sampels.csv \
    --data_jsonl ./files/dev.jsonl \
    --data_lst ./files/dev-labels.lst \
    --verbose False \
    # --input_csv_files "files/PIQA-Survey-Te-Lin-To-Distribute-Part1_May 4, 2020_02.11.csv" \
    #                       "files/PIQA-Survey-Te-Lin-To-Distribute-Part2_May 4, 2020_02.14.csv" \
