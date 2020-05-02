python3 analyze_qualtrics_raw.py \
    --input_csv_files "files/BIN-PIQA-Survey-Te-Lin-To-Distribute-Part1_April 28, 2020_16.14.csv" \
    --task physicalbinqa \
    --samples_csv ./files/binpiqa_survey_sampels.csv \
    --data_jsonl ./files/bin-dev.jsonl \
    --data_lst ./files/bin-dev-labels.lst \
    --verbose False \
