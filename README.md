# Qualtrics Annotations Analysis Codes

## Basic usage
Refer to the script `analyze_qualtrics_raw_piqa.sh` and/or `analyze_qualtrics_raw_binpiqa.sh`.  
Execute with following command
```bash
sh analyze_qualtrics_raw_piqa.sh
```
The script will then print out various analysis results from the given input fields.  

The most important fileds will be `--input_csv_files`, and it can take multiple input csv files, eg.
```bash
python3 analyze_qualtrics_raw.py --input_csv_files [CSV_FILE1] [CSV_FILE2] ... --[Other args] ...
```
Currently in the example it is only taking the one from `Part1`, it can handle `Part2` as well.  

the rest should be easy to modify in the script.

### Legacy File  
Use the following command to transform files under `legacy_files` folder to the usual format.  
(Qualtrics legacy exporter exports fields a bit different from normal exporter.)
```bash
python3 transform_legacy.py --input_legacy_file legacy_files/PIQASurveyTeLinToDistributePart2.csv --out_file_path files/PIQASurveyTeLinToDistributePart2.csv --num_blocks 24
```


## Functions & Data Schema
In `analyze_qualtrics_raw.py`, refer to the function `analyze_pipeline` for the analysis engine.  
Each data instance from our sampled PIQA/BINPIQA will be augmented with the qualtrics results, as `qualt_sorted_dict`.  
The sample execution above will also show one data instance to have a peak. In general it should look like:

```bash
Qualtric ID: 10_Q10
{ 'annotations': { '1. choice': [1],
                   '2. confidence': [100.0],
                   '3. others agreement': [90.0],
                   '4. if common sense': [1],
                   '5. education level': [0],
                   '6. clearness': [0],
                   '7. category': [[0, 4]]},
  'goal': 'To learn how to ride a bike.',
  'gt_label': 1,
  'id': '3e2db322-eca6-4f1d-807e-0fb02e193e7d',
  'sol1': 'Just get on the bike and start riding to learn how to do it '
          'consistently.',
  'sol2': 'Start with training wheels to help will balance while moving then '
          'gradually take them off.'}
```

`simple_analysis` function will be the main function processing the resulting augmented data instance.  
There will be new functions and handling model performances added later.
