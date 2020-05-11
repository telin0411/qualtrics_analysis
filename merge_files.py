import csv


task = "binpiqa"
task = "piqa"


f1 = "files/{}_survey_sampels.csv".format(task)
f2 = "files/{}_survey_sampels_hard.csv".format(task)

fr1 = csv.DictReader(open(f1, "r"))
fr2 = csv.DictReader(open(f2, "r"))

data = []

for row in fr1:
    if task == "piqa":
        goal, sol1, sol2 = row["goal"], row["sol1"], row["sol2"]
        new_row = {"goal": goal, "sol1": sol1, "sol2": sol2}
    else:
        goal, sol = row["goal"], row["sol"]
        new_row = {"goal": goal, "sol": sol}
    data.append(new_row)

for row in fr2:
    if task == "piqa":
        goal, sol1, sol2 = row["goal"], row["sol1"], row["sol2"]
        new_row = {"goal": goal, "sol1": sol1, "sol2": sol2}
    else:
        goal, sol = row["goal"], row["sol"]
        new_row = {"goal": goal, "sol": sol}
    data.append(new_row)

print (len(data))

if task == "piqa":
    fieldnames = ["goal", "sol1", "sol2"]
else:
    fieldnames = ["goal", "sol"]
fo = "files/{}_survey_sampels_all.csv".format(task)
fro = csv.DictWriter(open(fo, "w"), fieldnames=fieldnames)
fro.writeheader()
for new_row in data:
    fro.writerow(new_row)

if task == "piqa":
    empty_row = {"goal":"", "sol1":"", "sol2":""}
else:
    empty_row = {"goal":"", "sol":""}
fro.writerow(empty_row)
