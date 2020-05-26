# dicts for each categorical selections
EDUCATION_LEVEL = {
    "Most 10 year olds": 0,
    "Not most 10 year olds": 1,
    "Only domain experts": 2,
    # "No one": 3,
    "Not sure": 4,
}
EDUCATION_LEVEL_ID = {v: u for u, v in EDUCATION_LEVEL.items()}


CLEARNESS = {
    "Everything": 0,
    "Did not understand": 1,
    "Did not make any sense": 2,
    "Underspecified": 3,
    "Not sure": 4,
    "None of the above": 5,
}
CLEARNESS_ID = {v: u for u, v in CLEARNESS.items()}


CATEGORIES = {
    "Typical Functions": 0,
    "Affordances": 1,
    "Spatial Relationships": 2,
    "Definitional Attributes": 3,
    "Everyday Knowledge": 4,
    # "None of the above": 5,
    # "Not sure": 5,
}
CATEGORIES_ID = {v: u for u, v in CATEGORIES.items()}


IF_COMMON_SENSE = {
    "Common Sense": 1,
    "Not Common Sense": -1,
    "Neutral": 0,
}
IF_COMMON_SENSE_ID = {v: u for u, v in IF_COMMON_SENSE.items()}


OVERALL = {
    "Performance": 0,
}
OVERALL_ID = {v: u for u, v in OVERALL.items()}


TASK_ABR = {
    "physicaliqa": "piqa",
    "physicalbinqa": "binpiqa",
}


CAT_NAMES = {
    "edu": "Demographics",
    "cat": "Physical Common Sense Categories",
    "com": "Is Common Sense",
    "mcq_com": "Is Common Sense",
    "clearness": "Clearness of Questions",
    "overall": "Overall Performances"
}
CAT_DICTS = {
    "edu": EDUCATION_LEVEL,
    "cat": CATEGORIES,
    "com": IF_COMMON_SENSE,
    "mcq_com": IF_COMMON_SENSE,
    "clearness": CLEARNESS,
    "overall": OVERALL,
}
CAT_ID_DICTS = {
    "edu": EDUCATION_LEVEL_ID,
    "cat": CATEGORIES_ID,
    "com": IF_COMMON_SENSE_ID,
    "mcq_com": IF_COMMON_SENSE_ID,
    "clearness": CLEARNESS_ID,
    "overall": OVERALL_ID,
}
