COLUMN_TIMESTAMP = 'time_idx'
COLUMN_GROUPID = 'group_id'
COLUMN_ID = []
COLUMN_TARGET = ['output_0']
COLUMN_OBSERVABLE = ["input_0", "input_1", "input_2", "input_3", "input_4", "input_5", "input_6"]
COLUMN_CONTROL = []
COLUMN_INPUT = ["input_0", "input_1", "input_2", "input_3", "input_4", "input_5", "input_6"]
# COLUMN_INPUT = ["input_0", "input_1", "input_2", "input_3", "input_4", "input_5", "input_6", "output_0"]

RATE_TRAINIG_DATASET = 0.8
RATE_VALIDATION_DATASET = 0.1
RATE_TEST_DATASET = 1 - RATE_TRAINIG_DATASET - RATE_VALIDATION_DATASET

SEED = 42