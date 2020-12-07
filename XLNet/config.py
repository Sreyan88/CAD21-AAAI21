# Defines all paths and model hyperparameters

# Dataset files' paths
train_file = 'train.txt'
dev_file = 'dev.txt'
test_file = 'test.txt'

# Preprocessing variables
# maximum length of sentence after tokenization
MAX_LEN = 185

# tokenization using caseless or case tokens
to_case = False

# model name
model_name = 'xlnet-base-cased'

# Model hyperparameters
to_freeze = False
freeze_layers = 23

xlnet_dim = 13*768
hidden_dim1 = 256
hidden_dim2 = 64
final_size = 1

dropout_prob = 0.3

# batch size for training.
batch_size = 16

# Optimizer parameters
learning_rate = 2e-5
epsilon = 1e-8


num_epochs = 100

max_accuracy = 0
max_match = [0,0,0]

val_out = ""
test_out = ""

save_path = './'
# Index of the run of current model, change it after each run
ind = 1
