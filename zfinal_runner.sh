
# IMPORTANT: The first three step must have RAW data from Rao's HiC
# storing the `root_dir`/raw as defined in `Arg_Parser.py`
# For example: "/data/RaoHiC/raw/GM12878/10kb_resolution_intrachromosomal"

# Reading raw data
# python Read_Data.py -c GM12878

# Downsampling data
# python Downsample.py -hr 10kb -lr 40kb -lrc 100 -r 16 -c GM12878

# Generating trainable/predictable data
# python Generate.py -hr 10kb -lr 40kb -chunk 40 -stride 40 -bound 201 -scale 1 -c GM12878

# Predicting data
# python 40x40_Predict.py -m HiCARN_1 -lr 40kb -ckpt root_dir/checkpoints/weights_file.pytorch -f hicarn_10kb40kb_c40_s40_b201_nonpool_human_GM12878_test.npz -c GM12878_HiCARN_1

# Training the model
# python HiCARN_1_Train.py
