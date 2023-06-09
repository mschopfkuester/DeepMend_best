EXPERIMENT="$1"
INPUT_FILE="$2"
INPUT_DIR=`dirname $INPUT_FILE`
if [ ! -f "$INPUT_DIR"/model_normalized_oriented_scaled.obj ];
then
    python ../fracturing/processor/process_normalize.py \
        "$INPUT_FILE" \
        "$INPUT_DIR"/model_normalized_oriented_scaled.obj \
        --skip_check \
        --debug
fi
if [ ! -f "$INPUT_DIR"/pts.npz ];
then
python ../fracturing/processor/process_sample.py \
    "$INPUT_DIR"/model_normalized_oriented_scaled.obj \
    "$INPUT_DIR"/pts.npz \
    --debug
fi
if [ ! -f "$INPUT_DIR"/sdf.npz ];
then
python ../fracturing/processor/process_sdf.py \
    "$INPUT_DIR"/model_normalized_oriented_scaled.obj \
    "$INPUT_DIR"/sdf.npz \
    --samples "$INPUT_DIR"/pts.npz \
    --debug
fi

if [ ! -f "$INPUT_DIR"/"$INPUT_DIR"/code"$ATTEMPT""$NAME".pth ];
then

NAME="_test"
UNIFORM_RATIO="0.2"
CHK="2000"
NUMITS="1600"
LREG="0.0001"
LR="0.005"
LAMBDAner="0.00001"
LAMBDAPROX="0.005"

echo "Running ""$INPUT_FILE"" using experiment ""$EXPERIMENT"

python python/reconstruct_single.py \
    -e "$EXPERIMENT" \
    -c "$CHK" \
    --input_mesh "$INPUT_DIR"/model_normalized_oriented_scaled.obj \
    --input_points "$INPUT_DIR"/pts.npz \
    --input_sdf "$INPUT_DIR"/sdf.npz \
    --output_meshes "$INPUT_DIR"/predicted"$ATTEMPT""$NAME".obj \
    --output_code "$INPUT_DIR"/code"$ATTEMPT""$NAME".pth \
    --num_iters "$NUMITS" \
    --lambda_reg "$LREG" \
    --learning_rate "$LR" \
    --lambda_ner "$LAMBDAner" \
    --lambda_prox "$LAMBDAPROX" \
    --uniform_ratio "$UNIFORM_RATIO" \
    --debug --overwrite --gif
fi
