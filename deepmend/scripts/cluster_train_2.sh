
DATADIR1='/home/michael/test_shapeNet/ShapeNet/ShapeNetCore.v2'
DATADIR2='/home/michael/test_shapeNet_inference2/ShapeNet/ShapeNetCore.v2'
json_file='/home/michael/test_shapeNet/ShapeNet/ShapeNetCore.v2/cars_split.json'
object_class='cars'
path_model='/home/michael/model/latest.pth'
python python/train_2_all_cluster.py \
    --k 23 \
    --datadir_1 "$DATADIR1" \
    --datadir_2 "$DATADIR2" \
    --train_test_file "$json_file" \
    --object_class "$object_class" \
    --path_model "$path_model" 
