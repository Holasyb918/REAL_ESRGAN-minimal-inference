set -eu

input=asserts
outputs=outputs
model_name=RealESRGAN_x2
# model_name=RealESRGAN_x4plus_anime_6B
scale=4

python sr_minimal_inference.py \
    --input $input \
    --output $outputs \
    --model_name $model_name \
    --scale $scale
