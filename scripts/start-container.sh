set -eu

IMG=registry.cn-beijing.aliyuncs.com/oneflow/oneflow-sd:cu116
docker pull ${IMG}
docker run --rm -it \
    --gpus all --ipc=host --ulimit memlock=-1 --ulimit stack=67108864 \
    -v ${HF_HOME}:${HF_HOME} \
    -v ${PWD}:${PWD} \
    -w ${PWD} \
    -e HF_HOME=${HF_HOME} \
    -e HUGGING_FACE_HUB_TOKEN=${HUGGING_FACE_HUB_TOKEN} \
    ${IMG} \
    python3 /demos/oneflow-t2i.py
# oneflowinc/oneflow-sd:cu116
