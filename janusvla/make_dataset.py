import tensorflow_datasets as tfds
import tensorflow as tf
# 加载数据集，并指定下载目录为 ./datasets
ds_train,metadata= tfds.load(
    'ucsd_pick_and_place_dataset_converted_externally_to_rlds',
    data_dir='./datasets',
    download=True,
    try_gcs=False,
    split=['train'],
    with_info=True,
    
)
print(metadata)
train_dataset = ds_train[0]
# 随机打乱数据集
train_dataset = train_dataset.shuffle(buffer_size=metadata.splits['train'].num_examples)
print(train_dataset)
# 取一个示例
example = next(iter(train_dataset.take(1)))
print(metadata.splits['train'].num_examples)
# 打印示例
steps = example['steps']
print(f"Number of Steps: {len(steps)}")
for step in steps:
    language_instruction = step['language_instruction'].numpy().decode('utf-8')
    print(f"Language Instruction: {language_instruction}")
    state = step['observation']['state'].numpy()
    image = step['observation']['image'].numpy()
    print(f"Observation Image Shape: {image.shape}")
    print(f"Observation State: {state}")