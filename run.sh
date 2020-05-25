# for training
python3 main.py --load_config=fcn8s_shufflenet_train.yaml train Train FCN8sShuffleNet

# for testing with validation data
python3 main.py --load_config=fcn8s_shufflenet_test.yaml test_eval Train FCN8sShuffleNet

