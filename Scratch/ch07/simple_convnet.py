# coding: utf-8
import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
import numpy as np
import gzip
import matplotlib.pyplot as plt

# =====================================================================
# [필수 수정] 교재 소스코드(common 폴더가 들어있는)의 '절대 경로'를 적어주세요.
BOOK_PATH = r"C:\Users\김민규\Desktop\deep-learning-from-scratch-master"
# =====================================================================

sys.path.append(BOOK_PATH)

from common.util import smooth_curve
from common.optimizer import Adam
from ch07.simple_convnet import SimpleConvNet

# 교재 load_mnist 대신 Fashion MNIST 로컬 로더 (CNN용 4차원)
def load_fashion_mnist(data_path, normalize=True):
    all_files = os.listdir(data_path)

    def find_file(keyword):
        for f in all_files:
            if keyword in f: return f
        raise FileNotFoundError(f"{keyword} 파일을 찾을 수 없습니다.")

    x_train = np.frombuffer(gzip.decompress(open(os.path.join(data_path, find_file('train-images')), 'rb').read()), np.uint8, offset=16).reshape(-1, 1, 28, 28)
    x_test  = np.frombuffer(gzip.decompress(open(os.path.join(data_path, find_file('t10k-images')),  'rb').read()), np.uint8, offset=16).reshape(-1, 1, 28, 28)
    t_train = np.frombuffer(gzip.decompress(open(os.path.join(data_path, find_file('train-labels')), 'rb').read()), np.uint8, offset=8)
    t_test  = np.frombuffer(gzip.decompress(open(os.path.join(data_path, find_file('t10k-labels')),  'rb').read()), np.uint8, offset=8)

    if normalize:
        x_train = x_train.astype(np.float32) / 255.0
        x_test  = x_test.astype(np.float32)  / 255.0

    return (x_train, t_train), (x_test, t_test)

# 0. Fashion MNIST 데이터 읽기==========
current_dir = os.path.dirname(os.path.abspath(__file__))
(x_train, t_train), (x_test, t_test) = load_fashion_mnist(os.path.join(current_dir, 'fashion'))

train_size = x_train.shape[0]
batch_size = 128
max_iterations = 6000

# 1. 실험용 설정==========
optimizers = {}
optimizers['Adam'] = Adam()

networks = {}
train_loss = {}
for key in optimizers.keys():
    networks[key] = SimpleConvNet(
        input_dim=(1, 28, 28),
        conv_param={'filter_num': 64, 'filter_size': 3, 'pad': 1, 'stride': 1},
        hidden_size=256, output_size=10, weight_init_std=0.01)
    train_loss[key] = []

# 2. 훈련 시작==========
for i in range(max_iterations):
    batch_mask = np.random.choice(train_size, batch_size)
    x_batch = x_train[batch_mask]
    t_batch = t_train[batch_mask]

    for key in optimizers.keys():
        grads = networks[key].gradient(x_batch, t_batch)
        optimizers[key].update(networks[key].params, grads)

        loss = networks[key].loss(x_batch, t_batch)
        train_loss[key].append(loss)

    if i % 100 == 0:
        print("===========" + "iteration:" + str(i) + "===========")
        train_acc = networks['Adam'].accuracy(x_train[:3000], t_train[:3000])
        test_acc  = networks['Adam'].accuracy(x_test[:1000],  t_test[:1000])
        print("train acc:" + str(round(train_acc * 100, 2)) + "%, test acc:" + str(round(test_acc * 100, 2)) + "%")

# 최종 정확도==========
print("\n========== 최종 정확도 ==========")
train_acc = networks['Adam'].accuracy(x_train, t_train)
test_acc  = networks['Adam'].accuracy(x_test,  t_test)
print("train acc:" + str(round(train_acc * 100, 2)) + "%, test acc:" + str(round(test_acc * 100, 2)) + "%")

# 3. 그래프 그리기==========
markers = {"Adam": "D"}
x = np.arange(max_iterations)
for key in optimizers.keys():
    plt.plot(x, smooth_curve(train_loss[key]), marker=markers[key], markevery=100, label=key)
plt.xlabel("iterations")
plt.ylabel("loss")
plt.ylim(0, 1)
plt.legend()
plt.show()
