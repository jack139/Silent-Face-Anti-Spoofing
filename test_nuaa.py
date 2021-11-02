import os
from test_img import fas_check


#nuaa_root = "/home/tao/Downloads/CelebA_Spoof_zip2/CelebA_Spoof/CelebA_Spoof_Croped/Data"
#train_csv = "high_30k_train.csv"
#val_csv = "high_30k_test.csv"

nuaa_root = "/media/tao/_dde_data/Datasets/CelebA_Spoof_Croped/Data"
train_csv = "train.csv"
val_csv = "test.csv"


def test(input_file):
    N = 0
    T = 0
    with open(input_file, 'r') as f:
        for l in f:
            d = l.strip().split(',')
            _, img_file = os.path.split(d[0])
            img_path = os.path.join(nuaa_root, d[0])
            r, score = fas_check(img_path)
            N += 1
            spoof = 0 if r==False else 1
            if spoof==int(d[1]):
                T += 1

            print(img_file, score, d[1], spoof)

    return T/N

# val 0.7081

if __name__ == '__main__':
    #acc1 = test(os.path.join(nuaa_root, train_csv))
    #print(acc1)

    acc2 = test(os.path.join(nuaa_root, val_csv))
    print(acc2)
    #print(acc1, acc2)
