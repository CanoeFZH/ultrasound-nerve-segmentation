import numpy as np

imgs_mask_test = np.load('imgs_mask_test.npy')
imgs_mask_test_bin = np.load('imgs_mask_test_bin.npy')
res = []
for t ,b in zip(imgs_mask_test,imgs_mask_test_bin):
    if b >=0.5:
        res.append(t)
    else:
        res.append(np.zeros(t.shape))

res = np.array(res)
print res.shape
np.save('imgs_mask_test.npy', res)