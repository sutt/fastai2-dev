# import time
# from matplotlib import pyplot as plt

# height = [0.8755819797515869,
#         0.018214376643300056,
#         0.0069943685084581375,
#         0.00005601025259238668,
#         0.004579802043735981,
#         0.0017798723420128226,
#         0.00743057020008564,
#         0.0018005078891292214,
#         0.0681077167391777,
#         0.0005398029461503029,
#         0.01424149889498949,
#         0.0006736230570822954]

# x = ['black-bishop',
#     'black-king',
#     'black-knight',
#     'black-pawn',
#     'black-queen',
#     'black-rook',
#     'white-bishop',
#     'white-king',
#     'white-knight',
#     'white-pawn',
#     'white-queen',
#     'white-rook']


# print('hey')
# plt.bar(x, height)
# plt.xticks(rotation='vertical')
# plt.show()
# time.sleep(3)

# https://stackoverflow.com/questions/11874767/how-do-i-plot-in-real-time-in-a-while-loop-using-matplotlib
import numpy as np
import matplotlib.pyplot as plt

plt.axis([0, 10, 0, 1])

for i in range(10):
    y = np.random.random()
    plt.scatter(i, y)
    plt.pause(0.05)

# plt.show()