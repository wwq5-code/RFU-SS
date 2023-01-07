import matplotlib.pyplot as plt
import numpy as np

# user num = 50
labels = ['0.05', '0.1', '0.15', '0.2', '0.25']
unl_fr = [10*10*2.76 , 10*10*2.76, 10*10*2.76 , 10*10*2.76, 10*10*2.76   ]
unl_br = [27*100/6000*2.76, 24*100/6000*2.76, 20*100/6000*2.76, 18*100/6000*2.76, 22*100/6000*2.76]
unl_vib = [31*100/6000*2.76, 24*100/6000*2.76, 20*100/6000*2.76, 24*100/6000*2.76, 20*100/6000*2.76]
unl_self_r = [2*56*100/6000*2.76, 2*50*100/6000*2.76, 2*56*100/6000*2.76, 2*50*100/6000*2.76, 2*48*100/6000*2.76]
unl_hess_r = [40*100/6000*2.76 +27.6 , 16*100/6000*2.76 +27.6, 9*100/6000*2.76 +27.6 , 17*100/6000*2.76 +27.6, 6*100/6000*2.76 +27.6 ]

# unl_hess_r = [,  3, 9, 10]

x = np.arange(len(labels))  # the label locations
width = 0.6  # the width of the bars
# no_noise = np.around(no_noise,0)
# samping = np.around(samping,0)
# ldp = np.around(ldp,0)


plt.figure()
#plt.subplots(figsize=(8, 5.3))
plt.bar(x - width / 4 - width / 8, unl_br, width=0.168, label='BIU', color='orange', hatch='\\')
plt.bar(x - width / 16, unl_vib, width=0.168, label='VIBU', color='silver', hatch='/')

plt.bar(x + width / 4, unl_self_r, width=0.168, label='IBFU-SS', color='g', hatch='x')


# plt.bar(x + width / 2 - width / 8, unl_hess_r, width=0.148, label='HFU', hatch='-')
# Add some text for labels, title and custom x-axis tick labels, etc.
plt.ylabel('Running Time (s)', fontsize=20)
# ax.set_title('Performance of Different Users n')
plt.xticks(x, labels, fontsize=20)
# ax.set_xticklabels(labels,fontsize=15)

my_y_ticks = np.arange(0, 7, 1.5)
plt.yticks(my_y_ticks, fontsize=20)
# ax.set_yticklabels(my_y_ticks,fontsize=15)

plt.legend(loc='upper right', fontsize=15)
plt.xlabel('$\\beta_u$' ,fontsize=20)
# ax.bar_label(rects1, padding=1)
# ax.bar_label(rects2, padding=3)
# ax.bar_label(rects3, padding=3)

plt.tight_layout()

plt.rcParams['figure.figsize'] = (2.0, 1)
plt.rcParams['image.interpolation'] = 'nearest'
plt.rcParams['figure.subplot.left'] = 0.11
plt.rcParams['figure.subplot.bottom'] = 0.08
plt.rcParams['figure.subplot.right'] = 0.977
plt.rcParams['figure.subplot.top'] = 0.969
plt.savefig('cifar_rt_ur_bar.png', dpi=200)
plt.show()
