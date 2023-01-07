import matplotlib.pyplot as plt
import numpy as np

# user num = 50
labels = ['2%', '4%', '6%', '8%', '10%']
unl_fr = [10*10*2.76 , 10*10*2.76, 10*10*2.76 , 10*10*2.76, 10*10*2.76   ]
unl_br = [44*100/6000*2.76, 22*100/6000*2.76, 24*100/6000*2.76, 32*100/6000*2.76, 37*100/6000*2.76]
unl_vib = [44*100/6000*2.76, 23*100/6000*2.76, 24*100/6000*2.76, 30*100/6000*2.76, 34*100/6000*2.76]
unl_self_r = [2*80*100/6000*2.76, 2*57*100/6000*2.76, 2*50*100/6000*2.76, 2*76*100/6000*2.76, 2*78*100/6000*2.76]
unl_hess_r = [40*100/6000*2.76 +27.6 , 16*100/6000*2.76 +27.6, 9*100/6000*2.76 +27.6 , 17*100/6000*2.76 +27.6, 16*100/6000*2.76 +27.6 ]

x = np.arange(len(labels))  # the label locations
width = 0.6  # the width of the bars
# no_noise = np.around(no_noise,0)
# samping = np.around(samping,0)
# ldp = np.around(ldp,0)


plt.figure()
#plt.subplots(figsize=(8, 5.3))
#plt.bar(x - width / 2 - width / 8 + width / 8, unl_fr, width=0.168, label='Retrain', color='dodgerblue', hatch='/')
plt.bar(x - width / 2 - width / 8 + width / 8, unl_br, width=0.168, label='VBU', color='orange', hatch='\\')
plt.bar(x - width / 8 - width / 16, unl_vib, width=0.168, label='VIBU', color='silver', hatch='/')
plt.bar(x + width / 8, unl_self_r, width=0.168, label='VIBU-SS', color='g', hatch='x')
plt.bar(x + width / 2 - width / 8 + width / 16, unl_hess_r, width=0.168, label='HBU', color='tomato', hatch='-')



# Add some text for labels, title and custom x-axis tick labels, etc.
plt.ylabel('Running Time (s)', fontsize=20)
# ax.set_title('Performance of Different Users n')
plt.xticks(x, labels, fontsize=20)
# ax.set_xticklabels(labels,fontsize=15)

my_y_ticks = np.arange(0, 33.1, 5)
plt.yticks(my_y_ticks, fontsize=20)
# ax.set_yticklabels(my_y_ticks,fontsize=15)

plt.legend(loc='upper left', fontsize=15)
plt.xlabel('$\it{EDR}$' ,fontsize=20)
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
plt.savefig('cifar_rt_er_bar.png', dpi=200)
plt.show()
