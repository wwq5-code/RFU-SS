

import numpy as np
import matplotlib.pyplot as plt

epsilon = 3
beta = 1 / epsilon


y_bfu_back_acc = [1.0   , 1.0   , 1.0   , 1.0   , 1.0   , 0.9799, 0.9899, 0.9599, 0.9799999594688416, 0.9099999666213989, 0.85999995470047, 0.8100, 0.8399, 0.7699999809265137, 0.72999995946884, 0.6800000071525574, 0.6699999570846558, 0.610000014305114, 0.599999964237213, 0.4799999892711639, 0.3999999761581421, 0.389999985694885, 0.299999982118606, 0.34000000357627, 0.26999998092651, 0.2299999892711639, 0.1099999994039535, 0.1899999976158142, 0.0599999986588954, 0.0999999940395355, 0.0199999995529651, 0.0999999940395355, 0.0199999995529651, 0.0999999940395355, 0.0199999995529651,  0.0199999995529651]
y_bfu_acc      = [0.9899, 1.0   , 1.0   , 1.0   , 1.0   , 0.9799, 0.9899, 0.9699, 0.9799999594688416, 0.9699999690055847, 0.98999994993209, 0.9599, 0.9799, 0.9599999785423279, 0.94999998807907, 0.9599999785423279, 0.9099999666213989, 0.979999959468841, 0.949999988079071, 0.9599999785423279, 0.9399999976158142, 0.949999988079071, 0.949999988079071, 0.93999999761581, 0.94999998807907, 0.9699999690055847, 0.9599999785423279, 0.9300000071525574, 0.9300000071525574, 0.9099999666213989, 0.8799999952316284, 0.9099999666213989, 0.8799999952316284, 0.9099999666213989, 0.8799999952316284,  0.8799999952316284]
y_vibu_back_acc= [1.0   , 1.0   , 1.0   , 1.0   , 1.0   , 0.9899, 0.9699, 0.9499, 0.9099999666213989, 0.8999999761581421, 0.87000000476837, 0.8700, 0.8599, 0.8100000023841858, 0.77999997138977, 0.75              , 0.7599999904632568, 0.699999988079071, 0.709999978542327, 0.5899999737739563, 0.5099999904632568, 0.569999992847442, 0.560000002384185, 0.47999998927116, 0.55000001192092, 0.5399999618530273, 0.5600000023841858, 0.4499999880790714, 0.3899999856948852, 0.3999999761581421, 0.2999999821186065, 0.1700000017881393, 0.2199999988079071, 0.1299999952316284, 0.04999999701976776, 0.04999999701976776]
y_vibu_acc     = [0.9899, 0.9899, 1.0   , 0.9899, 1.0   , 1.0   , 0.9799, 0.9699, 0.9799999594688416, 0.949999988079071 , 0.96999996900558, 0.9699, 0.9799, 0.9599999785423279, 0.98999994993209, 0.9599999785423279, 0.9199999570846558, 0.989999949932098, 0.979999959468841, 0.9799999594688416, 0.9899999499320984, 0.989999949932098, 0.969999969005584, 0.94999998807907, 0.96999996900558, 0.9599999785423279, 0.9599999785423279, 0.9899999499320984, 0.9099999666213989, 0.9399999976158142, 0.9399999976158142, 0.9399999976158142, 0.9300000071525574, 0.9399999976158142, 0.9399999976158142, 0.9399999976158142]
y_ss_back_acc  = [1.0   , 1.0   , 1.0   , 1.0   , 0.9899, 1.0   , 0.9799, 0.9799, 0.949999988079071 , 0.9399999976158142, 0.91999995708465, 0.8499, 0.8499, 0.7799999713897705, 0.85999995470047, 0.8799999952316284, 0.75              , 0.740000009536743, 0.699999988079071, 0.7299999594688416, 0.7299999594688416, 0.729999959468841, 0.719999969005584, 0.65999996662139, 0.56000000238418, 0.5699999928474426, 0.4799999892711639, 0.5199999809265137, 0.4499999880790710, 0.5               , 0.3999999761581421, 0.3499999940395355, 0.2999999821186065, 0.3799999952316284, 0.22999998927116394, 0.17999999225139618]
y_ss_acc       = [0.9899, 1.0   , 1.0   , 0.9899, 1.0   , 0.9899, 0.9899, 1.0   , 0.9799999594688416, 0.9899999499320984, 0.98999994993209, 1.0   , 1.0   , 0.9899999499320984, 1.0             , 0.9899999499320984, 1.0               , 1.0              , 1.0              , 1.0               , 0.9899999499320984, 0.969999969005584, 1.0              , 1.0             , 1.0             , 0.9899999499320984, 0.9899999499320984, 1.0               , 0.9399999976158142, 0.9799999594688416, 0.9899999499320984, 0.9899999499320984, 0.9699999690055847, 1.0               , 1.0                , 1.0]

y_hfu_back_acc = [0.99  , 0.2899, 0.0   , 0.0   , 0    ]
y_hfu_acc      = [1.0   , 1.0   , 1.0   , 1.0   , 1.0   , 0.9899, 1.0   , 0.9799, 0.17999999225139618, 0.0]

x=[]
y_unl_s = []
y_unl_self_s =[]
y_nips_rkl_s =[]
y_hessian_30_s =[]
for i in range(36):
    # print(np.random.laplace(0, 1)/10+0.2)
    x.append(i)
    #y_fkl[i] = y_fkl[i*2]*100
    y_bfu_back_acc[i] = y_bfu_back_acc[i]*100
    y_bfu_acc[i] = y_bfu_acc[i]*100
    y_vibu_back_acc[i] = y_vibu_back_acc[i]*100
    y_vibu_acc[i] = y_vibu_acc[i]*100
    y_ss_back_acc[i] = y_ss_back_acc[i]*100
    y_ss_acc[i] = y_ss_acc[i]*100
    # y_hfu_back_acc[i] = y_hfu_back_acc[i]*100
    # y_hfu_acc[i] = y_hfu_acc[i]*100


plt.figure()
plt.plot(x, y_bfu_acc, color='orange',  marker='x',  label='BFU',linewidth=4,  markersize=10)
plt.plot(x, y_ss_acc, color='g',  marker='*',  label='BFU-SS',linewidth=4, markersize=10)
# #plt.plot(x, y_fkl, color='g',  marker='+',  label='VRFL')
plt.plot(x, y_vibu_acc, color='r',  marker='p',  label='HBU',linewidth=4, markersize=10)

# plt.plot(x, unl_fr, color='blue', marker='^', label='Retrain',linewidth=4, markersize=10)
# plt.plot(x, unl_br, color='orange',  marker='x',  label='BFU',linewidth=4,  markersize=10)
# plt.plot(x, unl_self_r, color='g',  marker='*',  label='BFU-SS',linewidth=4, markersize=10)
# plt.plot(x, unl_hess_r, color='r',  marker='p',  label='HFU',linewidth=4, markersize=10)

# plt.plot(x, y_unl_s, color='b', marker='^', label='Normal Bayessian Fed Unlearning',linewidth=3, markersize=8)
# plt.plot(x, y_unl_self_s, color='r',  marker='x',  label='Self-sharing Fed Unlearning',linewidth=3, markersize=8)
# #plt.plot(x, y_fkl, color='g',  marker='+',  label='VRFL')
# plt.plot(x, y_hessian_30_s, color='y',  marker='*',  label='Unlearning INFOCOM22',linewidth=3, markersize=8)


# plt.grid()
leg = plt.legend(fancybox=True, shadow=True)
plt.xlabel('Epoch' ,fontsize=20)
plt.ylabel('Accuracy (%)' ,fontsize=20)
my_y_ticks = np.arange(80 ,105,5)
plt.yticks(my_y_ticks,fontsize=20)
plt.xticks(x,fontsize=20)
# plt.title('CIFAR10 IID')
plt.legend(loc='best',fontsize=20)
plt.tight_layout()
#plt.title("Fashion MNIST")
plt.rcParams['figure.figsize'] = (2.0, 1)
plt.rcParams['image.interpolation'] = 'nearest'
plt.rcParams['figure.subplot.left'] = 0.11
plt.rcParams['figure.subplot.bottom'] = 0.08
plt.rcParams['figure.subplot.right'] = 0.977
plt.rcParams['figure.subplot.top'] = 0.969
plt.savefig('mnist_client_detail_acc.png', dpi=200)
plt.show()