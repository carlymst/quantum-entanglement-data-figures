from matplotlib import pyplot as plt
import numpy as np
from matplotlib.ticker import (MultipleLocator, AutoMinorLocator)

SMALL_SIZE = 12
MEDIUM_SIZE = 15

plt.rc('font', size=SMALL_SIZE)  # controls default text sizes
plt.rc('axes', titlesize=SMALL_SIZE)  # font size of the axes title
plt.rc('axes', labelsize=MEDIUM_SIZE)  # font size of the x and y labels
plt.rc('xtick', labelsize=SMALL_SIZE)  # font size of the tick labels
plt.rc('ytick', labelsize=SMALL_SIZE)  # font size of the tick labels
plt.rc('legend', fontsize=SMALL_SIZE)  # legend font size

plt.title('90$^\circ$ (Maximum)', size=18)
plt.xlabel('Energy Level (keV)')
plt.ylabel('Coincidence Counts')

plt.plot(
    [0, 3.2958984375, 6.591796875, 9.8876953125, 13.18359375, 16.4794921875, 19.775390625, 23.0712890625, 26.3671875,
     29.6630859375, 32.958984375, 36.2548828125, 39.55078125, 42.8466796875, 46.142578125, 49.4384765625, 52.734375,
     56.0302734375, 59.326171875, 62.6220703125, 65.91796875, 69.2138671875, 72.509765625, 75.8056640625, 79.1015625,
     82.3974609375, 85.693359375, 88.9892578125, 92.28515625, 95.5810546875, 98.876953125, 102.1728515625, 105.46875,
     108.7646484375, 112.060546875, 115.3564453125, 118.65234375, 121.9482421875, 125.244140625, 128.5400390625,
     131.8359375, 135.1318359375, 138.427734375, 141.7236328125, 145.01953125, 148.3154296875, 151.611328125,
     154.9072265625, 158.203125, 161.4990234375, 164.794921875, 168.0908203125, 171.38671875, 174.6826171875,
     177.978515625, 181.2744140625, 184.5703125, 187.8662109375, 191.162109375, 194.4580078125, 197.75390625,
     201.0498046875, 204.345703125, 207.6416015625, 210.9375, 214.2333984375, 217.529296875, 220.8251953125,
     224.12109375, 227.4169921875, 230.712890625, 234.0087890625, 237.3046875, 240.6005859375, 243.896484375,
     247.1923828125, 250.48828125, 253.7841796875, 257.080078125, 260.3759765625, 263.671875, 266.9677734375,
     270.263671875, 273.5595703125, 276.85546875, 280.1513671875, 283.447265625, 286.7431640625, 290.0390625,
     293.3349609375, 296.630859375, 299.9267578125, 303.22265625, 306.5185546875, 309.814453125, 313.1103515625,
     316.40625, 319.7021484375, 322.998046875, 326.2939453125, 329.58984375, 332.8857421875, 336.181640625,
     339.4775390625, 342.7734375, 346.0693359375, 349.365234375, 352.6611328125, 355.95703125, 359.2529296875,
     362.548828125, 365.8447265625, 369.140625, 372.4365234375, 375.732421875, 379.0283203125, 382.32421875,
     385.6201171875, 388.916015625, 392.2119140625, 395.5078125, 398.8037109375, 402.099609375, 405.3955078125,
     408.69140625, 411.9873046875, 415.283203125, 418.5791015625, 421.875, 425.1708984375, 428.466796875,
     431.7626953125, 435.05859375, 438.3544921875, 441.650390625, 444.9462890625, 448.2421875, 451.5380859375,
     454.833984375, 458.1298828125, 461.42578125, 464.7216796875, 468.017578125, 471.3134765625, 474.609375,
     477.9052734375, 481.201171875, 484.4970703125, 487.79296875, 491.0888671875, 494.384765625, 497.6806640625,
     500.9765625, 504.2724609375, 507.568359375, 510.8642578125, 514.16015625, 517.4560546875, 520.751953125,
     524.0478515625, 527.34375, 530.6396484375, 533.935546875, 537.2314453125, 540.52734375, 543.8232421875,
     547.119140625, 550.4150390625, 553.7109375, 557.0068359375, 560.302734375, 563.5986328125, 566.89453125,
     570.1904296875, 573.486328125, 576.7822265625, 580.078125, 583.3740234375, 586.669921875, 589.9658203125,
     593.26171875, 596.5576171875, 599.853515625, 603.1494140625, 606.4453125, 609.7412109375, 613.037109375,
     616.3330078125, 619.62890625, 622.9248046875, 626.220703125, 629.5166015625, 632.8125, 636.1083984375,
     639.404296875, 642.7001953125, 645.99609375, 649.2919921875, 652.587890625, 655.8837890625, 659.1796875,
     662.4755859375, 665.771484375, 669.0673828125, 672.36328125, 675.6591796875, 678.955078125, 682.2509765625,
     685.546875, 688.8427734375, 692.138671875, 695.4345703125, 698.73046875, 702.0263671875, 705.322265625,
     708.6181640625, 711.9140625, 715.2099609375, 718.505859375, 721.8017578125, 725.09765625, 728.3935546875,
     731.689453125, 734.9853515625, 738.28125, 741.5771484375, 744.873046875, 748.1689453125, 751.46484375,
     754.7607421875, 758.056640625, 761.3525390625, 764.6484375, 767.9443359375, 771.240234375, 774.5361328125,
     777.83203125, 781.1279296875, 784.423828125, 787.7197265625, 791.015625, 794.3115234375, 797.607421875,
     800.9033203125, 804.19921875, 807.4951171875, 810.791015625, 814.0869140625, 817.3828125, 820.6787109375,
     823.974609375, 827.2705078125, 830.56640625, 833.8623046875, 837.158203125, 840.4541015625, 843.75, 847.0458984375,
     850.341796875, 853.6376953125, 856.93359375, 860.2294921875, 863.525390625, 866.8212890625, 870.1171875,
     873.4130859375, 876.708984375, 880.0048828125, 883.30078125, 886.5966796875, 889.892578125, 893.1884765625,
     896.484375, 899.7802734375, 903.076171875, 906.3720703125, 909.66796875, 912.9638671875, 916.259765625,
     919.5556640625, 922.8515625, 926.1474609375, 929.443359375, 932.7392578125, 936.03515625, 939.3310546875,
     942.626953125, 945.9228515625, 949.21875, 952.5146484375, 955.810546875, 959.1064453125, 962.40234375,
     965.6982421875, 968.994140625, 972.2900390625, 975.5859375, 978.8818359375, 982.177734375, 985.4736328125,
     988.76953125, 992.0654296875, 995.361328125, 998.6572265625, 1001.953125, 1005.2490234375, 1008.544921875,
     1011.8408203125, 1015.13671875, 1018.4326171875, 1021.728515625, 1025.0244140625, 1028.3203125, 1031.6162109375,
     1034.912109375, 1038.2080078125, 1041.50390625, 1044.7998046875, 1048.095703125, 1051.3916015625, 1054.6875,
     1057.9833984375, 1061.279296875, 1064.5751953125, 1067.87109375, 1071.1669921875, 1074.462890625, 1077.7587890625,
     1081.0546875, 1084.3505859375, 1087.646484375, 1090.9423828125, 1094.23828125, 1097.5341796875, 1100.830078125,
     1104.1259765625, 1107.421875, 1110.7177734375, 1114.013671875, 1117.3095703125, 1120.60546875, 1123.9013671875,
     1127.197265625, 1130.4931640625, 1133.7890625, 1137.0849609375, 1140.380859375, 1143.6767578125, 1146.97265625,
     1150.2685546875, 1153.564453125, 1156.8603515625, 1160.15625, 1163.4521484375, 1166.748046875, 1170.0439453125,
     1173.33984375, 1176.6357421875, 1179.931640625, 1183.2275390625, 1186.5234375, 1189.8193359375, 1193.115234375,
     1196.4111328125, 1199.70703125, 1203.0029296875, 1206.298828125, 1209.5947265625, 1212.890625, 1216.1865234375,
     1219.482421875, 1222.7783203125, 1226.07421875, 1229.3701171875, 1232.666015625, 1235.9619140625, 1239.2578125,
     1242.5537109375, 1245.849609375, 1249.1455078125, 1252.44140625, 1255.7373046875, 1259.033203125, 1262.3291015625,
     1265.625, 1268.9208984375, 1272.216796875, 1275.5126953125, 1278.80859375, 1282.1044921875, 1285.400390625,
     1288.6962890625, 1291.9921875, 1295.2880859375, 1298.583984375, 1301.8798828125, 1305.17578125, 1308.4716796875,
     1311.767578125, 1315.0634765625, 1318.359375, 1321.6552734375, 1324.951171875, 1328.2470703125, 1331.54296875,
     1334.8388671875, 1338.134765625, 1341.4306640625, 1344.7265625, 1348.0224609375, 1351.318359375, 1354.6142578125,
     1357.91015625, 1361.2060546875, 1364.501953125, 1367.7978515625, 1371.09375, 1374.3896484375, 1377.685546875,
     1380.9814453125, 1384.27734375, 1387.5732421875, 1390.869140625, 1394.1650390625, 1397.4609375, 1400.7568359375],
    [0, 0, 0, 0, 0, 0, 0, 0, 3, 62, 284, 118, 110, 115, 122, 123, 117, 110, 120, 126, 143, 140, 148, 151, 149, 142, 149,
     142, 134, 134, 142, 165, 141, 148, 164, 144, 161, 161, 187, 182, 157, 181, 167, 195, 160, 191, 184, 194, 196, 219,
     210, 257, 228, 250, 249, 250, 310, 277, 320, 314, 324, 337, 338, 366, 344, 365, 420, 378, 363, 396, 395, 385, 360,
     343, 357, 370, 370, 365, 356, 361, 335, 357, 384, 412, 409, 390, 378, 394, 419, 410, 401, 406, 455, 403, 428, 418,
     401, 445, 405, 445, 409, 405, 409, 372, 400, 401, 367, 393, 399, 417, 374, 364, 350, 380, 395, 373, 355, 344, 346,
     332, 366, 367, 366, 379, 361, 374, 363, 365, 315, 313, 352, 364, 326, 368, 363, 337, 376, 344, 349, 385, 349, 337,
     365, 371, 340, 398, 334, 369, 344, 355, 352, 349, 354, 361, 355, 371, 374, 331, 352, 346, 355, 336, 333, 339, 373,
     356, 323, 344, 358, 350, 357, 365, 383, 339, 342, 363, 355, 326, 350, 371, 314, 328, 365, 333, 356, 371, 322, 321,
     343, 320, 304, 316, 334, 303, 317, 322, 309, 310, 332, 311, 302, 305, 329, 301, 304, 309, 285, 279, 300, 328, 300,
     283, 311, 316, 289, 288, 304, 289, 290, 309, 299, 289, 250, 269, 285, 285, 285, 262, 270, 303, 286, 266, 257, 290,
     253, 258, 250, 326, 266, 269, 238, 234, 227, 234, 264, 234, 244, 233, 240, 239, 234, 221, 200, 227, 238, 218, 219,
     220, 212, 199, 194, 242, 217, 195, 224, 197, 198, 196, 216, 210, 196, 218, 184, 192, 184, 232, 209, 209, 187, 183,
     199, 175, 174, 186, 199, 148, 194, 182, 195, 186, 193, 184, 185, 137, 184, 170, 176, 177, 176, 157, 148, 162, 157,
     164, 148, 148, 163, 160, 142, 119, 165, 167, 133, 133, 147, 137, 168, 154, 147, 158, 143, 159, 151, 161, 149, 153,
     150, 158, 143, 146, 148, 156, 145, 153, 139, 125, 141, 162, 144, 122, 149, 152, 138, 143, 161, 135, 149, 147, 166,
     151, 196, 162, 187, 164, 186, 204, 197, 229, 241, 241, 255, 274, 289, 287, 296, 307, 342, 333, 367, 343, 403, 371,
     399, 395, 409, 413, 442, 398, 487, 492, 433, 429, 426, 404, 412, 402, 427, 391, 321, 341, 345, 315, 269, 254, 244,
     246, 240, 202, 175, 148, 139, 127, 134, 128, 100, 69, 83, 67, 64, 77, 41, 52, 52, 58, 39, 35, 68, 32, 46, 34, 50,
     35, 48, 45, 33, 47],
    color="blue", label="Total"
)

plt.plot(
    [0, 3.2958984375, 6.591796875, 9.8876953125, 13.18359375, 16.4794921875, 19.775390625, 23.0712890625, 26.3671875,
     29.6630859375, 32.958984375, 36.2548828125, 39.55078125, 42.8466796875, 46.142578125, 49.4384765625, 52.734375,
     56.0302734375, 59.326171875, 62.6220703125, 65.91796875, 69.2138671875, 72.509765625, 75.8056640625, 79.1015625,
     82.3974609375, 85.693359375, 88.9892578125, 92.28515625, 95.5810546875, 98.876953125, 102.1728515625, 105.46875,
     108.7646484375, 112.060546875, 115.3564453125, 118.65234375, 121.9482421875, 125.244140625, 128.5400390625,
     131.8359375, 135.1318359375, 138.427734375, 141.7236328125, 145.01953125, 148.3154296875, 151.611328125,
     154.9072265625, 158.203125, 161.4990234375, 164.794921875, 168.0908203125, 171.38671875, 174.6826171875,
     177.978515625, 181.2744140625, 184.5703125, 187.8662109375, 191.162109375, 194.4580078125, 197.75390625,
     201.0498046875, 204.345703125, 207.6416015625, 210.9375, 214.2333984375, 217.529296875, 220.8251953125,
     224.12109375, 227.4169921875, 230.712890625, 234.0087890625, 237.3046875, 240.6005859375, 243.896484375,
     247.1923828125, 250.48828125, 253.7841796875, 257.080078125, 260.3759765625, 263.671875, 266.9677734375,
     270.263671875, 273.5595703125, 276.85546875, 280.1513671875, 283.447265625, 286.7431640625, 290.0390625,
     293.3349609375, 296.630859375, 299.9267578125, 303.22265625, 306.5185546875, 309.814453125, 313.1103515625,
     316.40625, 319.7021484375, 322.998046875, 326.2939453125, 329.58984375, 332.8857421875, 336.181640625,
     339.4775390625, 342.7734375, 346.0693359375, 349.365234375, 352.6611328125, 355.95703125, 359.2529296875,
     362.548828125, 365.8447265625, 369.140625, 372.4365234375, 375.732421875, 379.0283203125, 382.32421875,
     385.6201171875, 388.916015625, 392.2119140625, 395.5078125, 398.8037109375, 402.099609375, 405.3955078125,
     408.69140625, 411.9873046875, 415.283203125, 418.5791015625, 421.875, 425.1708984375, 428.466796875,
     431.7626953125, 435.05859375, 438.3544921875, 441.650390625, 444.9462890625, 448.2421875, 451.5380859375,
     454.833984375, 458.1298828125, 461.42578125, 464.7216796875, 468.017578125, 471.3134765625, 474.609375,
     477.9052734375, 481.201171875, 484.4970703125, 487.79296875, 491.0888671875, 494.384765625, 497.6806640625,
     500.9765625, 504.2724609375, 507.568359375, 510.8642578125, 514.16015625, 517.4560546875, 520.751953125,
     524.0478515625, 527.34375, 530.6396484375, 533.935546875, 537.2314453125, 540.52734375, 543.8232421875,
     547.119140625, 550.4150390625, 553.7109375, 557.0068359375, 560.302734375, 563.5986328125, 566.89453125,
     570.1904296875, 573.486328125, 576.7822265625, 580.078125, 583.3740234375, 586.669921875, 589.9658203125,
     593.26171875, 596.5576171875, 599.853515625, 603.1494140625, 606.4453125, 609.7412109375, 613.037109375,
     616.3330078125, 619.62890625, 622.9248046875, 626.220703125, 629.5166015625, 632.8125, 636.1083984375,
     639.404296875, 642.7001953125, 645.99609375, 649.2919921875, 652.587890625, 655.8837890625, 659.1796875,
     662.4755859375, 665.771484375, 669.0673828125, 672.36328125, 675.6591796875, 678.955078125, 682.2509765625,
     685.546875, 688.8427734375, 692.138671875, 695.4345703125, 698.73046875, 702.0263671875, 705.322265625,
     708.6181640625, 711.9140625, 715.2099609375, 718.505859375, 721.8017578125, 725.09765625, 728.3935546875,
     731.689453125, 734.9853515625, 738.28125, 741.5771484375, 744.873046875, 748.1689453125, 751.46484375,
     754.7607421875, 758.056640625, 761.3525390625, 764.6484375, 767.9443359375, 771.240234375, 774.5361328125,
     777.83203125, 781.1279296875, 784.423828125, 787.7197265625, 791.015625, 794.3115234375, 797.607421875,
     800.9033203125, 804.19921875, 807.4951171875, 810.791015625, 814.0869140625, 817.3828125, 820.6787109375,
     823.974609375, 827.2705078125, 830.56640625, 833.8623046875, 837.158203125, 840.4541015625, 843.75, 847.0458984375,
     850.341796875, 853.6376953125, 856.93359375, 860.2294921875, 863.525390625, 866.8212890625, 870.1171875,
     873.4130859375, 876.708984375, 880.0048828125, 883.30078125, 886.5966796875, 889.892578125, 893.1884765625,
     896.484375, 899.7802734375, 903.076171875, 906.3720703125, 909.66796875, 912.9638671875, 916.259765625,
     919.5556640625, 922.8515625, 926.1474609375, 929.443359375, 932.7392578125, 936.03515625, 939.3310546875,
     942.626953125, 945.9228515625, 949.21875, 952.5146484375, 955.810546875, 959.1064453125, 962.40234375,
     965.6982421875, 968.994140625, 972.2900390625, 975.5859375, 978.8818359375, 982.177734375, 985.4736328125,
     988.76953125, 992.0654296875, 995.361328125, 998.6572265625, 1001.953125, 1005.2490234375, 1008.544921875,
     1011.8408203125, 1015.13671875, 1018.4326171875, 1021.728515625, 1025.0244140625, 1028.3203125, 1031.6162109375,
     1034.912109375, 1038.2080078125, 1041.50390625, 1044.7998046875, 1048.095703125, 1051.3916015625, 1054.6875,
     1057.9833984375, 1061.279296875, 1064.5751953125, 1067.87109375, 1071.1669921875, 1074.462890625, 1077.7587890625,
     1081.0546875, 1084.3505859375, 1087.646484375, 1090.9423828125, 1094.23828125, 1097.5341796875, 1100.830078125,
     1104.1259765625, 1107.421875, 1110.7177734375, 1114.013671875, 1117.3095703125, 1120.60546875, 1123.9013671875,
     1127.197265625, 1130.4931640625, 1133.7890625, 1137.0849609375, 1140.380859375, 1143.6767578125, 1146.97265625,
     1150.2685546875, 1153.564453125, 1156.8603515625, 1160.15625, 1163.4521484375, 1166.748046875, 1170.0439453125,
     1173.33984375, 1176.6357421875, 1179.931640625, 1183.2275390625, 1186.5234375, 1189.8193359375, 1193.115234375,
     1196.4111328125, 1199.70703125, 1203.0029296875, 1206.298828125, 1209.5947265625, 1212.890625, 1216.1865234375,
     1219.482421875, 1222.7783203125, 1226.07421875, 1229.3701171875, 1232.666015625, 1235.9619140625, 1239.2578125,
     1242.5537109375, 1245.849609375, 1249.1455078125, 1252.44140625, 1255.7373046875, 1259.033203125, 1262.3291015625,
     1265.625, 1268.9208984375, 1272.216796875, 1275.5126953125, 1278.80859375, 1282.1044921875, 1285.400390625,
     1288.6962890625, 1291.9921875, 1295.2880859375, 1298.583984375, 1301.8798828125, 1305.17578125, 1308.4716796875,
     1311.767578125, 1315.0634765625, 1318.359375, 1321.6552734375, 1324.951171875, 1328.2470703125, 1331.54296875,
     1334.8388671875, 1338.134765625, 1341.4306640625, 1344.7265625, 1348.0224609375, 1351.318359375, 1354.6142578125,
     1357.91015625, 1361.2060546875, 1364.501953125, 1367.7978515625, 1371.09375, 1374.3896484375, 1377.685546875,
     1380.9814453125, 1384.27734375, 1387.5732421875, 1390.869140625, 1394.1650390625, 1397.4609375, 1400.7568359375],
    [0, 0, 0, 0, 0, 0, 0, 0, 2, 7, 31, 10, 10, 13, 8, 13, 9, 17, 10, 14, 17, 22, 11, 10, 20, 23, 29, 27, 19, 11, 17, 17,
     18, 22, 15, 14, 26, 22, 15, 22, 20, 28, 32, 18, 34, 32, 32, 35, 38, 47, 29, 42, 29, 37, 49, 39, 43, 51, 48, 60, 49,
     43, 45, 61, 66, 46, 57, 67, 66, 53, 46, 53, 63, 58, 76, 57, 45, 66, 66, 46, 61, 54, 76, 60, 61, 54, 63, 54, 55, 46,
     71, 52, 53, 56, 46, 61, 60, 63, 66, 46, 50, 70, 39, 70, 41, 67, 57, 60, 55, 54, 49, 49, 57, 49, 50, 49, 44, 54, 61,
     46, 39, 39, 55, 39, 42, 52, 51, 43, 47, 52, 44, 37, 44, 44, 48, 38, 36, 34, 40, 44, 43, 55, 45, 42, 35, 42, 49, 39,
     52, 50, 28, 46, 32, 39, 51, 28, 45, 34, 32, 33, 35, 32, 27, 28, 20, 32, 29, 29, 30, 35, 30, 31, 30, 41, 37, 32, 36,
     37, 30, 25, 28, 31, 24, 28, 35, 33, 35, 41, 26, 27, 19, 25, 33, 22, 20, 20, 25, 22, 26, 23, 18, 27, 17, 25, 19, 20,
     23, 24, 19, 24, 11, 15, 19, 21, 36, 24, 19, 22, 14, 22, 21, 20, 19, 20, 20, 21, 13, 13, 21, 18, 23, 28, 14, 12, 20,
     13, 18, 12, 15, 10, 16, 10, 19, 15, 19, 18, 13, 16, 20, 16, 16, 14, 17, 17, 14, 20, 20, 13, 18, 22, 13, 10, 21, 12,
     19, 26, 12, 15, 17, 12, 15, 14, 18, 14, 14, 9, 12, 11, 16, 16, 7, 9, 13, 6, 16, 12, 11, 11, 6, 13, 10, 12, 14, 18,
     11, 9, 12, 14, 5, 6, 11, 10, 10, 10, 16, 11, 11, 12, 10, 10, 5, 10, 8, 5, 19, 5, 6, 8, 9, 11, 7, 6, 8, 8, 9, 4, 7,
     7, 7, 11, 9, 11, 14, 10, 5, 6, 10, 5, 8, 10, 5, 8, 5, 8, 7, 11, 11, 7, 4, 10, 5, 8, 5, 6, 10, 9, 6, 7, 10, 15, 7,
     12, 3, 7, 7, 7, 10, 4, 7, 9, 7, 12, 9, 12, 7, 9, 3, 8, 9, 6, 7, 5, 8, 3, 11, 11, 7, 6, 7, 4, 7, 11, 10, 5, 6, 9,
     10, 4, 9, 8, 11, 7, 8, 6, 5, 2, 4, 5, 5, 6, 8, 6, 5, 2, 6, 3, 6, 8, 5, 3, 5, 5, 11, 7, 8, 8],
    color="orange", label="Background"
)

leg = plt.legend(loc='upper center')

plt.minorticks_on()
plt.tick_params(axis="both", which="major", direction="in", width=2, length=10)
plt.tick_params(axis="both", which="minor", direction="in", width=2, length=5)

plt.xlim([-55, 1400])
plt.ylim([-30, 500])

plt.savefig('raw data 90.png', dpi=1200)
plt.show()