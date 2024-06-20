import numpy as np
import matplotlib.pyplot as plt

# 定义Sigmoid函数
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# 创建一个包含广泛值的数组，用于绘制图形
x = np.linspace(-10, 10, 1000)
# 生成从-10到10的1000个点

# 计算Sigmoid函数在这些点上的值
y = sigmoid(x)

# 绘制Sigmoid曲线
plt.plot(x, y)

# 添加标题和轴标签
plt.title('Sigmoid Function')
plt.xlabel('Input Value')
plt.ylabel('Output Value')

# 显示网格
plt.grid(True)

# 显示图形
plt.show()