
import torch

def fix(net,boundary:float):
    revsion(net,boundary)



# 这里改一个relu，就会改掉所有使用这个relu的层。
def revsion(net,boundary:float):
    r = vars(net)
    dic = r['_modules']
    for k ,value in dic.items():
        if "block" in k:
            revsion(net.__getattr__(k))
        if value.__str__() in ["ReLU()","ReLU6()","LeakyReLU()","RReLU()"]:
            net.__setattr__(k,create_relu(boundary))


def create_relu(max_value):
    def ownrelu(x):
        return min(torch.nn.ReLU(x),max_value)
    return ownrelu



def create_chrelu(values):
    def chrelu(x):
        return relu(x,max_value = values)
    return chrelu
def relu(x, alpha=0.0, max_value=None, threshold=0.0):
    dtype = getattr(x, "dtype", K.floatx())
    if alpha != 0.0:
        if max_value is None and threshold == 0:
            return torch.nn.LeakyReLU(x, alpha=alpha)

        if threshold != 0:
            negative_part = torch.nn.ReLU(-x + threshold)
        else:
            negative_part = torch.nn.ReLU(-x)

    clip_max = max_value is not None

    if threshold != 0:
        x = x * tf.cast(tf.greater(x, threshold), dtype=dtype)
    elif max_value == 6:
        x = torch.nn.ReLU6(x)
        clip_max = False
    else:
        x = torch.nn.ReLU(x)

    if clip_max:
        
        zero = torch.as_tensor(torch.zeros(x.shape[-1]),dtype = x.dtype.base_dtype)
        cmpval = tf.Variable(zero)
        sess.run(tf.global_variables_initializer())
        val = sess.run(cmpval)
        for i in range(0,x.shape[-1]):
            val[i] = max_value[i] # val[i]就是该层第i个卷积核对应relu的上界。
        value = torch.as_tensor(tf.assign(cmpval,val)) # 利用广播机制得到上界和下界0
        x = tf.clip_by_value(x,zero,value) # 映射x
    if alpha != 0.0:
        alpha = torch.as_tensor(alpha, x.dtype.base_dtype)
        x -= alpha * negative_part
    return x