import numpy as np
from numpy.testing._private.utils import raises




class Tensor:
    def __init__(self, data=None, *, requires_grad: bool=False) -> None:
        self.data = None
        if isinstance(data, (int, float, bool)):
            self.data = [data]
        elif isinstance(data, list):
            self.data = np.array(data)
        elif isinstance(data, (np.ndarray, np.float, np.float32, np.float64)):
            self.data = data.copy()
        elif isinstance(data, Tensor):
            self.data = data
        else:
            raise ValueError("输入的是什么玩意？")
        self.requires_grad = requires_grad
        self.grad = None
    
    def __str__(self) -> str:
        return self.__repr__()
    
    def __repr__(self) -> str:
        return "Tensor with shape: {}\n".format(self.shape)
    
    def __len__(self):
        return len(self.data)
    
    def __contains__(self, item):
        return self.data.__contains__(item)
    

##################################
    def __getitem__(self, item):
        # return self.data
        print(item)
##################################

    def __setitem__(self, key, value):
        print(key, value)
    
    @property
    def shape(self):
        """ 
        返回Tensor的shape

        @Returns
        --------
        Tuple[int, ...]

        @Examples
        ---------
        """
        return self.data.shape
    
    @property
    def dtype(self):
        """
        返回当前Tensor的数据类型

        @Returns
        --------
        Tensor dtype object

        @Examples
        ---------
        """
        return self.data.dtype
    
    @property
    def T(self):
        """
        返回Tensor的转置

        @Returns
        --------
        Tensor

        @Examples
        ---------
        """
        return self.data
        # 还未实现