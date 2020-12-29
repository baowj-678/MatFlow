""" MatFlow 数据定义
@Author: Bao Wenjie
@Email: bwj_678@qq.com
@github: https://github.com/baowj-678
@latest-change 2020/12/26
"""
from typing import Any, Optional
import numpy as np
from numpy.testing._private.utils import raises
import dtype as mfType

class Tensor:
    def __init__(self, data=None, *, requires_grad: bool=False, dtype: Optional[Any]=mfType.float32) -> None:
        self.data = None
        self.dtype = dtype
        if isinstance(data, (int, float, bool)):
            self.data = np.array([data], dtype=dtype)
        elif isinstance(data, list):
            self.data = np.array(data, dtype=dtype)
        elif isinstance(data, (np.ndarray, np.float, np.float32, np.float64)):
            self.data = data.copy().astype(dtype)
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
        return self.dtype
    
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
    
    @property
    def double(self):
        """
        将原数据转成double类型

        @Returns
        --------
        Tensor(dtype=double)
        """
        return self.data.astype(mfType.double)
    
    @property
    def int(self):
        """ 将原数据转成int类型

        @Returns
        --------
        Tensor(dtype=int)
        """
        return self.data.astype(mfType.int)

    @property
    def long(self):
        """ 将原数据转成long类型

        @Returns
        --------
        Tensor(dtype=long)
        """
        return self.data.astype(mfType.long)

    @property
    def float(self):
        """ 将原数据转成float类型

        @Returns
        --------
        Tensor(dtype=float)
        """
        return self.data.astype(mfType.float)

    @property
    def float32(self):
        """ 将原数据转成float32类型

        @Returns
        --------
        Tensor(dtype=float32)
        """
        return self.data.astype(mfType.float32)

    @property
    def float64(self):
        """ 将原数据转成float64类型

        @Returns
        --------
        Tensor(dtype=float64)
        """
        return self.data.astype(mfType.float64)

    @property
    def int32(self):
        """ 将原数据转成int32类型

        @Returns
        --------
        Tensor(dtype=int32)
        """
        return self.data.astype(mfType.int32)

    @property
    def int64(self):
        """ 将原数据转成int64类型

        @Returns
        --------
        Tensor(dtype=int64)
        """
        return self.data.astype(mfType.int64)