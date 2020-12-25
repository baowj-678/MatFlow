""" MatFlow 的数据类型
@Author: Bao Wenjie
@Email: bwj_678@qq.com
@github: https://github.com/baowj-678
@latest-change 2020/12/26
"""

import numpy as np
class float(np.float):
    def __init__(self) -> None:
        super().__init__()

class float32(np.float):
    def __init__(self) -> None:
        super().__init__()

class float64(np.float64):
    def __init__(self) -> None:
        super().__init__()

class double(np.float64):
    def __init__(self) -> None:
        super().__init__()

class int(np.int32):
    def __init__(self) -> None:
        super().__init__()

class int32(np.int32):
    def __init__(self) -> None:
        super().__init__()

class int64(np.int64):
    def __init__(self) -> None:
        super().__init__()

class long(np.int64):
    def __init__(self) -> None:
        super().__init__()


