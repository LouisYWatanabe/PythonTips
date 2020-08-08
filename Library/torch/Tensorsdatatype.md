# Tensorsのデータ型

データ型 | dtype属性への記述 | 対応するPython／NumPy（np）のデータ型
---------|-----------------|--------------------------------
Boolean（真偽値）|torch.bool|bool／np.bool
8-bitの符号なし整数|torch.uint8|int／np.uint8
8-bitの符号付き整数|torch.int8|int／np.int8
16-bitの符号付き整数|torch.int16 ／ torch.short|int／np.uint16
32-bitの符号付き整数|torch.int32 ／ torch.int|int／np.uint32
64-bitの符号付き整数|torch.int64 ／ torch.long|int／np.uint64
16-bitの浮動小数点|torch.float16 ／ torch.half|float／np.float16
32-bitの浮動小数点|torch.float32 ／ torch.float|float／np.float32
64-bitの浮動小数点|torch.float64 ／ torch.double|float／np.float64

