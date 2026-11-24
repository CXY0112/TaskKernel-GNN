[English](SOURCE.md)

# D2-D5a 解剖网格来源

- 数据集：**A statistical shape model of the human heart: biventricular model v2**
- 作者：Steffen Schuler、Axel Loewe
- 数据记录：<https://zenodo.org/records/4506463>
- 输入文件：`heart_sur.vtp`
- 下载地址：<https://zenodo.org/records/4506463/files/heart_sur.vtp?download=1>
- MD5：`11069c64928db9faf60d34ffe916b73d`
- 许可证：CC BY 4.0

原始数据是一个闭合的平均双心室表面，点标签包括基底、心外膜、左心室心内膜、右心室心内膜和心尖。D2-D5a 使用 Gmsh 4.15.2 对同一个表面生成多个分辨率的网格，从而在 Mesh-OOD 评估中固定解剖结构，只改变离散分辨率。

由于该表面数据没有提供受试者特异的心肌纤维，本实验采用有明确记录的规则法跨壁螺旋纤维场：纤维角从最近心内膜处的 (+60^circ) 旋转到心外膜处的 (-60^circ)。因此该实验是连接解剖几何与算法验证的中间阶段，不属于临床级纤维重建。

