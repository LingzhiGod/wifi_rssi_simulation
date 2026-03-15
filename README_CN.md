# WiFi RSSI 三维室内高保真仿真（MATLAB）

本仓库提供一个面向室内定位实验的高保真 WiFi RSSI 仿真器，核心脚本为 `wifi_rssi_simulation.m`。

它支持：
- 可配置三维房间几何
- 带材料属性的墙体/家具障碍物
- 多 AP 发射与信道重叠干扰
- LOS + 多径反射（镜像源简化射线追踪）
- 阴影衰落与小尺度衰落（Rayleigh/Rician）
- 人体动态衰减（轨迹移动）
- 结构化数据导出（MAT/CSV）
- 三维场景、热力图与动态动画可视化

## 1. 仓库结构

- `wifi_rssi_simulation.m`：主脚本（含模块化本地函数）
- `output/wifi_rssi_results.mat`：仿真结束后生成
- `output/wifi_rssi_results.csv`：仿真结束后生成

## 2. 快速开始

## 2.1 环境要求

- MATLAB R2020b 或更高版本（推荐）
- 当前实现不依赖额外工具箱

## 2.2 运行方式

在 MATLAB 命令行执行：

```matlab
run('wifi_rssi_simulation.m')
```

运行完成后会得到：
- 完整结构化结果 `.mat`
- 展平后的 `.csv`
- 3D 场景图、2D RSSI 切片热力图、3D RSSI 分布图
- 可选动态动画

## 3. 内置示例场景

脚本默认演示场景满足你的需求：
- 房间：`10m x 10m x 3m`
- AP：3 个（天花板附近布置）
- 障碍物：2 个（隔断墙 + 柜体）
- 人体：1 个（圆柱体模型 + 路径点动态移动）
- 网格：3D 测量点网格（步长可配置）
- 时间：按 `dt` 离散动态更新

## 4. 模型模块设计

脚本采用模块化函数组织，主要包括：

- `defaultSimulationConfig`：仿真总参数
- `createExampleScenario`：场景构建（房间/AP/障碍/人体）
- `createMeasurementGrid`：测量网格生成
- `runRssiSimulation`：主循环
- `computeReceivedPowerDbm`：单链路接收功率计算
- `buildImageSources` / `reflectedComponents`：多径反射
- `computeObstacleLossDb`：障碍物穿透衰减
- `computeHumanLossDb`：人体衰减
- `channelOverlapFactor`：重叠信道干扰权重
- `smallScaleFadingGain`：小尺度衰落
- `exportSimulationData`：数据导出
- 可视化函数：场景绘制、切片热力图、3D 散点、动画

## 5. 详细算法说明

## 5.1 几何与材料建模

使用三维笛卡尔坐标系：
- 房间边界：6 个轴对齐平面（`x=0/x=L/y=0/y=W/z=0/z=H`）
- 障碍物：轴对齐包围盒（AABB）
- 人体：有限高垂直圆柱体

材料参数：
- `reflectionCoeff`：反射系数（幅度）
- `absorptionCoeff`：吸收系数
- `attenuationDbPerMeter`：介质每米衰减（dB/m）

## 5.2 LOS 基础路径损耗

设 AP 与接收点距离为 `d`。

1) 波长

\[
\lambda = \frac{c}{f}
\]

2) 参考距离 `d0=1m` 处 FSPL

\[
\text{FSPL}_{1m}(dB)=20\log_{10}\left(\frac{4\pi d_0}{\lambda}\right)
\]

3) 对数距离模型

\[
PL(dB)=\text{FSPL}_{1m}+10n\log_{10}\left(\frac{d}{d_0}\right)
\]

其中 `n` 为路径损耗指数（可调）。

LOS 功率（尚未与多径相干叠加）为：

\[
P_{r,LOS}(dBm)=P_t-PL-L_{obs}-L_{human}-L_{NLOS}+X_\sigma
\]

- `L_obs`：障碍物损耗
- `L_human`：人体损耗
- `L_NLOS`：遮挡时的附加惩罚
- `X_sigma ~ N(0, sigma^2)`：对数正态阴影衰落（dB 域）

## 5.3 障碍物穿透衰减

对每个障碍物，使用 slab 相交算法计算 AP-接收点线段在盒体内的长度 `\ell`：

\[
L_{obs,i}(dB)=\ell\cdot\alpha_i+1.5\ell\cdot a_i
\]

- `alpha_i = attenuationDbPerMeter`
- `a_i = absorptionCoeff`

总障碍损耗为所有相交障碍项之和。

## 5.4 人体衰减

人体按有限高圆柱建模，计算链路线段在圆柱内长度 `\ell_j`：

\[
L_{human,j}(dB)=\ell_j\cdot\beta_j
\]

- `beta_j = attenuationDbPerMeter`

所有人体项求和得到 `L_human`。

## 5.5 NLOS 与衍射简化惩罚

若 LOS 被障碍物或人体阻挡，附加：

\[
L_{NLOS}=L_{nlosPenalty}+L_{diffractionPenalty}
\]

这是工程近似，用于快速体现遮挡与衍射造成的额外损耗。

## 5.6 多径反射（镜像源法）

开启后，脚本将 AP 对房间边界平面进行镜像，生成最高 `R` 阶反射路径。

对第 `k` 条反射路径：
- 路径长度 `d_k`
- 反射幅度增益

\[
g_k=\prod_{r\in path_k}\left(\rho_r(1-a_r)\right)
\]

其中 `rho=reflectionCoeff`，`a=absorptionCoeff`。

对应功率（dBm）近似：

\[
P_{r,k}(dBm)=P_t-PL(d_k)-L_{refl,k}-L_{obs,k}-L_{order,k}
\]

\[
L_{refl,k}=-20\log_{10}(g_k)
\]

总场采用复数相干叠加：

\[
E_{tot}(f)=\sqrt{P_{r,LOS}^{lin}}e^{-j2\pi f d_{LOS}/c}+\sum_k\sqrt{P_{r,k}^{lin}}e^{-j2\pi f d_k/c}
\]
\[
P_{tot}^{lin}(f)=|E_{tot}(f)|^2
\]

## 5.7 频率选择性衰落（可选）

若开启，则在带宽内取多个子载波频点 `f_m` 计算并平均：

\[
P_{avg}^{lin}=\frac{1}{N_{sc}}\sum_{m=1}^{N_{sc}}P_{tot}^{lin}(f_m)
\]

用于近似 OFDM 系统的频率选择性效应。

## 5.8 小尺度衰落

对 `P_avg` 再乘以随机衰落增益 `G`：

- Rayleigh：`h ~ CN(0,1)`，`G=|h|^2`
- Rician：
  - `K=10^(K_dB/10)`
  - `h=sqrt(K/(K+1))+sqrt(1/(2(K+1)))*(nI+j*nQ)`
  - `G=|h|^2`

最终线性功率：

\[
P_r^{lin}=P_{avg}^{lin}\cdot G
\]

## 5.9 AP 间干扰与噪声

目标 AP 为 `a`，来自 AP `b` 的干扰按频谱重叠系数缩放：

\[
\eta_{ab}=\max\left(0,1-\frac{|f_a-f_b|}{BW_{shared}}\right),\quad BW_{shared}=\frac{BW_a+BW_b}{2}
\]

总干扰：

\[
I_a^{lin}=\sum_{b\neq a}\eta_{ab}P_{r,b}^{lin}
\]

噪声与背景：

\[
N^{lin}=P_{noise}^{lin}+P_{bg}^{lin}
\]

SINR：

\[
SINR_a=\frac{P_{r,a}^{lin}}{I_a^{lin}+N^{lin}}
\]

脚本还输出等效 RSSI（干扰惩罚形式）：

\[
RSSI_{eff,a}(dBm)=RSSI_a(dBm)-10\log_{10}\left(1+\frac{I_a^{lin}}{P_{r,a}^{lin}}\right)
\]

## 5.10 动态人体轨迹更新

每个时间步：
- 按 waypoint-time 线性插值人体位置（支持循环轨迹）
- 重新计算所有 AP-网格链路的衰减
- 更新 `signal/interference/SINR/effectiveRSSI` 张量

## 6. 主循环流程

在时间 `t`：
- 更新人体状态
- 遍历每个网格点 `p`、每个 AP `a`：
  - 计算 LOS 路损
  - 加入障碍物/人体损耗与 NLOS 惩罚
  - 加入反射多径（若启用）
  - 加入阴影衰落与小尺度衰落
- 计算 AP 间干扰、SINR、等效 RSSI
- 存储结果张量

## 7. 输出数据格式

## 7.1 MAT 文件

`results` 内核心字段：
- `signalDbm`：`[nPoints, nAP, nTimes]`
- `interferenceDbm`：`[nPoints, nAP, nTimes]`
- `sinrDb`：`[nPoints, nAP, nTimes]`
- `effectiveRssiDbm`：`[nPoints, nAP, nTimes]`
- `time`、人体状态、时间戳等

同时保存 `grid`、`aps`、`sim`。

## 7.2 CSV 字段

`time_s, point_id, x_m, y_m, z_m, ap_name, signal_dbm, interference_dbm, sinr_db, effective_rssi_dbm`

每行对应一个 `(time, point, ap)` 观测。

## 8. 可视化能力

- 3D 场景：房间线框 + 障碍物 + AP + 人体轨迹/人体模型
- 2D 切片热力图：指定高度层的 RSSI 分布
- 3D 散点图：全空间 RSSI
- 动画：人体移动与 RSSI 动态演化

## 9. 参数调节指南

在 `defaultSimulationConfig` 中可调：

- 几何/网格
  - `roomDims`, `gridStep`, `gridZLevels`
- 时间
  - `dt`, `simDuration`
- 传播
  - `pathLossExponent`, `nlosPenaltyDb`, `diffractionPenaltyDb`
- 多径
  - `enableRayTracing`, `maxReflectionOrder`, `maxReflectionComponents`
- 频率选择性
  - `enableFrequencySelective`, `signalBandwidthHz`, `numSubcarriers`
- 随机信道
  - `shadowSigmaDb`, `smallScaleModel`, `ricianK_dB`
- 干扰噪声
  - `ambientNoiseDbm`, `backgroundInterferenceDbm`

在 `createExampleScenario` 中可调：
- AP 位置、发射功率、频率、信道、带宽
- 障碍物几何和材料属性
- 人体尺寸、衰减系数、轨迹路径与时间戳

## 10. 复杂度与工程说明

若网格点数 `N_p`、AP 数 `N_a`、时间步数 `N_t`、反射分量数 `N_r`：
- 复杂度近似为 `O(N_t * N_p * N_a * N_r)`，另加干扰计算项。

工程近似说明：
- 本模型是高保真工程仿真，不是全波电磁求解器。
- 衍射与材料机理采用可计算、可调参数近似。
- 当前反射主要来自房间边界平面，可扩展到障碍物表面反射。

## 11. 可扩展方向

建议扩展：
- UTD/刀刃衍射模型
- 入射角相关反射/透射系数
- 天线方向图与极化
- 时间相关信道与多普勒
- OFDM 子载波级 CSI 导出
- 与定位算法联动（KNN/WKNN/粒子滤波/深度学习）

## 12. 复现实验

脚本已设置 `rng(42)`，保证随机衰落结果可复现。
如需不同随机场景，请修改随机种子。

## 13. 许可

本项目采用 **Apache License 2.0** 许可证。

完整条款见 [LICENSE](LICENSE)。
