import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from scipy.signal import find_peaks, correlate
import torch.nn.functional as F
import matplotlib.font_manager as fm

from config import RIGHT_DATA_PATH, LEFT_DATA_PATH, target1, target2, modify_csv_save, modify_result_save, \
    modify_model_save, modify_result_loss_save

font_path = "simhei.ttf"
font_prop = fm.FontProperties(fname=font_path)
fm.fontManager.addfont(font_path)
plt.rcParams['font.sans-serif'] = font_prop.get_name()
plt.rcParams['axes.unicode_minus'] = False

# 固定随机种子，保证结果可复现
np.random.seed(42)
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)

# 设备配置
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# ==============================================
# 1. 数据加载与预处理
# ==============================================

def load_and_filter_keypoints(file_path, target_kpts):
    """加载关键点CSV，筛选目标关键点，按帧排序，去除缺失值"""
    df = pd.read_csv(file_path)
    kpt_dict = {}
    for kpt in target_kpts:
        kpt_data = df[df['kpt_name'] == kpt].sort_values('frame').reset_index(drop=True).dropna(subset=['x', 'y', 'conf'])
        kpt_dict[kpt] = kpt_data
    return kpt_dict

# 加载左脚踝关键点数据
right_kpts = load_and_filter_keypoints(RIGHT_DATA_PATH, [target1])
left_kpts = load_and_filter_keypoints(LEFT_DATA_PATH, [target1])

#  坐标序列 + 置信度序列
right_clean_y = right_kpts[target1][target2].values  # 右目干净坐标（训练用）
right_conf = right_kpts[target1]['conf'].values   # 右目置信度
left_noisy_y = left_kpts[target1][target2].values      # 左目带遮挡坐标（推理用）
left_conf = left_kpts[target1]['conf'].values      # 左目置信度

# 对齐帧长度
total_frames = min(len(right_clean_y), len(left_noisy_y))
right_clean_y = right_clean_y[:total_frames]
right_conf = right_conf[:total_frames]
left_noisy_y = left_noisy_y[:total_frames]
left_conf = left_conf[:total_frames]
left_raw_y = left_noisy_y.copy()  # 保留原始左目数据用于对比
frame = np.arange(total_frames)

print(f"Loaded data - Total frames: {total_frames}")
print(f"Right clean coordinate shape: {right_clean_y.shape}")
print(f"Left noisy coordinate shape: {left_noisy_y.shape}")

# 从右目干净数据提取步态周期
def get_gait_cycle(clean_sequence, min_cycle=60, max_cycle=150):
    """自相关法提取精准步态周期（帧）"""
    seq_norm = (clean_sequence - clean_sequence.mean()) / clean_sequence.std()
    corr = correlate(seq_norm, seq_norm, mode='full')
    corr = corr[len(corr)//2:]
    peaks, _ = find_peaks(corr, height=0.5*corr.max())
    valid_peaks = peaks[(peaks >= min_cycle) & (peaks <= max_cycle)]
    if len(valid_peaks) == 0:
        print("Warning: Auto cycle detection failed, use default 100 frames")
        return 100
    gait_cycle = valid_peaks[0]
    print(f"Detected gait cycle from right data: {gait_cycle} frames")
    return gait_cycle

gait_cycle = get_gait_cycle(right_clean_y)

# 坐标归一化
scaler_right = MinMaxScaler(feature_range=(0, 1))
right_y_scaled = scaler_right.fit_transform(right_clean_y.reshape(-1, 1)).flatten()

scaler_left = MinMaxScaler(feature_range=(0, 1))
left_y_scaled = scaler_left.fit_transform(left_noisy_y.reshape(-1, 1)).flatten()

# ==============================================
# 2. 改进1：多尺度时序特征增强模块（MTFEM）
# ==============================================
class CausalConv1d(nn.Module):
    """
    要求的一维因果卷积：保证t时刻输出仅依赖t及之前的输入，不破坏时序因果性
    """
    def __init__(self, in_channels, out_channels, kernel_size=3, dilation=1):
        super(CausalConv1d, self).__init__()
        # 因果卷积专属padding：仅在序列左侧padding，保证时序因果性
        self.padding = dilation * (kernel_size - 1)
        self.conv = nn.Conv1d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            padding=self.padding,
            dilation=dilation
        )

    def forward(self, x):
        # 输入维度: [batch, seq_len, in_channels]
        x = x.transpose(1, 2)  # 转为Conv1d要求的格式: [batch, in_channels, seq_len]
        out = self.conv(x)
        out = out[:, :, :-self.padding]  # 切除右侧padding，保证因果性
        out = out.transpose(1, 2)  # 转回: [batch, seq_len, out_channels]
        return out

class MTFEM(nn.Module):
    """ 改进1：多尺度时序特征增强模块
        3个并行分支，膨胀率1/2/4，分别捕捉短/中/长时序运动特征
    """
    def __init__(self, in_channels=1, out_channels=64, kernel_size=3):
        super(MTFEM, self).__init__()
        #  3个分支，卷积核3，膨胀率1、2、4
        self.branch_short = nn.Sequential(
            CausalConv1d(in_channels, out_channels, kernel_size, dilation=1),
            nn.LayerNorm(out_channels),
            nn.GELU()
        )
        self.branch_mid = nn.Sequential(
            CausalConv1d(in_channels, out_channels, kernel_size, dilation=2),
            nn.LayerNorm(out_channels),
            nn.GELU()
        )
        self.branch_long = nn.Sequential(
            CausalConv1d(in_channels, out_channels, kernel_size, dilation=4),
            nn.LayerNorm(out_channels),
            nn.GELU()
        )
        #  1×1卷积融合多分支特征，输出增强时序特征
        self.fusion_conv = nn.Conv1d(out_channels * 3, out_channels, kernel_size=1)
        self.fusion_norm = nn.LayerNorm(out_channels)

    def forward(self, x):
        # 输入维度: [batch, seq_len, 1] (单坐标特征)
        feat_short = self.branch_short(x)  # [batch, seq_len, 64]
        feat_mid = self.branch_mid(x)      # [batch, seq_len, 64]
        feat_long = self.branch_long(x)    # [batch, seq_len, 64]

        # 通道维度拼接
        concat_feat = torch.cat([feat_short, feat_mid, feat_long], dim=-1)  # [batch, seq_len, 192]

        # 1×1卷积融合特征
        concat_feat = concat_feat.transpose(1, 2)  # [batch, 192, seq_len]
        fused_feat = self.fusion_conv(concat_feat)  # [batch, 64, seq_len]
        fused_feat = fused_feat.transpose(1, 2)      # [batch, seq_len, 64]
        fused_feat = self.fusion_norm(fused_feat)     # LayerNorm归一化

        return fused_feat

# ==============================================
# 3.  改进2：遮挡感知自适应门控LSTM单元（OAG-LSTM）
# ==============================================
class OAG_LSTM_Cell(nn.Module):
    """ 改进2：遮挡感知自适应门控LSTM单元
        遵循公式4-13至4-17实现
    """
    def __init__(self, input_size, hidden_size):
        super(OAG_LSTM_Cell, self).__init__()
        self.input_size = input_size    # MTFEM输出的特征维度
        self.hidden_size = hidden_size  # LSTM隐藏层维度

        # 原生LSTM门控（ 遗忘门、输出门保持不变）
        self.W_f = nn.Linear(input_size + hidden_size, hidden_size)  # 遗忘门
        self.W_i = nn.Linear(input_size + hidden_size, hidden_size)  # 输入门
        self.W_c = nn.Linear(input_size + hidden_size, hidden_size)  # 候选细胞状态
        self.W_o = nn.Linear(input_size + hidden_size, hidden_size)  # 输出门

        #  遮挡感知门（公式4-13），额外输入1维置信度
        self.W_og = nn.Linear(input_size + hidden_size + 1, hidden_size)

        # 激活函数
        self.sigmoid = nn.Sigmoid()
        self.tanh = nn.Tanh()

    def forward(self, x_t, conf_t, h_prev, c_prev):
        """
        单时间步前向传播
        :param x_t: 当前时刻MTFEM增强特征 [batch, input_size]
        :param conf_t: 当前时刻置信度 [batch, 1]
        :param h_prev: 上一时刻隐藏状态 [batch, hidden_size]
        :param c_prev: 上一时刻细胞状态 [batch, hidden_size]
        :return: h_t 当前隐藏状态, c_t 当前细胞状态
        """
        # 拼接输入特征与上一时刻隐藏状态
        combined = torch.cat([x_t, h_prev], dim=1)  # [batch, input_size + hidden_size]

        # 要求：遗忘门、输出门保持原生LSTM计算逻辑不变
        f_t = self.sigmoid(self.W_f(combined))  # 公式：遗忘门
        o_t = self.sigmoid(self.W_o(combined))  # 公式：输出门

        # 公式4-13：遮挡感知门计算
        og_combined = torch.cat([combined, conf_t], dim=1)  # 拼接置信度
        og_t = self.sigmoid(self.W_og(og_combined))         # 遮挡感知权重 [0,1]

        # 公式4-14：优化输入门，融入遮挡感知权重
        i_t_raw = self.sigmoid(self.W_i(combined))
        i_t = i_t_raw * og_t

        # 公式4-15：优化候选细胞状态，融入遮挡感知权重
        c_tilde_t = self.tanh(self.W_c(combined)) * og_t

        # 公式4-16：细胞状态更新
        c_t = f_t * c_prev + i_t * c_tilde_t

        # 公式4-17：隐藏状态输出
        h_t = o_t * self.tanh(c_t)

        return h_t, c_t

class OAG_LSTM_Layer(nn.Module):
    """
    封装OAG-LSTM单元为双向LSTM层，适配时序序列输入
    """
    def __init__(self, input_size, hidden_size, bidirectional=True):
        super(OAG_LSTM_Layer, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.bidirectional = bidirectional
        self.num_directions = 2 if bidirectional else 1

        # 前向OAG-LSTM单元
        self.forward_cell = OAG_LSTM_Cell(input_size, hidden_size)
        # 反向OAG-LSTM单元（双向）
        self.backward_cell = OAG_LSTM_Cell(input_size, hidden_size) if bidirectional else None

    def forward(self, x_feat, x_conf):
        """
        完整序列前向传播
        :param x_feat: MTFEM增强特征 [batch, seq_len, input_size]
        :param x_conf: 置信度序列 [batch, seq_len, 1]
        :return: 输出序列 [batch, seq_len, hidden_size*num_directions]
        """
        batch_size, seq_len, _ = x_feat.shape
        device = x_feat.device

        # ======================
        # 前向传播
        # ======================
        forward_hidden = []
        h_f = torch.zeros(batch_size, self.hidden_size, device=device)
        c_f = torch.zeros(batch_size, self.hidden_size, device=device)

        for t in range(seq_len):
            h_f, c_f = self.forward_cell(
                x_feat[:, t, :],
                x_conf[:, t, :],
                h_f, c_f
            )
            forward_hidden.append(h_f)
        forward_output = torch.stack(forward_hidden, dim=1)  # [batch, seq_len, hidden_size]

        # ======================
        # 反向传播（双向）
        # ======================
        if self.bidirectional:
            backward_hidden = []
            h_b = torch.zeros(batch_size, self.hidden_size, device=device)
            c_b = torch.zeros(batch_size, self.hidden_size, device=device)

            for t in reversed(range(seq_len)):
                h_b, c_b = self.backward_cell(
                    x_feat[:, t, :],
                    x_conf[:, t, :],
                    h_b, c_b
                )
                backward_hidden.insert(0, h_b)
            backward_output = torch.stack(backward_hidden, dim=1)  # [batch, seq_len, hidden_size]

            # 拼接前向和反向输出
            final_output = torch.cat([forward_output, backward_output], dim=-1)
        else:
            final_output = forward_output

        return final_output

# ==============================================
# 4. 完整步态矫正模型（MTFEM + OAG-LSTM）
# ==============================================
class GaitCorrectionModel(nn.Module):
    def __init__(self, mtfem_in_dim=1, mtfem_out_dim=64, lstm_hidden_dim=128, bidirectional=True):
        super(GaitCorrectionModel, self).__init__()
        # 改进1：多尺度时序特征增强模块
        self.mtfem = MTFEM(in_channels=mtfem_in_dim, out_channels=mtfem_out_dim)
        # 改进2：遮挡感知双向OAG-LSTM层
        self.oag_lstm = OAG_LSTM_Layer(
            input_size=mtfem_out_dim,
            hidden_size=lstm_hidden_dim,
            bidirectional=bidirectional
        )
        # 输出层：映射回1维坐标值
        lstm_out_dim = lstm_hidden_dim * 2 if bidirectional else lstm_hidden_dim
        self.fc = nn.Sequential(
            nn.Linear(lstm_out_dim, lstm_hidden_dim),
            nn.ReLU(),
            nn.Linear(lstm_hidden_dim, 1),
            nn.Sigmoid()  # 约束输出在0-1归一化范围
        )

    def forward(self, x_coord, x_conf):
        """
        模型前向传播
        :param x_coord: 带噪坐标序列 [batch, seq_len, 1]
        :param x_conf: 置信度序列 [batch, seq_len, 1]
        :return: 矫正后的坐标序列 [batch, seq_len, 1]
        """
        # 1. MTFEM提取多尺度时序特征
        mtfem_feat = self.mtfem(x_coord)
        # 2. OAG-LSTM结合置信度做时序建模与噪声抑制
        lstm_out = self.oag_lstm(mtfem_feat, x_conf)
        # 3. 输出层生成矫正后的坐标
        corrected_coord = self.fc(lstm_out)
        return corrected_coord

# ==============================================
# 5. PyTorch Dataset 构建
# ==============================================
# 序列长度设置为1.5个步态周期
seq_len = int(gait_cycle * 1.5)
step = int(gait_cycle * 0.25)  # 滑动步长

class GaitCorrectionDataset(Dataset):
    def __init__(self, coord_sequence, conf_sequence, seq_len, step, noise_level=0.08):
        self.seq_len = seq_len
        self.coord_sequence = torch.FloatTensor(coord_sequence)
        self.conf_sequence = torch.FloatTensor(conf_sequence)
        self.noise_level = noise_level

        # 生成滑动窗口样本索引
        self.sample_indices = []
        for i in range(0, len(coord_sequence) - seq_len + 1, step):
            self.sample_indices.append(i)

    def __len__(self):
        return len(self.sample_indices)

    def __getitem__(self, idx):
        start_idx = self.sample_indices[idx]
        end_idx = start_idx + self.seq_len

        # 干净坐标序列 + 置信度序列
        clean_coord = self.coord_sequence[start_idx:end_idx]
        conf_seq = self.conf_sequence[start_idx:end_idx]

        # 仅给坐标加噪声，模拟遮挡导致的检测错误，置信度保持不变
        noisy_coord = clean_coord + torch.randn_like(clean_coord) * self.noise_level

        # 维度适配：[seq_len] → [seq_len, 1]
        noisy_coord = noisy_coord.unsqueeze(-1)
        conf_seq = conf_seq.unsqueeze(-1)
        clean_coord = clean_coord.unsqueeze(-1)

        return noisy_coord, conf_seq, clean_coord

# 构建训练数据集（仅用右目干净数据训练）
train_dataset = GaitCorrectionDataset(right_y_scaled, right_conf, seq_len, step, noise_level=0.08)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

print(f"Training samples count: {len(train_dataset)}")
print(f"Sequence length per sample: {seq_len} frames")

# 初始化模型
model = GaitCorrectionModel(
    mtfem_in_dim=1,
    mtfem_out_dim=64,
    lstm_hidden_dim=128,
    bidirectional=True
).to(device)

# 损失函数与优化器
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

print("="*60)
print("Model Structure (Strictly Follow Paper):")
print(model)
print("="*60)

# ==============================================
# 6. 模型训练
# ==============================================
num_epochs = 150
patience = 15
best_loss = float('inf')
early_stop_counter = 0
train_losses = []

print("Starting model training...")
for epoch in range(num_epochs):
    model.train()
    epoch_loss = 0.0

    for noisy_coord, conf_seq, clean_coord in train_loader:
        # 数据移到设备
        noisy_coord = noisy_coord.to(device)
        conf_seq = conf_seq.to(device)
        clean_coord = clean_coord.to(device)

        # 前向传播
        optimizer.zero_grad()
        pred_coord = model(noisy_coord, conf_seq)
        loss = criterion(pred_coord, clean_coord)

        # 反向传播与优化
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)  # 梯度裁剪防止爆炸
        optimizer.step()

        epoch_loss += loss.item() * noisy_coord.size(0)

    # 计算平均损失
    epoch_loss = epoch_loss / len(train_loader.dataset)
    train_losses.append(epoch_loss)

    print(f'Epoch {epoch+1}/{num_epochs} | Train Loss: {epoch_loss:.8f}')

    # 早停与最优模型保存
    if epoch_loss < best_loss:
        best_loss = epoch_loss
        early_stop_counter = 0
        torch.save(model.state_dict(), modify_model_save)
    else:
        early_stop_counter += 1
        if early_stop_counter >= patience:
            print(f'Early stopping triggered at epoch {epoch+1}')
            break

# 加载最优模型
model.load_state_dict(torch.load(modify_model_save))
model.eval()

# 绘制训练收敛曲线（保持英文，因为是训练过程）
plt.figure(figsize=(10, 4))
plt.plot(train_losses, label='Training Loss', color='#1f77b4', linewidth=2)
plt.title('Model Training Convergence Curve', fontsize=12)
plt.xlabel('Epoch', fontsize=10)
plt.ylabel('MSE Loss', fontsize=10)
plt.legend()
plt.grid(True, alpha=0.3)
#plt.show()
plt.savefig(modify_result_loss_save)
# ==============================================
# 7. 左目遮挡序列全序列矫正推理
# ==============================================
def reconstruct_full_sequence(coord_sequence, conf_sequence, model, seq_len, device):
    """滑动窗口全序列矫正，无任何平滑处理，100%使用模型原始输出"""
    seq_len_total = len(coord_sequence)
    reconstructed_scaled = np.zeros(seq_len_total)
    count = np.zeros(seq_len_total)

    model.eval()
    with torch.no_grad():
        # 步长1帧滑动，保证每帧都被预测，重叠区域取平均
        for i in range(0, seq_len_total - seq_len + 1, 1):
            # 提取当前窗口
            window_coord = coord_sequence[i:i+seq_len].reshape(1, seq_len, 1)
            window_conf = conf_sequence[i:i+seq_len].reshape(1, seq_len, 1)

            # 转为张量并移到设备
            window_coord_tensor = torch.FloatTensor(window_coord).to(device)
            window_conf_tensor = torch.FloatTensor(window_conf).to(device)

            # 模型预测
            pred_window = model(window_coord_tensor, window_conf_tensor).cpu().numpy().flatten()

            # 累加结果
            reconstructed_scaled[i:i+seq_len] += pred_window
            count[i:i+seq_len] += 1

    # 重叠区域取平均
    reconstructed_scaled = reconstructed_scaled / np.maximum(count, 1)
    # 边缘区域（无足够上下文）用原始值填充
    edge_len = seq_len // 2
    reconstructed_scaled[:edge_len] = coord_sequence[:edge_len]
    reconstructed_scaled[-edge_len:] = coord_sequence[-edge_len:]

    return reconstructed_scaled

# 执行左目序列矫正
print("Starting left sequence correction...")
reconstructed_left_scaled = reconstruct_full_sequence(left_y_scaled, left_conf, model, seq_len, device)
# 反归一化回原始像素范围
reconstructed_left_y = scaler_left.inverse_transform(reconstructed_left_scaled.reshape(-1, 1)).flatten()

# ==============================================
#  8. 评估指标计算
# ==============================================
def evaluate_gait_correction(
        right_clean_y, left_noisy_y, left_corrected_y,
        right_conf, left_conf, scaler_right, scaler_left, gait_cycle
):
    """
    步态遮挡矫正评估函数
    从4个维度全面衡量效果
    """
    print("\n" + "="*60)
    print("步态遮挡矫正效果评估报告")
    print("="*60)
    metrics = {}
    # --------------------------
    # 1. 趋势对齐度（核心指标）
    # --------------------------
    # 归一化到0-1范围，只看趋势，不看数值
    right_norm = scaler_right.transform(right_clean_y.reshape(-1,1)).flatten()
    left_corrected_norm = scaler_left.transform(left_corrected_y.reshape(-1,1)).flatten()
    left_noisy_norm = scaler_left.transform(left_noisy_y.reshape(-1,1)).flatten()
    trend_mse_noisy = mean_squared_error(right_norm, left_noisy_norm)
    trend_mse_corrected = mean_squared_error(right_norm, left_corrected_norm)
    trend_improvement = (1 - trend_mse_corrected / trend_mse_noisy) * 100
    metrics['trend_mse_noisy'] = trend_mse_noisy
    metrics['trend_mse_corrected'] = trend_mse_corrected
    metrics['trend_improvement_rate'] = trend_improvement
    print(f"\n【1】趋势对齐度（核心指标）：")
    print(f"    矫正前 与右侧趋势MSE: {trend_mse_noisy:.6f}")
    print(f"    矫正后 与右侧趋势MSE: {trend_mse_corrected:.6f}")
    print(f"    趋势对齐提升幅度: {trend_improvement:.2f}%")
    # --------------------------
    # 2. 步态周期保真度
    # --------------------------
    def get_all_cycles(sequence, min_cycle=60, max_cycle=150):
        """提取序列中所有检测到的步态周期"""
        seq_norm = (sequence - sequence.mean()) / sequence.std()
        corr = correlate(seq_norm, seq_norm, mode='full')
        corr = corr[len(corr)//2:]
        peaks, _ = find_peaks(corr, height=0.5*corr.max())
        valid_peaks = peaks[(peaks >= min_cycle) & (peaks <= max_cycle)]
        return valid_peaks
    right_cycles = get_all_cycles(right_clean_y)
    left_corrected_cycles = get_all_cycles(left_corrected_y)
    if len(right_cycles) > 0 and len(left_corrected_cycles) > 0:
        main_right_cycle = right_cycles[0]
        main_left_cycle = left_corrected_cycles[0]
        cycle_error = abs(main_right_cycle - main_left_cycle) / main_right_cycle * 100
    else:
        cycle_error = -1
    metrics['gait_cycle_error_rate'] = cycle_error
    print(f"\n【2】步态周期保真度：")
    if cycle_error >= 0:
        print(f"    右侧标准主周期: {main_right_cycle} 帧")
        print(f"    左侧矫正后主周期: {main_left_cycle} 帧")
        print(f"    步态周期相对误差: {cycle_error:.2f}%")
    else:
        print(f"    警告: 周期检测失败，无法评估周期保真度")
    # --------------------------
    # 3. 序列平滑度（去噪效果）
    # --------------------------
    def calculate_smoothness(sequence):
        """计算序列的平滑度：一阶差分的方差，越小越平滑"""
        velocity = np.diff(sequence)
        return np.var(velocity)
    smoothness_noisy = calculate_smoothness(left_noisy_y)
    smoothness_corrected = calculate_smoothness(left_corrected_y)
    smoothness_improvement = (1 - smoothness_corrected / smoothness_noisy) * 100
    metrics['smoothness_noisy'] = smoothness_noisy
    metrics['smoothness_corrected'] = smoothness_corrected
    metrics['smoothness_improvement_rate'] = smoothness_improvement
    print(f"\n【3】序列平滑度（去噪效果）：")
    print(f"    矫正前序列平滑度（方差）: {smoothness_noisy:.2f}")
    print(f"    矫正后序列平滑度（方差）: {smoothness_corrected:.2f}")
    print(f"    平滑度提升幅度: {smoothness_improvement:.2f}%")
    # --------------------------
    # 4. 置信度一致性校验
    # --------------------------
    # 定义高置信度和低置信度阈值
    conf_high_thresh = 0.8
    conf_low_thresh = 0.5
    # 高置信度区域：模型应该尽量保持不变
    high_conf_mask = left_conf > conf_high_thresh
    if np.sum(high_conf_mask) > 0:
        high_conf_change = np.mean(np.abs(left_corrected_y[high_conf_mask] - left_noisy_y[high_conf_mask]))
    else:
        high_conf_change = -1
    # 低置信度区域：模型应该大胆修正
    low_conf_mask = left_conf < conf_low_thresh
    if np.sum(low_conf_mask) > 0:
        # 低置信度区域，看和右侧趋势的对齐度改善
        low_conf_right_norm = right_norm[low_conf_mask]
        low_conf_noisy_norm = left_noisy_norm[low_conf_mask]
        low_conf_corrected_norm = left_corrected_norm[low_conf_mask]
        low_conf_trend_improvement = (
                                             1 - mean_squared_error(low_conf_right_norm, low_conf_corrected_norm) /
                                             mean_squared_error(low_conf_right_norm, low_conf_noisy_norm)
                                     ) * 100
    else:
        low_conf_trend_improvement = -1
    metrics['high_conf_change'] = high_conf_change
    metrics['low_conf_trend_improvement'] = low_conf_trend_improvement
    print(f"\n【4】置信度一致性校验：")
    if high_conf_change >= 0:
        print(f"    高置信度(>{conf_high_thresh})区域平均修正幅度: {high_conf_change:.2f} 像素（数值越小越好）")
    if low_conf_trend_improvement >= 0:
        print(f"    低置信度(<{conf_low_thresh})区域趋势提升幅度: {low_conf_trend_improvement:.2f}%（数值越大越好）")
    print("\n" + "="*60)
    return metrics

# 执行评估
evaluation_metrics = evaluate_gait_correction(
    right_clean_y, left_noisy_y, reconstructed_left_y,
    right_conf, left_conf,
    scaler_right, scaler_left,
    gait_cycle
)

# ==============================================
#9. 结果可视化
# ==============================================
vis_start = 1000
vis_end = 1500
vis_frame = frame[vis_start:vis_end]

plt.figure(figsize=(16, 16))

# 1. 核心矫正结果
plt.subplot(2, 1, 1)
plt.plot(vis_frame, left_noisy_y[vis_start:vis_end], label='原始带噪Y坐标', color='#ff4b42', alpha=0.6)
plt.plot(vis_frame, reconstructed_left_y[vis_start:vis_end], label='矫正后Y坐标', color='#1f77b4', linestyle='--', linewidth=2)
plt.title(f'左脚踝Y坐标矫正结果（趋势提升{evaluation_metrics["trend_improvement_rate"]:.1f}%，平滑度提升{evaluation_metrics["smoothness_improvement_rate"]:.1f}%）', fontsize=13)
plt.ylabel('Y坐标（像素）', fontsize=11)
plt.legend()
plt.grid(True, alpha=0.3)

# 2. 右侧标准趋势参考
plt.subplot(2, 1, 2)
plt.plot(vis_frame, right_clean_y[vis_start:vis_end], label='右侧干净Y坐标', color='#9467bd', linewidth=2)
plt.title('右脚踝Y坐标（标准趋势参考）', fontsize=13)
plt.ylabel('Y坐标（像素）', fontsize=11)
plt.legend()
plt.grid(True, alpha=0.3)

# # 3. 归一化趋势对齐验证
# plt.subplot(2, 1, 3)
# right_scaled_vis = scaler_right.transform(right_clean_y[vis_start:vis_end].reshape(-1,1)).flatten()
# recon_scaled_vis = scaler_left.transform(reconstructed_left_y[vis_start:vis_end].reshape(-1,1)).flatten()
# plt.plot(vis_frame, right_scaled_vis, label='右侧归一化标准趋势', color='#9467bd', linewidth=3)
# plt.plot(vis_frame, recon_scaled_vis, label='左侧矫正后归一化趋势', color='#1f77b4', linestyle='--', linewidth=2)
# plt.title('归一化趋势对齐验证', fontsize=14, fontweight='bold')
# plt.ylabel('归一化数值（0-1）', fontsize=11)
# plt.legend()
# plt.grid(True, alpha=0.3)
#
# # 4. 左侧关键点置信度可视化
# plt.subplot(4, 1, 4)
# plt.plot(vis_frame, left_conf[vis_start:vis_end], label='左侧检测置信度', color='#2ca02c', linewidth=1.5)
# plt.axhline(y=0.8, color='red', linestyle='--', label='高置信度阈值')
# plt.axhline(y=0.5, color='orange', linestyle='--', label='低置信度阈值')
# plt.title('左侧关键点检测置信度', fontsize=13)
# plt.xlabel('帧序号', fontsize=11)
# plt.ylabel('置信度（0-1）', fontsize=11)
# plt.legend()
# plt.grid(True, alpha=0.3)

plt.tight_layout()
#plt.show()
plt.savefig(modify_result_save)
# --------------------------
#  结果保存
# --------------------------
result_df = pd.DataFrame({
    "帧序号": frame,
    "左侧原始带噪Y坐标": left_noisy_y,
    "左侧矫正后Y坐标": reconstructed_left_y,
    "左侧检测置信度": left_conf,
    "右侧标准干净Y坐标": right_clean_y
})
result_df.to_csv(modify_csv_save, index=False, encoding="utf-8-sig")
