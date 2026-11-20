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
import matplotlib.font_manager as fm
import matplotlib.pyplot as plt

from config import LEFT_DATA_PATH, RIGHT_DATA_PATH, target1, target2, origin_model_save, origin_result_save, \
    origin_csv_save, origin_result_loss_save

font_path = "simhei.ttf"
font_prop = fm.FontProperties(fname=font_path)
fm.fontManager.addfont(font_path)
plt.rcParams['font.sans-serif'] = font_prop.get_name()
plt.rcParams['axes.unicode_minus'] = False

# 固定随机种子保证结果可复现
np.random.seed(42)
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)

# 设备配置
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"使用设备: {device}")


def load_and_filter_keypoints(file_path, target_kpts):
    df = pd.read_csv(file_path)
    kpt_dict = {}
    for kpt in target_kpts:
        # 严格按帧排序，去除缺失值，保证时序连续
        kpt_data = df[df['kpt_name'] == kpt].sort_values('frame').reset_index(drop=True).dropna(subset=['x', 'y', 'conf'])
        kpt_dict[kpt] = kpt_data
    return kpt_dict

# 加载左脚踝关键点数据
right_kpts = load_and_filter_keypoints(RIGHT_DATA_PATH, [target1])
left_kpts = load_and_filter_keypoints(LEFT_DATA_PATH, [target1])

# 提取坐标+置信度序列
right_clean_y = right_kpts[target1][target2].values
right_conf = right_kpts[target1]['conf'].values
left_noisy_y = left_kpts[target1][target2].values
left_conf = left_kpts[target1]['conf'].values

# 严格对齐左右侧帧长度
total_frames = min(len(right_clean_y), len(left_noisy_y))
right_clean_y = right_clean_y[:total_frames]
right_conf = right_conf[:total_frames]
left_noisy_y = left_noisy_y[:total_frames]
left_conf = left_conf[:total_frames]
frame = np.arange(total_frames)

print(f"加载真实数据完成 - 总帧数: {total_frames}")
print(f"右侧干净序列长度: {right_clean_y.shape}")
print(f"左侧带噪序列长度: {left_noisy_y.shape}")

# --------------------------
# 步态周期提取函数
# --------------------------
def get_gait_cycle(clean_sequence, min_cycle=60, max_cycle=150):
    """从干净序列中自动计算精准的步态周期（帧）"""
    seq_norm = (clean_sequence - clean_sequence.mean()) / clean_sequence.std()
    corr = correlate(seq_norm, seq_norm, mode='full')
    corr = corr[len(corr)//2:]  # 取后半段
    peaks, _ = find_peaks(corr, height=0.5*corr.max())
    valid_peaks = peaks[(peaks >= min_cycle) & (peaks <= max_cycle)]
    if len(valid_peaks) == 0:
        print("警告: 自动周期检测失败，使用默认值100帧")
        return 100
    gait_cycle = valid_peaks[0]
    return gait_cycle

gait_cycle = get_gait_cycle(right_clean_y)
print(f"从右侧数据检测到步态周期: {gait_cycle} 帧")

# --------------------------
# 数据归一化
# --------------------------
# 右侧归一化（训练用，模型学习此范围的标准趋势）
scaler_right = MinMaxScaler(feature_range=(0, 1))
right_scaled = scaler_right.fit_transform(right_clean_y.reshape(-1, 1)).flatten()

# 左侧归一化（推理用，最终反归一化回此范围）
scaler_left = MinMaxScaler(feature_range=(0, 1))
left_scaled = scaler_left.fit_transform(left_noisy_y.reshape(-1, 1)).flatten()

# --------------------------
# 3. PyTorch 数据集构建
# --------------------------
# 序列长度设置为1.5个步态周期，保证模型学习到完整的周期趋势
seq_len = int(gait_cycle * 1.5)
step = int(gait_cycle * 0.25)  # 滑动步长，保证样本重叠充分
n_features = 1

class GaitDenoiseDataset(Dataset):
    def __init__(self, clean_sequence, seq_len, step, noise_level=0.05):
        self.seq_len = seq_len
        self.clean_sequence = torch.FloatTensor(clean_sequence)
        self.noise_level = noise_level

        # 生成滑动窗口样本
        self.samples = []
        for i in range(0, len(clean_sequence) - seq_len + 1, step):
            self.samples.append(self.clean_sequence[i:i+seq_len])

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        clean_seq = self.samples[idx]
        # 给干净序列加随机噪声，让模型学习去噪+趋势还原
        noisy_seq = clean_seq + torch.randn_like(clean_seq) * self.noise_level
        # 输入：带噪声的序列，输出：干净的标准序列
        return noisy_seq.unsqueeze(-1), clean_seq.unsqueeze(-1)

# 创建训练数据集
train_dataset = GaitDenoiseDataset(right_scaled, seq_len, step, noise_level=0.08)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
print(f"训练样本数量: {len(train_dataset)}")
print(f"每个样本序列长度: {seq_len} 帧 (1.5个步态周期)")

# --------------------------
# BiLSTM去噪自编码器模型
# --------------------------
class BiLSTM_DenoiseAutoEncoder(nn.Module):
    def __init__(self, input_size, hidden_size=128, num_layers=2, dropout=0.1):
        super(BiLSTM_DenoiseAutoEncoder, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        # 编码器：双向LSTM提取步态周期特征
        self.encoder = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0
        )

        # 解码器：双向LSTM还原符合右侧标准的干净序列
        self.decoder = nn.LSTM(
            input_size=hidden_size * 2,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0
        )

        # 输出层：还原为0-1范围的序列值
        self.fc = nn.Linear(hidden_size * 2, input_size)
        self.sigmoid = nn.Sigmoid()  # 约束输出在0-1范围，和归一化匹配

    def forward(self, x):
        # x shape: (batch_size, seq_len, input_size)
        # 编码器：提取步态特征
        enc_out, _ = self.encoder(x)
        # 解码器：还原标准趋势序列
        dec_out, _ = self.decoder(enc_out)
        # 输出层：生成最终干净序列
        out = self.fc(dec_out)
        out = self.sigmoid(out)
        return out

# 初始化模型
model = BiLSTM_DenoiseAutoEncoder(input_size=n_features, hidden_size=128, num_layers=2).to(device)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
print("="*50)
print("模型结构:")
print(model)
print("="*50)

# --------------------------
# 模型训练
# --------------------------
num_epochs = 150
patience = 15
best_loss = float('inf')
early_stop_counter = 0
train_losses = []

print("开始模型训练...")
for epoch in range(num_epochs):
    model.train()
    epoch_loss = 0.0

    for noisy_seq, clean_seq in train_loader:
        noisy_seq = noisy_seq.to(device)
        clean_seq = clean_seq.to(device)

        optimizer.zero_grad()
        output_seq = model(noisy_seq)
        loss = criterion(output_seq, clean_seq)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        epoch_loss += loss.item() * noisy_seq.size(0)

    epoch_loss = epoch_loss / len(train_loader.dataset)
    train_losses.append(epoch_loss)

    print(f'轮次 {epoch+1}/{num_epochs}, 训练损失: {epoch_loss:.8f}')

    # 保存最优模型
    if epoch_loss < best_loss:
        best_loss = epoch_loss
        early_stop_counter = 0
        torch.save(model.state_dict(), origin_model_save)
    else:
        early_stop_counter += 1
        if early_stop_counter >= patience:
            print(f'在第 {epoch+1} 轮触发早停，停止训练')
            break

# 加载最优模型
model.load_state_dict(torch.load(origin_model_save))
model.eval()

# 绘制训练收敛曲线
plt.figure(figsize=(10, 4))
plt.plot(train_losses, label='训练损失', color='#1f77b4', linewidth=2)
plt.title('训练收敛曲线', fontsize=12)
plt.xlabel('训练轮次', fontsize=10)
plt.ylabel('MSE损失值', fontsize=10)
plt.legend()
plt.grid(True, alpha=0.3)
#plt.show()
plt.savefig(origin_result_loss_save)

# --------------------------
# 全序列推理矫正
# --------------------------
def reconstruct_full_sequence(noisy_scaled, model, seq_len, device):
    """
    滑动窗口全序列重构，保证每一段都符合右侧标准趋势
    """
    seq_len_total = len(noisy_scaled)
    # 初始化输出数组
    reconstructed_scaled = np.zeros(seq_len_total)
    count = np.zeros(seq_len_total)  # 用于重叠区域平均

    model.eval()
    with torch.no_grad():
        # 滑动窗口遍历全序列
        for i in range(0, seq_len_total - seq_len + 1, 1):
            # 提取当前窗口序列
            window_seq = noisy_scaled[i:i+seq_len].reshape(1, seq_len, 1)
            window_tensor = torch.FloatTensor(window_seq).to(device)
            # 模型输出符合右侧趋势的干净序列
            pred_window = model(window_tensor).cpu().numpy().flatten()
            # 累加结果，重叠区域取平均
            reconstructed_scaled[i:i+seq_len] += pred_window
            count[i:i+seq_len] += 1

    # 处理重叠区域，避免边缘值异常
    reconstructed_scaled = reconstructed_scaled / np.maximum(count, 1)
    # 前seq_len/2和后seq_len/2的边缘区域，用原始值填充（无足够上下文）
    edge_len = seq_len // 2
    reconstructed_scaled[:edge_len] = noisy_scaled[:edge_len]
    reconstructed_scaled[-edge_len:] = noisy_scaled[-edge_len:]

    return reconstructed_scaled

print("开始全序列矫正重构...")
reconstructed_left_scaled = reconstruct_full_sequence(left_scaled, model, seq_len, device)

# 反归一化回左侧原始数值范围
reconstructed_left_y = scaler_left.inverse_transform(reconstructed_left_scaled.reshape(-1, 1)).flatten()

# ==============================================
# 评估指标计算
# ==============================================
def evaluate_gait_correction(
        right_clean_y, left_noisy_y, left_corrected_y,
        right_conf, left_conf, scaler_right, scaler_left, gait_cycle
):
    """
    步态遮挡矫正 评估函数
    从4个维度全面衡量效果
    """
    print("\n" + "="*60)
    print("步态遮挡矫正效果评估报告")
    print("="*60)

    metrics = {}

    # --------------------------
    # 1. 趋势对齐度
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

    print(f"\n【1】趋势对齐度：")
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
    # 3. 序列平滑度
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

    print(f"\n【3】序列平滑度：")
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
# 结果可视化（全中文标签）
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
# plt.subplot(4, 1, 3)
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
plt.savefig(origin_result_save)

# --------------------------
# 结果保存
# --------------------------
result_df = pd.DataFrame({
    "帧序号": frame,
    "左侧原始带噪Y坐标": left_noisy_y,
    "左侧矫正后Y坐标": reconstructed_left_y,
    "左侧检测置信度": left_conf,
    "右侧标准干净Y坐标": right_clean_y
})
result_df.to_csv(origin_csv_save, index=False, encoding="utf-8-sig")
