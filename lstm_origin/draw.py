import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

# -------------------------- 基础配置：字体与路径 --------------------------
# 字体设置（若simhei.ttf路径不正确，需修改为实际路径）
try:
    font_path = "simhei.ttf"
    font_prop = fm.FontProperties(fname=font_path)
    fm.fontManager.addfont(font_path)
    plt.rcParams['font.sans-serif'] = [font_prop.get_name()]
except FileNotFoundError:
    # 若SimHei字体不存在，使用系统默认中文字体
    plt.rcParams['font.sans-serif'] = ['WenQuanYi Zen Hei', 'Microsoft YaHei', 'SimSun']
plt.rcParams['axes.unicode_minus'] = False

# 数据路径与保存路径（可根据实际情况修改）
RIGHT_DATA_PATH = '/media/dzy/deep1/train_data2/jiaozheng/20260206_010114/keypoints_2d_right.csv'
LEFT_DATA_PATH = '/media/dzy/deep1/train_data2/jiaozheng/20260206_010114/keypoints_2d_left.csv'
SAVE_DIR = './data1/'

# -------------------------- 1. 数据读取与筛选函数 --------------------------
def load_and_filter_keypoints(file_path, target_kpts):
    """
    读取关键点CSV文件，筛选目标关键点并按帧排序
    :param file_path: CSV文件路径
    :param target_kpts: 目标关键点列表（如['rank', 'rtoe']）
    :return: 筛选后的数据字典（key=关键点名称，value=DataFrame）
    """
    df = pd.read_csv(file_path)
    kpt_dict = {}
    for kpt in target_kpts:
        # 筛选目标关键点 + 按帧排序 + 重置索引
        kpt_data = df[df['kpt_name'] == kpt].sort_values('frame').reset_index(drop=True)
        kpt_dict[kpt] = kpt_data
    return kpt_dict

# -------------------------- 2. 单个关键点绘图函数（核心修改） --------------------------
def plot_single_kpt_curve(kpt_data, reverse, kpt_name, side_label, save_dir):
    """
    单个关键点的X/Y坐标曲线（单独成图，2个子图：X坐标、Y坐标）
    :param kpt_data: 单个关键点的DataFrame（含frame、x、y列）
    :param kpt_name: 关键点名称（如"rank"）
    :param side_label: 左右侧标识（如"右侧"、"左侧"）
    :param save_dir: 图片保存目录
    """
    # 1. 创建画布（2行1列子图，分别展示X、Y坐标）
    fig, (ax_x, ax_y) = plt.subplots(2, 1, figsize=(12, 10))
    fig.suptitle(f'{side_label}{kpt_name}关键点坐标变化曲线',
                 fontsize=16, fontweight='bold', y=0.98)

    # 2. 样式配置（统一视觉风格）
    line_width = 2.5
    marker_size = 5
    color_x = '#2E86AB'  # X坐标曲线颜色（深蓝色）
    color_y = '#A23B72'  # Y坐标曲线颜色（深红色）

    # 3. 子图1：X坐标随帧变化
    xx = kpt_data['x']
    if reverse:
        xx = -xx+480
    ax_x.plot(kpt_data['frame'], xx,
              color=color_x, linewidth=line_width, marker='o', markersize=marker_size,
              label=f'{kpt_name} X坐标', alpha=0.8)
    ax_x.set_title(f'{kpt_name} X坐标变化', fontsize=14, fontweight='bold', pad=15)
    ax_x.set_xlabel('帧序号 (Frame)', fontsize=12)
    ax_x.set_ylabel('X坐标值 (像素)', fontsize=12)
    ax_x.legend(loc='best', fontsize=11)
    ax_x.grid(True, alpha=0.3, linestyle='--')
    ax_x.set_xlim(0, kpt_data['frame'].max())  # X轴范围匹配帧序号

    # 4. 子图2：Y坐标随帧变化
    ax_y.plot(kpt_data['frame'], kpt_data['y'],
              color=color_y, linewidth=line_width, marker='s', markersize=marker_size,
              label=f'{kpt_name} Y坐标', alpha=0.8)
    ax_y.set_title(f'{kpt_name} Y坐标变化', fontsize=14, fontweight='bold', pad=15)
    ax_y.set_xlabel('帧序号 (Frame)', fontsize=12)
    ax_y.set_ylabel('Y坐标值 (像素)', fontsize=12)
    ax_y.legend(loc='best', fontsize=11)
    ax_y.grid(True, alpha=0.3, linestyle='--')
    ax_y.set_xlim(0, kpt_data['frame'].max())  # X轴范围匹配帧序号

    # 5. 布局调整与保存
    plt.tight_layout()
    plt.subplots_adjust(top=0.93)  # 预留标题空间
    save_path = f'{save_dir}/{side_label}_{kpt_name}_x_y_curve.png'
    fig.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f'{side_label}{kpt_name}曲线已保存：{save_path}')

# -------------------------- 3. 数据加载与批量绘图 --------------------------
if __name__ == "__main__":

    right_kpts = load_and_filter_keypoints(RIGHT_DATA_PATH, ['lank', 'ltoe'])
    left_kpts = load_and_filter_keypoints(LEFT_DATA_PATH, ['lank', 'ltoe'])

    # 2. 打印数据基本信息（验证数据有效性）
    print("="*50)
    print("数据加载完成，基本信息：")
    print(f"右侧 - 左脚踝(lank)：{len(right_kpts['lank'])}帧")
    print(f"右侧 - 左脚尖(ltoe)：{len(right_kpts['ltoe'])}帧")
    print(f"左侧 - 左脚踝(lank)：{len(left_kpts['lank'])}帧")
    print(f"左侧 - 左脚尖(ltoe)：{len(left_kpts['ltoe'])}帧")
    print("="*50)

    # 3. 批量绘制单个关键点曲线（核心：每个关键点单独成图）
    # 右侧关键点
    # diff_result = right_kpts['lank'].copy()
    # # 对x/y/conf列做差值计算（df1 - df2）
    # diff_result[[ 'x', 'y', 'conf']] = right_kpts['lank'][[ 'x', 'y', 'conf']] - left_kpts['rank'][[ 'x', 'y', 'conf']]


    plot_single_kpt_curve(right_kpts['lank'][:600], True, 'lank（左脚踝）', '左侧', SAVE_DIR)
    plot_single_kpt_curve(right_kpts['ltoe'][:600], True, 'ltoe（左脚尖）', '左侧', SAVE_DIR)
    # 左侧关键点
    plot_single_kpt_curve(left_kpts['lank'][:600], False,'lank（左脚踝）', '左侧_乱', SAVE_DIR)
    plot_single_kpt_curve(left_kpts['ltoe'][:600], False,'ltoe（左脚尖）', '左侧_乱', SAVE_DIR)

    print("="*50)
    print("所有关键点曲线绘制完成！")
