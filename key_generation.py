import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import bchlib
import math
from collections import Counter
from hashlib import sha3_256
from scipy.special import erfc, gammaincc
from scipy import stats
from collections import Counter
import warnings
import matplotlib.font_manager as fm

# 禁用警告
warnings.filterwarnings("ignore")

# 设置中文字体
font_list = ['SimHei', 'Microsoft YaHei', 'WenQuanYi Zen Hei']
plt.rcParams['font.sans-serif'] = font_list
plt.rcParams['axes.unicode_minus'] = False

file1 = r'csi_key_227_1.txt'
file2 = r'csi_key_227_11.txt'
# 检查文件存在性
if not os.path.exists(file1) or not os.path.exists(file2):
    print("文件路径错误或文件不存在！")
    exit()

#用于解析.txt文件中的数据，.txt文件中数据是在采取中按一定的规律排列的，因此需要使用这个函数进行提取相关的csi，即函数输出的csi（一个一维数组）


num_eq = 8 # 这个变量的含义暂时不知道含义
CSI_LEN = 56 # length of single CSI
EQUALIZER_LEN = (56-4) # for non HT, four {32767,32767} will be padded to achieve 52 (non HT should have 48)
HEADER_LEN = 2 # timestamp and frequency offset


def parse_side_info(side_info, num_eq, CSI_LEN, EQUALIZER_LEN, HEADER_LEN):
    # print(len(side_info), num_eq, CSI_LEN, EQUALIZER_LEN, HEADER_LEN)
    CSI_LEN_HALF = round(CSI_LEN / 2)
    num_dma_symbol_per_trans = HEADER_LEN + CSI_LEN + num_eq * EQUALIZER_LEN
    num_int16_per_trans = num_dma_symbol_per_trans * 4 # 64bit per dma symbol
    num_trans = round(len(side_info) / num_int16_per_trans)
    # print(len(side_info), side_info.dtype, num_trans)
    side_info = side_info.reshape([num_trans, num_int16_per_trans])

    timestamp = side_info[:, 0] + pow(2, 16) * side_info[:, 1] + pow(2, 32) * side_info[:, 2] + pow(2, 48) * side_info[:, 3]

    freq_offset = (20e6 * np.int16(side_info[:, 4]) / 512) /(2 * 3.14159265358979323846)

    csi = np.zeros((num_trans, CSI_LEN), dtype='int16')
    csi = csi + csi *1j

    equalizer = np.zeros((0, 0), dtype='int16')
    if num_eq >0:
        equalizer = np.zeros((num_trans, num_eq * EQUALIZER_LEN), dtype='int16')
        equalizer = equalizer + equalizer * 1j

    for i in range(num_trans):
        tmp_vec_i = np.int16(side_info[i ,8:(num_int16_per_trans -1):4])
        tmp_vec_q = np.int16(side_info[i ,9:(num_int16_per_trans -1):4])
        tmp_vec = tmp_vec_i + tmp_vec_q *1j
        # csi[i,:] = tmp_vec[0:CSI_LEN]
        csi[i, :CSI_LEN_HALF] = tmp_vec[CSI_LEN_HALF:CSI_LEN]
        csi[i, CSI_LEN_HALF:] = tmp_vec[0:CSI_LEN_HALF]
        if num_eq > 0:
            equalizer[i, :] = tmp_vec[CSI_LEN:(CSI_LEN + num_eq * EQUALIZER_LEN)]
        # print(i, len(tmp_vec), len(tmp_vec[0:CSI_LEN]), len(tmp_vec[CSI_LEN:(CSI_LEN+num_eq*EQUALIZER_LEN)]))

    return timestamp, freq_offset, csi, equalizer

#具体提取csi的函数（输出csi的一维数组），其中调用parse_side_info函数进行提取操作
def get_csi(name):
    side_info_fd = open(name, 'r', encoding='cp936')
    tempcsi = []
   # plt.ion()

    one_len = 1896
    start = 0
    end = 1000
    i = start
    container = 200
    j = container
    sign = 1
    while i < end:
        try:
            if j == container:
                side_info = np.loadtxt(side_info_fd, dtype=float, max_rows=one_len * container)

                # print(side_info)
                # print(side_info.size)

                # print('')
                # print('')
                # print('')

                if side_info.size == 0:
                    print("file has read over")
                    break
                i += 1
                j = 0
            j += 1
            # print(len(side_info))
            timestamp, freq_offset, csi, equalizer = parse_side_info(side_info[(j - 1) * one_len:j * one_len], num_eq ,CSI_LEN, EQUALIZER_LEN, HEADER_LEN)
            # print(timestamp)
            # print(freq_offset)
            # print(csi[0,0:10])
            # print(equalizer[0,0:10])
            # display_side_info(freq_offset, csi, equalizer, waterfall_flag, CSI_LEN, EQUALIZER_LEN)
            if len(csi) != 1 or csi.size != 56:
                continue
            tempcsi = np.append(tempcsi, csi[0][5:51])
        except OSError as e:
            # 如果抛出异常，说明文件打开失败
            print("error of files", e)
            break

    side_info_fd.close()
    return tempcsi

#量化使用的函数，初定是将46*3的csi值进行8分均匀量化，并采用格雷码进行编码，生成256*3位初始密钥
def sinal_quantizationg(a_csi, b_csi):
    # 使用NumPy加速运算
    a_csi_abs = np.abs(a_csi)
    b_csi_abs = np.abs(b_csi)

    a_csi_abs_max, a_csi_abs_min = np.max(a_csi_abs), np.min(a_csi_abs)
    b_csi_abs_max, b_csi_abs_min = np.max(b_csi_abs), np.min(b_csi_abs)

    # 使用整数区间划分
    a_interval = (a_csi_abs_max - a_csi_abs_min) // 8
    b_interval = (b_csi_abs_max - b_csi_abs_min) // 8

    # 避免除以零
    if a_interval == 0 or b_interval == 0:
        raise ValueError("量化区间不能为零")

    # 格雷码编码
    interval = ['000', '001', '011', '010', '110', '111', '101', '100']

    a_t_kay = []
    b_t_kay = []

    for i in range(9, 265):
        # 转换为整数索引
        idx_a = int((a_csi_abs[i] - a_csi_abs_min) // a_interval)
        idx_b = int((b_csi_abs[i] - b_csi_abs_min) // b_interval)

        # 边界检查
        idx_a = min(max(idx_a, 0), 7)
        idx_b = min(max(idx_b, 0), 7)

        a_t_kay.append(interval[idx_a])
        b_t_kay.append(interval[idx_b])

        a_kay = ''.join(a_t_kay)
        b_kay = ''.join(b_t_kay)

    return a_kay, b_kay

#信息调和步骤，初定采用基于纠错编码的信息调和方案，采用BCH码进行纠错：
#计划将256*3位的初始密钥分为64*12的密钥块进行BCH码纠错，并从中选取若干块进行后续操作

# 设置BCH参数：纠错能力t=4，本原多项式对应m=7（码长n=127）
ECC_BITS = 10    # 可纠正最多4位错误（可调整）
ECC_POLY = 137  # 本原多项式设置
bch = bchlib.BCH(ECC_BITS, prim_poly = ECC_POLY)

#转换密钥形式以匹配bch库函数（256*3（b）->8*12（byte））
def bit_to_byte(i_kay):
    # 检查长度
    if len(i_kay) != 256 * 3:
        raise ValueError("字符串长度不正确，应为256×3=768")
    i_kay_groups = [i_kay[i * 64: (i + 1) * 64] for i in range(12)]

    # 将二进制码转化为字节形式：
    byte_groups = []
    for str in i_kay_groups:
        # 确保二进制字符串的长度为 64 位（如果不足，用 0 补齐）
        binary_str = str.zfill(64)
        # 将二进制字符串转换为整数
        integer_value = int(binary_str, 2)
        # 将整数转换为 8 字节的字节序列（大端字节序）
        byte_sequence = integer_value.to_bytes(8, byteorder='big')
        # 将字节序列转换为 bytearray
        byte_groups.append(bytearray(byte_sequence))

    return byte_groups

#BCH码生成函数：
def bch_generation(byte_groups):
    #生成BCH编码：
    bch_groups = []
    for group in byte_groups:
        ecc = bytearray(bch.encode(group))
        bch_groups.append(ecc)

    return bch_groups

#BCH码纠错函数：
def bch_correct(byte_groups, bch_groups):

    key_groups = []

    for i in range(12):
        bitflips = bch.decode(byte_groups[i], bch_groups[i])
        if bitflips >= 0:
            #print(f"第{i+1}组检测到 {bitflips} 位错误)")
            #print("纠正前数据:", byte_groups[i].hex())
            bch.correct(byte_groups[i], bch_groups[i])
            #print("纠正后数据:", byte_groups[i].hex())
            key_groups.append(byte_groups[i].hex())
        #else:
            #print("错误超出纠错能力，无法恢复")

    return byte_groups, key_groups

#评估纠错后的密钥块，并从中选择出64*4位密钥
def key_select(key_groups):

    if len(key_groups) < 2:
        raise ValueError("至少需要4个密钥")

    keys_evaluation = []

    #对于密钥性能的评估
    for idx, key in enumerate(key_groups):
        # 1. 频率偏差计算
        cnt_0 = key.count('0')
        freq_dev = abs(cnt_0 - 32) / 64

        # 2. 最长游程检测
        max_run = current_run = 1
        for i in range(1, len(key)):
            current_run = current_run + 1 if key[i] == key[i - 1] else 1
            if current_run > max_run:
                max_run = current_run

        # 3. 熵值计算
        p0 = cnt_0 / 64
        p1 = 1 - p0
        entropy = - (p0 * math.log2(p0) + p1 * math.log2(p1)) if p0 not in {0, 1} else 0

        # 4. 3-gram分布测试
        ngrams = [key[i:i + 3] for i in range(len(key) - 2)]
        expected = len(ngrams) / 8  # 3位组合共8种可能
        chi2 = sum((count - expected) ** 2 / expected for count in Counter(ngrams).values())

        # 计算综合评分（越小越好）
        score = (
                freq_dev * 10 +  # 频率权重
                max(0, max_run - 5) * 2 +  # 长游程惩罚
                (1 - entropy) * 5 +  # 熵值权重
                chi2 / 100  # 分布均匀性
        )

        keys_evaluation.append((idx, key, score))

    if len(keys_evaluation) < 2:
        raise ValueError("有效密钥不足4个")

    selected_keys = sorted(keys_evaluation, key=lambda x: x[2])[:2]

    return [(item[0], item[1]) for item in selected_keys]

#用于评估密钥的性能，主要用于评估生成的原始密钥的一致性，密钥随机性，以及密钥生成速率

  #评估密钥一致率
def key_BER(akey, bkey):
    akey_arr = "".join(akey)
    bkey_arr = "".join(bkey)

    if len(akey_arr) != len(bkey_arr):
        raise ValueError("输入的序列长度必须相同")

    diff_count = 0      #密钥不一致的位数
    total_length = len(akey_arr)
    for b1, b2 in zip(akey_arr, bkey_arr):
        if int(b1) != int(b2):
            diff_count += 1

    percentage = round((diff_count / total_length) * 100, 2) #密钥不一致率

    return diff_count, percentage



  #评估密钥的随机性

def single_bit_test(key):
    """单比特频数检测"""
    n = len(key)
    ones = sum(int(b) for b in key)
    p_value = stats.binomtest(ones, n, p=0.5).pvalue
    return {
        'p值': round(p_value, 4),
        '通过': p_value >= 0.01,
        '比例': round(ones / n, 4)
    }

def block_frequency_test(key, block_size=128):
    """块内频数检测"""
    n = len(key)
    num_blocks = n // block_size
    if num_blocks < 1:
        return {'通过': False, '错误': '块大小超过密钥长度'}

    proportions = []
    for i in range(num_blocks):
        block = key[i * block_size: (i + 1) * block_size]
        ones = sum(int(b) for b in block)
        proportions.append(ones / block_size)

    chi_square = 4 * block_size * sum((p - 0.5) ** 2 for p in proportions)
    p_value = stats.chi2.sf(chi_square, num_blocks)
    return {
        'p值': round(p_value, 4),
        '通过': p_value >= 0.01,
        '块比例': [round(p, 4) for p in proportions]
    }

def approximate_entropy_test(key, m=3):
    """近似熵检测"""
    n = len(key)

    def phi(m):
        if m == 0:
            return 0
        patterns = [key[i:i + m] for i in range(n - m + 1)]
        counts = Counter(patterns)
        total = len(patterns)
        return sum((v / total) * np.log(v / total) for v in counts.values())

    apen = phi(m) - phi(m + 1)
    chi_square = 2 * n * (np.log(2) - apen)
    p_value = stats.chi2.sf(chi_square, 2 ** m)
    return {
        'p值': round(p_value, 4),
        '通过': p_value >= 0.01
    }

def autocorrelation_test(key, max_lag=40):
    """自相关检测"""
    n = len(key)
    correlations = []
    for d in range(1, max_lag + 1):
        if d >= n:
            continue
        matches = sum(int(key[i]) == int(key[i + d]) for i in range(n - d))
        ratio = matches / (n - d)
        z = (ratio - 0.5) * np.sqrt((n - d) / 0.25)
        p_value = 2 * (1 - stats.norm.cdf(abs(z)))
        correlations.append({
            '滞后': d,
            '相关系数': round(ratio, 4),
            'p值': round(p_value, 4)
        })
    passed_ratio = sum(c['p值'] >= 0.01 for c in correlations) / max_lag
    return {
        '相关系数列表': correlations,
        '通过': passed_ratio >= 0.95
    }

# --------------------- 主检测函数 ---------------------
def randomness_test(key, block_size=128, m_approx_entropy=3, max_lag=40):
    """
    执行随机性检测
    参数：
        key: 二进制密钥序列（列表或字符串）
        block_size: 块检测的块大小
        m_approx_entropy: 近似熵的m值
        max_lag: 自相关检测最大滞后量
    """
    key_str = "".join(key) if isinstance(key, list) else key

    results = {
        '单比特频数': single_bit_test(key_str),
        '块内频数': block_frequency_test(key_str, block_size),
        '近似熵': approximate_entropy_test(key_str, m_approx_entropy),
        '自相关': autocorrelation_test(key_str, max_lag)
    }

    visualize_results(results)
    return results

#可视化函数
def visualize_results(results):
    """可视化检测结果"""
    plt.ioff()
    plt.figure(figsize=(12, 8))

    # 单比特频数
    plt.subplot(2, 2, 1)
    plt.bar(['实际比例'], [results['单比特频数']['比例']], color='skyblue')
    plt.axhline(0.5, color='r', linestyle='--')
    plt.ylim(0.4, 0.6)
    plt.title(f"单比特频数检测\np值={results['单比特频数']['p值']}")

    # 块内频数
    plt.subplot(2, 2, 2)
    props = results['块内频数'].get('块比例', [])
    plt.plot(props, 'o-', color='orange')
    plt.axhline(0.5, color='r', linestyle='--')
    plt.fill_between(range(len(props)), 0.45, 0.55, color='yellow', alpha=0.1)
    plt.title(f"块内频数检测\np值={results['块内频数']['p值']}")

    # 近似熵
    plt.subplot(2, 2, 3)
    status = '通过' if results['近似熵']['通过'] else '未通过'
    plt.text(0.5, 0.5, status,
             ha='center', va='center',
             fontsize=20,
             color='green' if results['近似熵']['通过'] else 'red')
    plt.axis('off')
    plt.title(f"近似熵检测\np值={results['近似熵']['p值']}")

    # 自相关
    plt.subplot(2, 2, 4)
    lags = [c['滞后'] for c in results['自相关']['相关系数列表']]
    corrs = [c['相关系数'] for c in results['自相关']['相关系数列表']]
    plt.plot(lags, corrs, 'o-', color='purple')
    plt.axhline(0.5, color='r', linestyle='--')
    plt.title("自相关检测")

    plt.tight_layout()
    plt.show(block=True)



# 主体运行部分，测试部分
k = 6
acsi = get_csi(file1)[:46*k]
bcsi = get_csi(file2)[:46*k]

# akey，bkey是由csi生成的初始密钥（0/1序列）
akey, bkey = sinal_quantizationg(acsi, bcsi)

a_byte_key = bit_to_byte(akey)
b_byte_key = bit_to_byte(bkey)

b_bch = bch_generation(b_byte_key)

# key_groups为纠错后的密钥，selected_keys为纠错后选择出的具有较高效率的密钥（256位）
corrected_a_byte_key, key_groups = bch_correct(a_byte_key, b_bch)
selected_keys = key_select(key_groups)

# 隐私放大过程
selected_str = [item[1] for item in selected_keys]
key_256 = "".join(selected_str)
key_256_byte = bytes.fromhex(key_256)
hash_obj = sha3_256(key_256_byte).digest()
final_key = hash_obj[:16]

# 打印最终生成的密钥
print(final_key.hex())

# 打印密钥一致率比对结果
diff, pren = key_BER(akey, bkey)
print(diff, pren)

# 打印密钥随机性检测结果

if __name__ == "__main__":

   final_key_hex = final_key.hex()
   final_key_bin = "".join([bin(int(c, 16))[2:].zfill(4) for c in final_key_hex])

   test_results = randomness_test(final_key_bin)
   print("随机性检测报告：")
   print("=" * 35)
#    for test_name, data in test_results.items():
#       print(f"{test_name:　<6} | 状态：{'通过' if data['通过'] else '未通过'} | p值：{data['p值']}")
      
      
      
      
      
      
      
      
      
csi_data = pd.DataFrame({
    "Alice":np.abs(acsi),
    "Bob":np.abs(bcsi)
})



def bit_to_byte(bit_string):
    # 补齐到8的倍数
    while len(bit_string) % 8 != 0:
        bit_string += '0'
    byte_array = bytearray()
    for i in range(0, len(bit_string), 8):
        byte = bit_string[i:i+8]
        byte_array.append(int(byte, 2))
    return bytes(byte_array)  


# =======================================界面设计==========================================

import streamlit as st

# 页面设置
st.set_page_config(
    page_title="密钥协商系统",
    page_icon="🔐",
    layout="wide"
)

# 初始化 Session State
if 'page' not in st.session_state:
    st.session_state.page = 'CSI采集'
if 'csi_data' not in st.session_state:
    st.session_state.csi_data = None
if 'init_key' not in st.session_state:
    st.session_state.init_key = None
if 'final_key' not in st.session_state:
    st.session_state.final_key = None
if 'evaluation_result' not in st.session_state:
    st.session_state.evaluation_result = None

# 顶部标题（加颜色）
st.markdown(
    """
    <h1 style='text-align: center; color: #5C6BC0; font-size: 48px; margin-bottom: 10px;'>🔐 密钥生成与协商系统</h1>
    <hr style="height:2px;border:none;color:#5C6BC0;background-color:#5C6BC0;" />
    """,
    unsafe_allow_html=True
)

# 侧边栏
with st.sidebar:
    st.markdown("<h2 style='color: #3949AB;'>🚀 功能导航</h2>", unsafe_allow_html=True)
    st.markdown("---")
    page = st.radio(
        "选择流程阶段",
        ["🔍 CSI采集和展示", "🛠️ 初始密钥生成", "🤝 密钥协商", "📈 密钥评估"],
        index=["🔍 CSI采集和展示", "🛠️ 初始密钥生成", "🤝 密钥协商", "📈 密钥评估"].index(
            {
                'CSI采集': "🔍 CSI采集和展示",
                '初始密钥': "🛠️ 初始密钥生成",
                '密钥协商': "🤝 密钥协商",
                '密钥评估': "📈 密钥评估",
            }[st.session_state.page]
        )
    )

# 页面映射
page_mapping = {
    "🔍 CSI采集和展示": "CSI采集",
    "🛠️ 初始密钥生成": "初始密钥",
    "🤝 密钥协商": "密钥协商",
    "📈 密钥评估": "密钥评估"
}
st.session_state.page = page_mapping[page]
st.session_state.csi_data = (acsi, bcsi)

# 主体内容
with st.container():
    if st.session_state.page == 'CSI采集':
        with st.container():
            st.markdown("<h3 style='color:#5C6BC0;'>🔍 CSI数据采集与展示</h3>", unsafe_allow_html=True)
            st.write("在此模块中，采集并可视化CSI数据。")
            st.markdown("<hr style='height:1px;border:none;background-color:#9FA8DA;' />", unsafe_allow_html=True)

            if "acsi" not in st.session_state:
                st.session_state.acsi = None
            if "bcsi" not in st.session_state:
                st.session_state.bcsi = None

            if st.button("📡 开始采集CSI数据", use_container_width=True):
                st.session_state.acsi = acsi
                st.session_state.bcsi = bcsi
                st.success("✅ CSI数据采集完成！", icon="📈")

            # 只要 acsi 和 bcsi 存在，就一直画图
            if st.session_state.acsi is not None and st.session_state.bcsi is not None:
                st.line_chart({
                'Alice': np.abs(st.session_state.acsi),
                'Bob': np.abs(st.session_state.bcsi)
                })

            if st.checkbox("显示原始数据"):
                st.table(csi_data)

    elif st.session_state.page == '初始密钥':
        with st.container():
            st.markdown("<h3 style='color:#5C6BC0;'>🛠️ 初始密钥生成</h3>", unsafe_allow_html=True)
            st.write("根据CSI数据生成初始密钥。")
            st.markdown("<hr style='height:1px;border:none;background-color:#9FA8DA;' />", unsafe_allow_html=True)

            if st.button("🔑 生成初始密钥", use_container_width=True):
                # 提取 Alice 和 Bob 的 CSI
                acsi, bcsi = st.session_state.csi_data

                # 使用量化函数生成二进制密钥
                a_init_key, b_init_key = sinal_quantizationg(acsi, bcsi)

                st.session_state.a_init_key = a_init_key
                st.session_state.b_init_key = b_init_key

                # 将二进制密钥转成 bytes 格式
                a_byte_key = bit_to_byte(a_init_key)
                b_byte_key = bit_to_byte(b_init_key)

                st.session_state.a_byte_key = a_byte_key
                st.session_state.b_byte_key = b_byte_key

                st.success("✅ 初始密钥生成成功！", icon="🔐")

            # 显示密钥
            if 'a_init_key' in st.session_state and 'b_init_key' in st.session_state:
                with st.expander("🔎 查看 Alice 和 Bob 的二进制密钥"):
                    st.write("🔑 Alice 的密钥（二进制）:")
                    st.code(st.session_state.a_init_key, language="text")
                    st.write("🔑 Bob 的密钥（二进制）:")
                    st.code(st.session_state.b_init_key, language="text")

                with st.expander("🔎 查看 Alice 和 Bob 的十六进制密钥"):
                    st.write(f"🔑 Alice 的十六进制密钥：{st.session_state.a_byte_key.hex()}")
                    st.write(f"🔑 Bob 的十六进制密钥：{st.session_state.b_byte_key.hex()}")

                # 分块对比密钥
                block_size = 16  # 每16 bit一块
                a_blocks = [st.session_state.a_init_key[i:i+block_size] for i in range(0, len(st.session_state.a_init_key), block_size)]
                b_blocks = [st.session_state.b_init_key[i:i+block_size] for i in range(0, len(st.session_state.b_init_key), block_size)]

                # 保持长度一致
                min_blocks = min(len(a_blocks), len(b_blocks))
                a_blocks = a_blocks[:min_blocks]
                b_blocks = b_blocks[:min_blocks]

                # 生成表格数据
                table_data = {
                    "块编号": [f"块{i+1}" for i in range(min_blocks)],
                    "Alice密钥块": a_blocks,
                    "Bob密钥块": b_blocks,
                    "是否一致": ["✅" if a_blocks[i] == b_blocks[i] else "❌" for i in range(min_blocks)]
                }

                st.markdown("<h4 style='color:#5C6BC0;'>🔍 分块对比表格</h4>", unsafe_allow_html=True)
                st.dataframe(table_data, use_container_width=True)


    elif st.session_state.page == '密钥协商':
        with st.container():
            st.markdown("<h3 style='color:#5C6BC0;'>🤝 密钥协商</h3>", unsafe_allow_html=True)
            st.write("基于初始密钥进行协商。")
            st.markdown("<hr style='height:1px;border:none;background-color:#9FA8DA;' />", unsafe_allow_html=True)

            if st.button("🔁 开始密钥协商", use_container_width=True):
                # 执行密钥协商过程（即隐私放大过程）
                selected_str = [item[1] for item in selected_keys]  # 提取 selected_keys 中的部分数据
                key_256 = "".join(selected_str)  # 合并成一个大字符串
                key_256_byte = bytes.fromhex(key_256)  # 转换为字节
                hash_obj = sha3_256(key_256_byte).digest()  # 哈希生成
                final_key = hash_obj[:16]  # 取前16个字节作为最终密钥

                st.session_state.final_key = final_key.hex()  # 保存最终密钥（以十六进制形式）

                st.success("✅ 密钥协商完成！", icon="🤝")
        
            # 显示最终协商的密钥
            if st.session_state.final_key:
                with st.expander("🔎 查看协商后的密钥"):
                    st.code(st.session_state.final_key, language="text")

    elif st.session_state.page == '密钥评估':
        with st.container():
            st.markdown("<h3 style='color:#5C6BC0;'>📈 密钥评估</h3>", unsafe_allow_html=True)
            st.write("评估协商后的密钥质量。")
            st.markdown("<hr style='height:1px;border:none;background-color:#9FA8DA;' />", unsafe_allow_html=True)

            if st.button("📊 开始评估", use_container_width=True):
                # 检查 final_key 是否存在且不为 None
                if 'final_key' in st.session_state and st.session_state.final_key:
                    final_key = st.session_state.final_key

                    # 确保 final_key 是字节数据，如果是字符串则转换为字节
                    if isinstance(final_key, str):
                        final_key_byte = bytes.fromhex(final_key)  # 如果 final_key 是十六进制字符串，转换为字节
                    else:
                        final_key_byte = final_key  # 如果本来就是字节，直接使用

                    # 将字节转换为二进制字符串
                    final_key_bin = "".join([bin(byte)[2:].zfill(8) for byte in final_key_byte])

                    # 调用随机性检测
                    test_results = randomness_test(final_key_bin)

                    # 保存评估结果到 SessionState
                    st.session_state.evaluation_result = test_results
                    st.success("✅ 密钥评估完成！", icon="📈")

                    # 可视化检测结果
                    st.subheader("🔍 随机性检测图像")
                    
                    # 创建图像
                    plt.ioff()  # 关闭交互式模式
                    fig, axes = plt.subplots(2, 2, figsize=(12, 8))

                    # 单比特频数
                    axes[0, 0].bar(['Actual Proportion'], [test_results['单比特频数']['比例']], color='skyblue')
                    axes[0, 0].axhline(0.5, color='r', linestyle='--')
                    axes[0, 0].set_ylim(0.4, 0.6)
                    axes[0, 0].set_title(f"Single Bit Frequency Test\np-value={test_results['单比特频数']['p值']}")

                    # 块内频数
                    props = test_results['块内频数'].get('块比例', [])
                    axes[0, 1].plot(props, 'o-', color='orange')
                    axes[0, 1].axhline(0.5, color='r', linestyle='--')
                    axes[0, 1].fill_between(range(len(props)), 0.45, 0.55, color='yellow', alpha=0.1)
                    axes[0, 1].set_title(f"Block Frequency Test\np-value={test_results['块内频数']['p值']}")

                    # 近似熵
                    status = 'Passed' if test_results['近似熵']['通过'] else 'Failed'
                    axes[1, 0].text(0.5, 0.5, status, ha='center', va='center', fontsize=20,
                                    color='green' if test_results['近似熵']['通过'] else 'red')
                    axes[1, 0].axis('off')
                    axes[1, 0].set_title(f"Approximate Entropy Test\np-value={test_results['近似熵']['p值']}")

                    # 自相关
                    lags = [c['滞后'] for c in test_results['自相关']['相关系数列表']]
                    corrs = [c['相关系数'] for c in test_results['自相关']['相关系数列表']]
                    axes[1, 1].plot(lags, corrs, 'o-', color='purple')
                    axes[1, 1].axhline(0.5, color='r', linestyle='--')
                    axes[1, 1].set_title("Autocorrelation Test")


                    plt.tight_layout()

                    # 显示图像
                    st.pyplot(fig)

                else:
                    st.error("🔴 无法进行评估，`final_key` 未生成或无效。")

            if 'evaluation_result' in st.session_state and st.session_state.evaluation_result:
                with st.expander("🔎 查看评估结果"):
                    test_results = st.session_state.evaluation_result

                    # 确保 test_results 有数据再进行遍历
                    if test_results:
                        for test_name, result in test_results.items():
                            status = "✅ 通过" if result.get('通过', False) else "❌ 未通过"
                            p_value = result.get('p值', 'N/A')  # 如果没有 p值，返回 'N/A'
                            st.write(f"**{test_name}** ：{status}，p值 = {p_value}")
                    else:
                        st.error("🔴 无法获取有效的评估结果，请检查密钥生成和检测流程。")
