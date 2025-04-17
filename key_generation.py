import numpy as np
import matplotlib.pyplot as plt
import os
import bchlib
import pandas as pd
import streamlit as st
import warnings
warnings.filterwarnings("ignore", message="loadtxt: input contained no data")

# 参数预设
num_eq = 8 # 这个变量的含义暂时不知道含义
CSI_LEN = 56 # CSI的长度
EQUALIZER_LEN = (56-4) # for non HT, four {32767,32767} will be padded to achieve 52 (non HT should have 48)
HEADER_LEN = 2 # timestamp and frequency offset

# 检查文件存在性
file1 = r'csi_key_227_1.txt'
file2 = r'csi_key_227_11.txt'
if not os.path.exists(file1) or not os.path.exists(file2):
    print("文件路径错误或文件不存在！")
    exit()

# 解析出CSI的函数，主要是将原始数据进行分割和转换，该函数的输入是一个包含原始数据的numpy数组，输出是时间戳、频率偏移、CSI和均衡器数据
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

# 提取CSI的函数，主要是从文件中读取数据并进行处理，该函数的输入是文件名，输出是CSI数据
def get_csi(name):
    side_info_fd = open(name, 'r', encoding='utf-8')
    tempcsi = []
    plt.ion()

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
                if side_info.size == 0:
                    print("file has read over")
                    break
                i += 1
                j = 0
            j += 1
            timestamp, freq_offset, csi, equalizer = parse_side_info(side_info[(j - 1) * one_len:j * one_len], num_eq ,CSI_LEN, EQUALIZER_LEN, HEADER_LEN)
            if len(csi) != 1 or csi.size != 56:
                continue
            tempcsi = np.append(tempcsi, csi[0][5:51])
        except OSError as e:
            # 如果抛出异常，说明文件打开失败
            print("error of files", e)
            break

    side_info_fd.close()
    return tempcsi

# 量化使用的函数，初定是将46*3的csi值进行8分均匀量化，并采用格雷码进行编码，生成256*3位初始密钥
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

# 信息调和步骤，初定采用基于纠错编码的信息调和方案，采用BCH码进行纠错：
# 计划将256*3位的初始密钥分为64*12的密钥块进行BCH码纠错，并从中选取若干块进行后续操作
# 设置BCH参数：纠错能力t=4，本原多项式对应m=7（码长n=127）
ECC_BITS = 10    # 纠错能力设置
ECC_POLY = 137  # 本原多项式设置
bch = bchlib.BCH(ECC_BITS, prim_poly = ECC_POLY)

# 转换密钥形式以匹配bch库函数（256*3（b）->8*12（byte））
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

# BCH码生成函数：
def bch_generation(byte_groups):
    # 生成BCH编码：
    bch_groups = []
    for group in byte_groups:
        ecc = bytearray(bch.encode(group))
        bch_groups.append(ecc)
    return bch_groups

# BCH码纠错函数：
def bch_correct(byte_groups, bch_groups):
    for i in range(12):
        bitflips = bch.decode(byte_groups[i],bch_groups[i])
        if bitflips >= 0:
            print(f"第{i+1}组检测到 {bitflips} 位错误)")
            print("纠正前数据:", byte_groups[i].hex())
            bch.correct(byte_groups[i], bch_groups[i])
            print("纠正后数据:", byte_groups[i].hex())
        else:
            print("错误超出纠错能力，无法恢复")
    return byte_groups


# ================主程序开始================
# 获取CSI数据
k = 6
acsi = get_csi(file1)[:46*k]
bcsi = get_csi(file2)[:46*k]

# 量化CSI数据，初步生成密钥
akey, bkey = sinal_quantizationg(acsi, bcsi)

# 将密钥转换为二进制字符串形式
a_byte_key = bit_to_byte(akey)
b_byte_key = bit_to_byte(bkey)

# 密钥协商
final_key = bch_correct(a_byte_key, bch_generation(b_byte_key))

# 打印出最终密钥
print("Alice的密钥：")
print(akey)
print("Bob的密钥：")
print(bkey)


# ================前端网页展示================
csi_data = pd.DataFrame({
    "Alice":np.abs(acsi),
    "Bob":np.abs(bcsi)
})

a_hex = [x.hex() for x in a_byte_key]  # 对a_byte_key中的每个字节数组进行hex转换
b_hex = [x.hex() for x in b_byte_key]  # 对b_byte_key中的每个字节数组进行hex转换


key_data = pd.DataFrame({
    "Alice": a_hex,
    "Bob": b_hex
})

final_key_16 = pd.DataFrame({
    'Final Key': [x.hex() for x in final_key]
})


# Streamlit前端展示
st.title("无线信道密钥生成")
st.divider()
st.subheader("通信双方的CSI数据对比")
st.line_chart({'Alice': np.abs(acsi), 'Bob': np.abs(bcsi)})
if st.checkbox("显示原始数据"):
    st.table(csi_data)
st.subheader("通信双方初步生成的密钥")
# if st.checkbox("Alice的密钥"):
#     st.write(akey)
# if st.checkbox("Bob的密钥"):
#     st.write(bkey)
# if st.checkbox("16进制密钥"):
#     st.write(key_data)
st.write(key_data)
st.subheader("通信双方的最终协商密钥：")
st.dataframe(final_key_16)
