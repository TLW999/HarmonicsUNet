import io
import os
import pandas as pd
from pydub import AudioSegment
from pydub.effects import normalize
import requests
import math
import hashlib
import csv
import librosa
import numpy as np

# 设置 Xeno-canto API v3 密钥
API_KEY = "c0daa02868b7a79d6e17f9fe0184cfd42b2b8daf"

# 鸟类列表：(学名, 中文名)
birds = [
    ("Turdus merula", "欧歌鸫"),          # 旋律性鸣声，清脆复杂
    ("Pycnonotus barbatus", "黑头鹎"),     # 中频鸣声，节奏鲜明
    ("Parus major", "大山雀"),            # 高频短促哨音
    ("Alcedo atthis", "普通翠鸟"),         # 尖锐单音叫声
    ("Falco tinnunculus", "红隼"),         # 高频尖叫声
    ("Picus viridis", "绿啄木鸟"),        # 啄木声和鸣叫
    ("Streptopelia decaocto", "欧斑鸠"),   # 低频咕咕声
    ("Motacilla alba", "白鹡鸰"),         # 短促清脆叫声
    ("Emberiza citrinella", "黄鹀"),       # 中频鸣声，节奏规律
    ("Larus argentatus", "银鸥"),          # 粗糙海鸟叫声
    ("Lanius collurio", "红背伯劳"),      # 刺耳鸣声
    ("Phoenicurus phoenicurus", "红尾鸲"), # 高频旋律性鸣声
    ("Cyanistes caeruleus", "蓝山雀"),    # 高频哨音，短促
    ("Cuculus canorus", "大杜鹃"),        # 经典“布谷”声
    ("Oriolus oriolus", "金黄鹂"),        # 悠扬旋律性鸣声
    ("Acridotheres cristatellus", "小八哥"), # 多样化叫声，模仿性强
    ("Dendrocopos major", "大斑啄木鸟"),  # 快速啄木声
    ("Aythya fuligula", "凤头潜鸭"),      # 低频水鸟叫声
    ("Phalacrocorax carbo", "普通鸬鹚"),  # 低沉咕噜声
    ("Leiothrix argentauris", "银耳相思鸟"), # 高频旋律性鸣声
    ("Muscicapa striata", "斑鹟"),        # 简单短促鸣声
    ("Pycnonotus cafer", "红臀鹎"),       # 中频鸣声，节奏鲜明
    ("Sitta europaea", "普通鸮"),         # 高频尖锐叫声
    ("Corvus corone", "小嘴乌鸦"),        # 低频粗糙叫声
    ("Buteo buteo", "普通鵟"),            # 高频鹰鸣
    ("Erithacus rubecula", "欧亚鸲"),     # 复杂旋律性鸣声
    ("Columba palumbus", "林鸽"),         # 低频咕咕声，节奏慢
    ("Fringilla coelebs", "苍头燕雀"),   # 高频快速鸣声
    ("Chloris chloris", "绿雀"),          # 高频旋律性鸣声
    ("Accipiter nisus", "雀鹰")           # 高频尖锐叫声
]
# 创建输出目录和 fold 目录
output_dir = "bird_audio_segments"
os.makedirs(output_dir, exist_ok=True)
for i in range(1, 11):  # 创建 fold1 到 fold10
    os.makedirs(os.path.join(output_dir, f"fold{i}"), exist_ok=True)


# 创建并初始化 CSV 文件
csv_path = os.path.join(output_dir, "bird_audio_segments.csv")
fieldnames = [
    "segment_file", "scientific_name", "chinese_name", "recording_id",
    "start_time_s", "end_time_s", "original_file", "fold", "classID", "class"
]
with open(csv_path, 'w', newline='', encoding='utf-8') as csvfile:
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    writer.writeheader()



# Xeno-canto API v3 查询
base_url = "https://xeno-canto.org/api/3/recordings"
headers = {}

# 目标样本数
target_samples = 6000
current_samples = 0

target_per_class = 200


# 检查是否已有进度
if os.path.exists(csv_path):
    with open(csv_path, 'r', encoding='utf-8') as csvfile:
        reader = csv.DictReader(csvfile)
        existing_rows = list(reader)
        current_samples = len(existing_rows)
        print(f"恢复已有进度: {current_samples} 个样本")
else:
    print("未找到进度文件，从头开始处理")



def is_noise_segment(segment, sr=16000, threshold=0.02):
    y = np.array(segment.get_array_of_samples()) / 32768.0
    spec = librosa.feature.melspectrogram(y=y, sr=sr, fmin=500, fmax=8000)
    return spec.mean() < threshold

# 生成 classID (基于学名的哈希值)
def generate_class_id(scientific_name):
    return hashlib.md5(scientific_name.encode()).hexdigest()[:8]  # 取前8位作为唯一ID


# 构建 classID 和 class 映射
class_mapping = {}
for scientific_name, chinese_name in birds:
    class_id = generate_class_id(scientific_name)
    class_mapping[scientific_name] = {
        "classID": class_id,
        "class": chinese_name
    }

#处理每个鸟类
for scientific_name, chinese_name in birds:
    if current_samples >= target_samples:
        break
    print(f"正在处理：{chinese_name} ({scientific_name})")

    per_class_count = 0
    # 查询 API v3
    params = {
        "query": f'sp:"{scientific_name}" q:A',  # 限定质量为 A
        "key": API_KEY,
        "page": 1,
        "per_page": 500  # 增加每页记录数
    }
    try:
        response = requests.get(base_url, headers=headers, params=params)
        print(f"响应状态码: {response.status_code}")
        print(f"响应数据: {response.text}")
        if response.status_code != 200:
            print(f"API 请求失败: {response.status_code} - {chinese_name}")
            continue


        data = response.json()
        recordings = data.get("recordings", [])
        if not recordings:
            print(f"未找到 {chinese_name} 的录音, 数据: {data}")
            continue
    except Exception as e:
        print(f"查询 {chinese_name} 失败: {e}")
        continue

    per_class_count = 0

    # 处理录音
    for rec in recordings:
        if current_samples >= target_samples or per_class_count >= target_per_class:
            break

        rec_id = rec.get("id")
        file_url = rec.get("file")
        if not rec_id or not file_url:
            print(f"录音缺少 id 或 file URL: {rec}")
            continue

        print(f"处理录音 ID: {rec_id}")

        # 下载音频文件
        try:
            audio_response = requests.get(file_url)
            if audio_response.status_code == 200:
                audio = AudioSegment.from_file(io.BytesIO(audio_response.content))
            else:
                continue
        except Exception as e:
            print(f"下载 {chinese_name} 的录音 {rec_id} 失败: {e}")
            continue

        # 加载并转换音频为 16kHz
        try:
            audio = audio.set_frame_rate(16000)  # 统一为 16kHz
            audio = normalize(audio)  # 标准化音量
            duration_ms = len(audio)
            segment_length_ms = 5 * 1000  # 5 秒
            num_segments = math.ceil(duration_ms / segment_length_ms)
            print(f"音频时长: {duration_ms / 1000:.1f}秒, 可切分 {num_segments} 个片段")

            # 处理每个片段
            for seg_idx in range(num_segments):
                if current_samples >= target_samples or per_class_count >= target_per_class:
                    break

                start_ms = seg_idx * segment_length_ms
                end_ms = min(start_ms + segment_length_ms, duration_ms)
                segment = audio[start_ms:end_ms]

                # 跳过噪声片段
                if is_noise_segment(segment):
                    print(f"跳过噪声片段: {start_ms / 1000}-{end_ms / 1000}秒")
                    continue

                # 分配到 fold (0-9 对应 fold1-fold10)
                fold_num = (current_samples % 10) + 1
                fold_dir = os.path.join(output_dir, f"fold{fold_num}")
                segment_name = f"{scientific_name}_{rec_id}_{int(start_ms / 1000)}.wav"
                segment_path = os.path.join(fold_dir, segment_name)

                # 保存片段
                segment.export(segment_path, format="wav")

                # 准备CSV数据
                row = {
                    "segment_file": segment_name,
                    "scientific_name": scientific_name,
                    "chinese_name": chinese_name,
                    "recording_id": rec_id,
                    "start_time_s": start_ms / 1000,
                    "end_time_s": end_ms / 1000,
                    "original_file": f"{scientific_name}_{rec_id}.wav",  # 原始文件名（未实际保存）
                    "fold": f"fold{fold_num}",
                    "classID": class_mapping[scientific_name]["classID"],
                    "class": class_mapping[scientific_name]["class"]
                }

                # 实时追加写入CSV
                with open(csv_path, 'a', newline='', encoding='utf-8') as csvfile:
                    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                    writer.writerow(row)

                per_class_count += 1
                current_samples += 1
                print(f"样本 {current_samples}/{target_samples}: 保存到 {fold_dir}/{segment_name}")

        except Exception as e:
            print(f"处理录音 {rec_id} 失败: {e}")
            continue

        print(f"处理完成！总样本数：{current_samples}")

        # 可选：将CSV转换为Pandas DataFrame进行验证
        try:
            df = pd.read_csv(csv_path)
            print("\nCSV文件摘要:")
            print(f"总行数: {len(df)}")
            print(f"鸟类种类: {df['class'].nunique()}")
            print(f"Fold分布:\n{df['fold'].value_counts()}")
        except Exception as e:
            print(f"读取CSV文件失败: {e}")