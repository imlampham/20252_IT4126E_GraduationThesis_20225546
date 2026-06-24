#!/usr/bin/env python3
"""
build_centroids.py — Tính centroid cho DeepMASQUE
==================================================
Chạy 1 lần trước khi demo để cache centroids.

Usage:
    python3 scripts/build_centroids.py

Output:
    checkpoints/centroids.npy
    checkpoints/label_map.json
    checkpoints/scaler.pkl
    checkpoints/selector.pkl
"""

import os, sys, gc, json, time, pickle
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_classif

# ── đường dẫn tương đối từ root project ──
ROOT         = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATASET_DIR  = os.path.join(ROOT, "Demo/dataset")
CKPT_DIR     = os.path.join(ROOT, "Demo/checkpoints")
WEIGHTS_PATH = os.path.join(CKPT_DIR, "latest.weights.h5")
OUT_CENTROIDS = os.path.join(CKPT_DIR, "centroids.npy")
OUT_LABELS    = os.path.join(CKPT_DIR, "label_map.json")
OUT_SCALER    = os.path.join(CKPT_DIR, "scaler.pkl")
OUT_SELECTOR  = os.path.join(CKPT_DIR, "selector.pkl")

SEQ_LENGTH    = 5000
META_DIM_RAW  = 13
META_DIM_SEL  = 9       # sau SelectKBest
#MAX_PER_CLASS = 50      # số trace tối đa mỗi class để tính centroid
BATCH_SIZE    = 16


# ══════════════════════════════════════════
# FEATURE EXTRACTION
# ══════════════════════════════════════════
def csv_to_features(df, seq_length=5000):
    """
    Input : DataFrame với cột:
            protocol;length;relative_time;direction;src_ip;src_port;dst_ip;dst_port
    Output: (dir_seq, iat_seq, size_norm, meta_13)
    """
    # 1. Trích xuất mảng thô ban đầu từ dữ liệu CSV
    dirs    = np.where(df['direction'].values == 0, -1, 1)
    times   = df['relative_time'].values
    lengths = df['length'].values

    # 2. Khởi tạo mảng đệm (Padding) với kiểu dữ liệu KHỚP TUYỆT ĐỐI với notebook gốc
    dir_seq  = np.zeros(seq_length, dtype=np.int8)       # Notebook dùng int8
    time_seq = np.zeros(seq_length, dtype=np.float32)    # Notebook dùng float32
    size_seq = np.zeros(seq_length, dtype=np.float32)    # Notebook dùng float32

    limit = min(len(dirs), seq_length)
    dir_seq[:limit]  = dirs[:limit]
    time_seq[:limit] = times[:limit]
    size_seq[:limit] = lengths[:limit]

    # 3. Tính toán Inter-Arrival Time (IAT) sau khi padding (Khớp với to_input_arrays)
    # Notebook thực hiện: inter[:, 1:] = time_arr[:, 1:] - time_arr[:, :-1] trên mảng đã padding
    iat_seq = np.zeros_like(time_seq)
    iat_seq[1:] = time_seq[1:] - time_seq[:-1]

    # 4. Chuẩn hóa kích thước gói tin (Size)
    size_norm = size_seq / 1500.0

    # 5. Tính toán Metadata 13 chiều dựa trên file gốc (Khớp logic notebook gốc)
    in_mask, out_mask = (dirs == -1), (dirs == 1)
    ti = int(np.sum(in_mask))
    to_ = int(np.sum(out_mask))
    tp = ti + to_
    tt = float(times[-1]) if tp > 0 else 0.0

    if tp == 0:
        meta = np.zeros(13, dtype=np.float32)
    else:
        msi = float(np.mean(lengths[in_mask]))  if ti   > 0 else 0.0
        mso = float(np.mean(lengths[out_mask])) if to_ > 0 else 0.0
        mst = float(np.mean(lengths))           if tp   > 0 else 0.0
        ri  = msi / mst if mst > 0 else 0.0
        ro  = mso / mst if mst > 0 else 0.0
        
        tin  = times[in_mask]
        tout = times[out_mask]
        mti  = float(np.mean(np.diff(tin)))  if len(tin)   > 1 else 0.0
        mto  = float(np.mean(np.diff(tout))) if len(tout) > 1 else 0.0
        
        meta = np.array(
            [tp, ti, to_, ti/tp, to_/tp, tt, tt/tp, mso, msi, ri, ro, mti, mto],
            dtype=np.float32
        )

    return dir_seq, iat_seq, size_norm, meta


# ══════════════════════════════════════════
# MODEL BUILD (copy từ app.py)
# ══════════════════════════════════════════
def build_model(num_classes, meta_dim, seq_length=5000):
    import tensorflow as tf
    from tensorflow.keras.models import Model
    from tensorflow.keras.layers import (
        Conv1D, MaxPooling1D, Dense, Activation, ZeroPadding1D,
        GlobalAveragePooling1D, Add, Concatenate, Dropout,
        BatchNormalization, Input, Lambda
    )

    params = {'kernel_initializer': 'he_normal'}

    def dilated_basic_1d(filters, suffix, stage=0, block=0,
                         kernel_size=3, numerical_name=False,
                         stride=None, dilations=(1, 1)):
        if stride is None:
            stride = 1 if block != 0 or stage == 0 else 2
        block_char = f'b{block}' if (block > 0 and numerical_name) else chr(ord('a') + block)
        stage_char  = str(stage + 2)

        def f(x):
            y = Conv1D(filters, kernel_size, padding='causal', strides=stride,
                       dilation_rate=dilations[0], use_bias=False,
                       name=f'res{stage_char}{block_char}_branch2a_{suffix}', **params)(x)
            y = BatchNormalization(epsilon=1e-5,
                                   name=f'bn{stage_char}{block_char}_branch2a_{suffix}')(y)
            y = Activation('relu',
                           name=f'res{stage_char}{block_char}_branch2a_relu_{suffix}')(y)
            y = Conv1D(filters, kernel_size, padding='causal', use_bias=False,
                       dilation_rate=dilations[1],
                       name=f'res{stage_char}{block_char}_branch2b_{suffix}', **params)(y)
            y = BatchNormalization(epsilon=1e-5,
                                   name=f'bn{stage_char}{block_char}_branch2b_{suffix}')(y)
            if block == 0:
                shortcut = Conv1D(filters, 1, strides=stride, use_bias=False,
                                  name=f'res{stage_char}{block_char}_branch1_{suffix}', **params)(x)
                shortcut = BatchNormalization(epsilon=1e-5,
                                              name=f'bn{stage_char}{block_char}_branch1_{suffix}')(shortcut)
            else:
                shortcut = x
            y = Add(name=f'res{stage_char}{block_char}_{suffix}')([y, shortcut])
            y = Activation('relu', name=f'res{stage_char}{block_char}_relu_{suffix}')(y)
            return y
        return f

    def ResNet18_1D(inputs, suffix, blocks=[2,2,2,2], numerical_names=[True]*4):
        x = ZeroPadding1D(padding=3, name=f'padding_conv1_{suffix}')(inputs)
        x = Conv1D(64, 7, strides=2, use_bias=False, name=f'conv1_{suffix}')(x)
        x = BatchNormalization(epsilon=1e-5, name=f'bn_conv1_{suffix}')(x)
        x = Activation('relu', name=f'conv1_relu_{suffix}')(x)
        x = MaxPooling1D(3, strides=2, padding='same', name=f'pool1_{suffix}')(x)
        features = 64
        for stage_id, iterations in enumerate(blocks):
            x = dilated_basic_1d(features, suffix, stage_id, 0,
                                 dilations=(1, 2), numerical_name=False)(x)
            for block_id in range(1, iterations):
                x = dilated_basic_1d(features, suffix, stage_id, block_id,
                                     dilations=(4, 8),
                                     numerical_name=(block_id > 0 and numerical_names[stage_id]))(x)
            features *= 2
        x = GlobalAveragePooling1D(name=f'pool5_{suffix}')(x)
        return x

    dir_input  = Input(shape=(seq_length, 1), name='dir_input')
    time_input = Input(shape=(seq_length, 1), name='time_input')
    size_input = Input(shape=(seq_length, 1), name='size_input')
    meta_input = Input(shape=(meta_dim,),     name='metadata_input')

    dir_out  = ResNet18_1D(dir_input,  'dir')
    time_out = ResNet18_1D(time_input, 'time')
    size_out = ResNet18_1D(size_input, 'size')

    meta_out = Dense(32)(meta_input)
    meta_out = BatchNormalization()(meta_out)
    meta_out = Activation('relu')(meta_out)

    combined = Concatenate()([dir_out, time_out, size_out, meta_out])

    fc = Dense(1024, name='fc1')(combined)
    fc = BatchNormalization(name='fc1_bn')(fc)
    fc = Activation('relu', name='fc1_relu')(fc)
    fc = Dropout(0.6, name='fc1_drop')(fc)
    class_out = Dense(num_classes, activation='softmax', name='class_output')(fc)

    import tensorflow as tf
    emb = Dense(128, name='emb_proj')(combined)
    emb_out = Lambda(
        lambda x: tf.math.l2_normalize(x, axis=1), name='emb_output'
    )(emb)

    return Model(
        inputs=[dir_input, time_input, size_input, meta_input],
        outputs=[class_out, emb_out]
    )


# ══════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════
def main():
    print("=" * 55)
    print("  DeepMASQUE — Build Centroids")
    print("=" * 55)

    # Kiểm tra thư mục
    for path, name in [(DATASET_DIR, "dataset/"), (CKPT_DIR, "checkpoints/"), (WEIGHTS_PATH, "latest.weights.h5")]:
        if not os.path.exists(path):
            print(f"[!] Not found: {name} ({path})")
            sys.exit(1)

    os.makedirs(CKPT_DIR, exist_ok=True)

    # ── Bước 1: Scan dataset ──
    print(f"\n[1/5] Scanning dataset: {DATASET_DIR}")
    classes = sorted([
        d for d in os.listdir(DATASET_DIR)
        if os.path.isdir(os.path.join(DATASET_DIR, d))
    ])
    print(f"      Found {len(classes)} classes")

    # ── Bước 2: Load tất cả traces ──
    print(f"\n[2/5] Loading traces (all)...")
    #print(f"\n[2/5] Loading traces (max {MAX_PER_CLASS} per class)...")
    t_start = time.time()

    all_dirs, all_iats, all_sizes, all_metas, all_labels = [], [], [], [], []
    label_map = {}
    skipped = 0

    for idx, cls_name in enumerate(classes):
        label_map[idx] = cls_name
        cls_dir = os.path.join(DATASET_DIR, cls_name)
        #files   = sorted([f for f in os.listdir(cls_dir) if f.endswith('.csv')])[:MAX_PER_CLASS]
        files = sorted([f for f in os.listdir(cls_dir) if f.endswith('.csv')])

        loaded = 0
        for fname in files:
            try:
                df = pd.read_csv(os.path.join(cls_dir, fname), sep=';')
                if len(df) < 10:
                    skipped += 1
                    continue
                d, t, s, m = csv_to_features(df, SEQ_LENGTH)
                all_dirs.append(d)
                all_iats.append(t)
                all_sizes.append(s)
                all_metas.append(m)
                all_labels.append(idx)
                loaded += 1
            except Exception:
                skipped += 1
                continue

        if (idx + 1) % 50 == 0 or idx == len(classes) - 1:
            print(f"      [{idx+1:3d}/{len(classes)}] {cls_name}: {loaded} traces")

    n_total = len(all_dirs)
    print(f"      Total: {n_total} traces loaded, {skipped} skipped")
    print(f"      Time:  {time.time()-t_start:.1f}s")

    # ── Bước 3: Fit scaler + selector trên metadata ──
    print(f"\n[3/5] Fitting scaler + SelectKBest (k={META_DIM_SEL})...")
    meta_arr   = np.array(all_metas,  dtype=np.float32)
    labels_arr = np.array(all_labels, dtype=np.int32)

    scaler = StandardScaler()
    meta_scaled = scaler.fit_transform(meta_arr)

    selector = SelectKBest(score_func=f_classif, k=META_DIM_SEL)
    meta_selected = selector.fit_transform(meta_scaled, labels_arr)

    # Save scaler + selector
    with open(OUT_SCALER, 'wb') as f: pickle.dump(scaler, f)
    with open(OUT_SELECTOR, 'wb') as f: pickle.dump(selector, f)
    print(f"      Saved: {OUT_SCALER}")
    print(f"      Saved: {OUT_SELECTOR}")

    # ── Bước 4: Load model + get embeddings ──
    print(f"\n[4/5] Loading model weights + computing embeddings...")
    print(f"      Building architecture...")
    model = build_model(
        num_classes=len(classes),
        meta_dim=META_DIM_SEL,
        seq_length=SEQ_LENGTH
    )
    model.load_weights(WEIGHTS_PATH)
    print(f"      Weights loaded: {WEIGHTS_PATH}")

    embeddings = []
    n_batches  = (n_total + BATCH_SIZE - 1) // BATCH_SIZE

    t_emb = time.time()
    for i in range(0, n_total, BATCH_SIZE):
        sl = slice(i, i + BATCH_SIZE)

        batch = {
            'dir_input':      np.expand_dims(np.array(all_dirs[sl]),  -1).astype(np.float32),
            'time_input':     np.expand_dims(np.array(all_iats[sl]),  -1).astype(np.float32),
            'size_input':     np.expand_dims(np.array(all_sizes[sl]), -1).astype(np.float32),
            'metadata_input': meta_selected[sl].astype(np.float32),
        }

        _, emb = model.predict(batch, verbose=0)
        embeddings.append(emb)

        batch_idx = i // BATCH_SIZE + 1
        if batch_idx % 20 == 0 or batch_idx == n_batches:
            elapsed  = time.time() - t_emb
            eta      = elapsed / batch_idx * (n_batches - batch_idx)
            print(f"      Batch {batch_idx:4d}/{n_batches}  |  elapsed: {elapsed:.0f}s  |  ETA: {eta:.0f}s")
        gc.collect()

    embeddings = np.concatenate(embeddings, axis=0)   # (N, 128)
    print(f"      Embeddings shape: {embeddings.shape}")

    # ── Bước 5: Tính centroids ──
    print(f"\n[5/5] Computing centroids for {len(classes)} classes...")
    centroids = np.zeros((len(classes), 128), dtype=np.float32)

    for c in range(len(classes)):
        mask = labels_arr == c
        if np.sum(mask) > 0:
            cent = np.mean(embeddings[mask], axis=0)
            cent = cent / (np.linalg.norm(cent) + 1e-8)
            centroids[c] = cent

    np.save(OUT_CENTROIDS, centroids)
    print(f"      Saved: {OUT_CENTROIDS}  shape={centroids.shape}")

    with open(OUT_LABELS, 'w') as f:
        json.dump(label_map, f, indent=2)
    print(f"      Saved: {OUT_LABELS}  ({len(label_map)} classes)")

    print(f"\n{'='*55}")
    print(f"  Done! Total time: {time.time()-t_start:.1f}s")
    print(f"  Flask backend giờ chạy được: python app.py")
    print(f"{'='*55}\n")


if __name__ == '__main__':
    main()