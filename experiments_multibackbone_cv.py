# experiments_multibackbone_cv.py
# -*- coding: utf-8 -*-
import os, json, time, random, warnings
warnings.filterwarnings("ignore")
import os
os.environ["TF_XLA_FLAGS"] = "--tf_xla_auto_jit=0"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras import mixed_precision
mixed_precision.set_global_policy("mixed_float16")

from sklearn.model_selection import train_test_split, StratifiedKFold, GroupKFold
from sklearn.metrics import (
    roc_auc_score, average_precision_score, precision_recall_curve,
    accuracy_score, f1_score, precision_score, recall_score, confusion_matrix,
    brier_score_loss, cohen_kappa_score
)

TRAIN_DIR = "/mnt/c/Users/juman/MLE_projects/mammography_images/train"
CSV_PATH  = "/mnt/c/Users/juman/MLE_projects/mammography_images/Training_set.csv"
OUT_DIR   = "./runs_multibackbone_cv"
SEED = 42


os.makedirs(OUT_DIR, exist_ok=True)
random.seed(SEED); np.random.seed(SEED); tf.keras.utils.set_random_seed(SEED)

from tensorflow.keras.applications import (
    MobileNetV3Large, ResNet50, DenseNet121, EfficientNetB0, EfficientNetB3, EfficientNetB7,
    InceptionV3, MobileNetV2
)
from tensorflow.keras.applications import (
    mobilenet_v3, resnet, densenet, efficientnet, inception_v3, mobilenet_v2
)

BACKBONES = {
    "MobileNetV3Large": dict(model=MobileNetV3Large, input=(224,224), preprocess=mobilenet_v3.preprocess_input),
    "MobileNetV2":      dict(model=MobileNetV2,      input=(224,224), preprocess=mobilenet_v2.preprocess_input),
    "ResNet50":         dict(model=ResNet50,         input=(224,224), preprocess=resnet.preprocess_input),
    "DenseNet121":      dict(model=DenseNet121,      input=(224,224), preprocess=densenet.preprocess_input),
    "EfficientNetB0":   dict(model=EfficientNetB0,   input=(224,224), preprocess=efficientnet.preprocess_input),
    "EfficientNetB3":   dict(model=EfficientNetB3,   input=(300,300), preprocess=efficientnet.preprocess_input),
    "EfficientNetB7":   dict(model=EfficientNetB7,   input=(600,600), preprocess=efficientnet.preprocess_input),
    "InceptionV3":      dict(model=InceptionV3,      input=(299,299), preprocess=inception_v3.preprocess_input),
}

CFG = dict(
    border_px=80,          
    batch=32,             
    epochs_stage1=8,
    epochs_stage2=5,
    unfreeze_ratio=0.25,
    lr1=1e-4, wd1=1e-2,
    lr2=5e-5, wd2=5e-3,
    label_smoothing=0.05,
    tta_eval=True,         
    use_attention=True,    
    screen_holdout=True,  
    do_cv=False,          
    n_folds=5,
    group_by_patient=True,
    TEST_SIZE=0.20,
    VAL_SIZE=0.20,
)

# ============ small utils ============

def _ece_score(y_true, y_prob, n_bins=15):
    """Expected Calibration Error (binary)."""
    bins = np.linspace(0, 1, n_bins+1)
    ids  = np.digitize(y_prob, bins) - 1
    ece, m = 0.0, len(y_true)
    for b in range(n_bins):
        mask = ids == b
        if not np.any(mask):
            continue
        conf = y_prob[mask].mean()
        acc  = (y_true[mask] == (y_prob[mask] >= 0.5)).mean()
        w    = mask.mean()
        ece += w * abs(acc - conf)
    return float(ece)

def _bin_metrics(y_true_bin, y_prob, thr=None):
    """Подбор лучшего порога по F1 и вычисление пакета метрик (binary)."""
    y_true_bin = np.asarray(y_true_bin).astype(int)
    y_prob     = np.asarray(y_prob).astype(float)

    P,R,T = precision_recall_curve(y_true_bin, y_prob)
    F1    = 2*P*R/(P+R+1e-9)
    best_i = int(np.argmax(F1[:-1])) if len(T) else 0
    best_thr = float(T[best_i]) if thr is None and len(T) else (0.5 if thr is None else float(thr))

    y_pred = (y_prob >= best_thr).astype(int)

    tn, fp, fn, tp = confusion_matrix(y_true_bin, y_pred).ravel()
    sens = tp / max(tp+fn, 1)  # recall
    spec = tn / max(tn+fp, 1)
    bal_acc = 0.5*(sens+spec)

    out = dict(
        roc_auc = float(roc_auc_score(y_true_bin, y_prob)),
        pr_auc  = float(average_precision_score(y_true_bin, y_prob)),
        f1_best = float(F1[best_i] if len(F1) else f1_score(y_true_bin, y_pred, zero_division=0)),
        thr_best = best_thr,
        acc    = float(accuracy_score(y_true_bin, y_pred)),
        prec   = float(precision_score(y_true_bin, y_pred, zero_division=0)),
        recall = float(recall_score(y_true_bin, y_pred)),
        sens   = float(sens),            # = recall
        spec   = float(spec),
        bal_acc= float(bal_acc),
        brier  = float(brier_score_loss(y_true_bin, y_prob)),
        ece    = _ece_score(y_true_bin, y_prob, n_bins=15),
        tn=int(tn), fp=int(fp), fn=int(fn), tp=int(tp),
    )
    return out

def _density_metrics(y_true_1to4, y_pred_1to4, p_proba=None):
    """Расширенные метрики 4-класса плотности. Классы 1..4, NaN пропускаем."""
    y_true = np.asarray(y_true_1to4, dtype=float)
    mask = np.isfinite(y_true)
    if not mask.any():
        return dict(
            acc=np.nan, precision_macro=np.nan, recall_macro=np.nan, f1_macro=np.nan,
            precision_micro=np.nan, recall_micro=np.nan, f1_micro=np.nan,
            kappa=np.nan, auc_macro_ovr=np.nan, n_valid=0, cm=[[0]*4 for _ in range(4)]
        )
    t = y_true[mask].astype(int)
    p = np.asarray(y_pred_1to4)[mask].astype(int)

    acc = accuracy_score(t, p)
    prec_macro = precision_score(t, p, average="macro", zero_division=0)
    rec_macro  = recall_score(t, p,    average="macro", zero_division=0)
    f1_macro   = f1_score(t, p,       average="macro", zero_division=0)
    prec_micro = precision_score(t, p, average="micro", zero_division=0)
    rec_micro  = recall_score(t, p,    average="micro", zero_division=0)
    f1_micro   = f1_score(t, p,       average="micro", zero_division=0)
    kappa = cohen_kappa_score(t, p)
    cm = confusion_matrix(t, p, labels=[1,2,3,4])

    auc_macro_ovr = None
    if p_proba is not None:
        from sklearn.preprocessing import label_binarize
        y_bin = label_binarize(t, classes=[1,2,3,4])
        try:
            auc_macro_ovr = roc_auc_score(y_bin, np.asarray(p_proba)[mask], multi_class="ovr", average="macro")
        except Exception:
            auc_macro_ovr = None

    return dict(
        acc=float(acc),
        precision_macro=float(prec_macro), recall_macro=float(rec_macro), f1_macro=float(f1_macro),
        precision_micro=float(prec_micro), recall_micro=float(rec_micro), f1_micro=float(f1_micro),
        kappa=float(kappa), auc_macro_ovr=(None if auc_macro_ovr is None else float(auc_macro_ovr)),
        n_valid=int(mask.sum()), cm=cm.tolist()
    )

# ============ data utils ============

def load_df(csv_path):
    df = pd.read_csv(csv_path)
    df["pathology_binary"] = df["label"].str.contains("Malignant").map({True:"Malignant", False:"Benign"})
    dens = df["label"].str.extract(r"Density(\d)").astype(float)
    df["density_numeric"] = dens[0]
    return df

def split_holdout(df, test_size=0.2, val_size=0.2, seed=SEED, group_by_patient=True):
    if group_by_patient and "patient_id" in df.columns:
        gkf = GroupKFold(n_splits=int(1/test_size))
        (trainval_idx, test_idx) = list(gkf.split(df, df["pathology_binary"], df["patient_id"]))[0]
        trainval, test_df = df.iloc[trainval_idx], df.iloc[test_idx]
    else:
        trainval, test_df = train_test_split(df, test_size=test_size,
                                             stratify=df["pathology_binary"], random_state=seed)
    train_df, val_df = train_test_split(trainval, test_size=val_size,
                                        stratify=trainval["pathology_binary"], random_state=seed)
    return train_df.reset_index(drop=True), val_df.reset_index(drop=True), test_df.reset_index(drop=True)

def build_tfds(df, cfg, preprocess, batch, augment, shuffle):
    inp = cfg['input']
    paths = (TRAIN_DIR + "/" + df["filename"]).values
    y_path = (df["pathology_binary"]=="Malignant").astype("int32").values
    y_dens = df["density_numeric"].astype("float32").values  # NaN если нет

    ds = tf.data.Dataset.from_tensor_slices((paths, y_path, y_dens))
    if shuffle:
        ds = ds.shuffle(len(df), seed=SEED, reshuffle_each_iteration=True)

    def _load(p, yp, yd):
        img = tf.io.read_file(p)
        img = tf.image.decode_jpeg(img, channels=3)
        img = tf.image.resize(img, inp, antialias=True)
        # border shave
        if CFG["border_px"]>0 and CFG["border_px"]*2 < inp[0] and CFG["border_px"]*2 < inp[1]:
            b = CFG["border_px"]
            img = tf.image.crop_to_bounding_box(img, b, b, inp[0]-2*b, inp[1]-2*b)
            img = tf.image.resize(img, inp, antialias=True)
        if augment:
            img = tf.image.random_flip_left_right(img)
            img = tf.image.random_brightness(img, 0.1)
            img = tf.image.random_contrast(img, 0.9, 1.1)
        img = preprocess(img)

        yd_int = tf.cast(tf.where(tf.math.is_finite(yd), yd-1, -1), tf.int32)  # 0..3 или -1
        dens_oh = tf.where(tf.equal(yd_int, -1),
                           tf.zeros((4,), tf.float32),
                           tf.one_hot(yd_int, 4, dtype=tf.float32))

        labels = {"pathology_output": tf.cast(yp, tf.float32),
                  "density_output": dens_oh}
        sample_w = {
            "pathology_output": tf.constant(1.0, tf.float32),           
            "density_output":   tf.where(tf.equal(yd_int,-1), 0.0, 1.0) # маска отсутствующих плотностей
        }
        return img, labels, sample_w

    ds = ds.map(_load, num_parallel_calls=tf.data.AUTOTUNE).batch(batch).prefetch(tf.data.AUTOTUNE)
    return ds

# ============ model ============

def spatial_attention_block(feat, name="sa"):
    x = layers.Conv2D(64, 3, padding="same", activation="relu", name=f"{name}_c1")(feat)
    x = layers.Conv2D(32, 3, padding="same", activation="relu", name=f"{name}_c2")(x)
    att = layers.Conv2D(1, 1, activation="sigmoid", name=f"{name}_mask")(x)
    out = layers.Multiply(name=f"{name}_mul")([feat, att])
    return out, att

def build_model(backbone_name, use_attention=True):
    cfg = BACKBONES[backbone_name]
    inp_size = cfg['input']; preprocess = cfg['preprocess']

    inputs = layers.Input(shape=(*inp_size, 3))
    try:
        bb = cfg['model'](weights='imagenet', include_top=False, input_shape=(*inp_size,3))
    except Exception:
        bb = cfg['model'](weights=None, include_top=False, input_shape=(*inp_size,3))
    bb.trainable = False

    feat = bb(inputs)
    att_map = None
    if use_attention:
        feat, att_map = spatial_attention_block(feat, name="sa")

    gap = layers.GlobalAveragePooling2D()(feat)
    x = layers.Dense(512, activation='relu')(gap); x = layers.BatchNormalization()(x); x = layers.Dropout(0.5)(x)
    x = layers.Dense(256, activation='relu')(x);  x = layers.BatchNormalization()(x); x = layers.Dropout(0.3)(x)

    p = layers.Dense(128, activation='relu')(x); p = layers.Dropout(0.25)(p)
    pathology_output = layers.Dense(1, activation='sigmoid', dtype='float32', name='pathology_output')(p)

    d = layers.Dense(128, activation='relu')(x); d = layers.Dropout(0.25)(d)
    density_output = layers.Dense(4, activation='softmax', dtype='float32', name='density_output')(d)

    model = models.Model(inputs, {"pathology_output": pathology_output,
                                  "density_output": density_output},
                         name=f"{backbone_name}_multitask")

    att_extractor = None
    if use_attention and att_map is not None:
        att_extractor = tf.keras.Model(inputs, att_map, name="attention_extractor")
    return model, cfg, att_extractor

def compile_model(model, lr, wd, label_smoothing, use_attention=True):
    model.compile(
        optimizer=tf.keras.optimizers.AdamW(learning_rate=lr, weight_decay=wd),
        loss={
            "pathology_output": "binary_crossentropy",
            "density_output": tf.keras.losses.CategoricalCrossentropy(label_smoothing=label_smoothing),
        },
        loss_weights={"pathology_output": 2.0, "density_output": 0.8},
        metrics={
            "pathology_output": [tf.keras.metrics.AUC(name="auc"), "accuracy"],
            "density_output":  ["accuracy"],
        },
    )

# ============ TTA util ============

def predict_tta_paths(model, paths, inp_size, preprocess):
    # id + hflip
    X = []
    for p in paths:
        im = tf.keras.utils.load_img(os.path.join(TRAIN_DIR, p))
        im = tf.keras.utils.img_to_array(im)
        im = tf.image.resize(im, inp_size, antialias=True).numpy()
        if CFG["border_px"]>0 and CFG["border_px"]*2 < inp_size[0] and CFG["border_px"]*2 < inp_size[1]:
            b = CFG["border_px"]; im = im[b:-b, b:-b]; im = tf.image.resize(im, inp_size, antialias=True).numpy()
        im = preprocess(im)
        X.append(im)
    X = np.stack(X, 0)
    outs = []
    for v in ["id","flip"]:
        Xv = X if v=="id" else X[:, :, ::-1, :]
        outs.append(model.predict(Xv, verbose=0))
    p_path = np.mean([o["pathology_output"] for o in outs], axis=0).ravel().astype("float32")
    p_dens = np.mean([o["density_output"]   for o in outs], axis=0).astype("float32")
    return p_path, p_dens

# ============ experiments ============

def holdout_experiment(backbone_name: str, df: pd.DataFrame):
    t0 = time.time()
    model, cfg, _ = build_model(backbone_name, use_attention=CFG["use_attention"])
    preprocess = cfg["preprocess"]; inp = cfg["input"]
    batch = CFG["batch"] if inp[0] < 600 else max(8, CFG["batch"] // 2)

    train_df, val_df, test_df = split_holdout(
        df, test_size=CFG["TEST_SIZE"], val_size=CFG["VAL_SIZE"],
        seed=SEED, group_by_patient=CFG["group_by_patient"]
    )

    # datasets
    train_ds = build_tfds(train_df, cfg, preprocess, batch, augment=True,  shuffle=True)
    val_ds   = build_tfds(val_df,   cfg, preprocess, batch, augment=False, shuffle=False)
    test_ds  = build_tfds(test_df,  cfg, preprocess, batch, augment=False, shuffle=False)

    overlap_train_val  = set(train_df["filename"]) & set(val_df["filename"])
    overlap_train_test = set(train_df["filename"]) & set(test_df["filename"])
    overlap_val_test   = set(val_df["filename"])   & set(test_df["filename"])

    print(f"[Overlap] Train∩Val:  {len(overlap_train_val)}")
    print(f"[Overlap] Train∩Test: {len(overlap_train_test)}")
    print(f"[Overlap] Val∩Test:   {len(overlap_val_test)}")

    if overlap_train_val or overlap_train_test or overlap_val_test:
        print("⚠️ WARNING: leakage detected (same filenames in multiple splits)!")
    else:
        print("✅ No filename overlap between splits.")

    import sys
    sys.exit()   
    compile_model(model, CFG["lr1"], CFG["wd1"], CFG["label_smoothing"], use_attention=CFG["use_attention"])
    cbs = [
        tf.keras.callbacks.EarlyStopping(monitor="val_pathology_output_auc", mode="max",
                                         patience=4, restore_best_weights=True, verbose=1),
        tf.keras.callbacks.ReduceLROnPlateau(monitor="val_pathology_output_auc", mode="max",
                                             factor=0.5, patience=2, verbose=1),
    ]
    # stage-1
    model.fit(train_ds, validation_data=val_ds, epochs=CFG["epochs_stage1"], verbose=1, callbacks=cbs)
    # stage-2 — частичный unfreeze
    if CFG["epochs_stage2"] > 0 and CFG["unfreeze_ratio"] > 0:
        bb = model.layers[1]
        if hasattr(bb, "layers") and len(bb.layers) > 0:
            n = len(bb.layers); unfreeze_from = int(n * (1 - CFG["unfreeze_ratio"]))
            for i, l in enumerate(bb.layers):
                l.trainable = (i >= unfreeze_from) and not isinstance(l, tf.keras.layers.BatchNormalization)
            compile_model(model, CFG["lr2"], CFG["wd2"], CFG["label_smoothing"], use_attention=CFG["use_attention"])
            model.fit(train_ds, validation_data=val_ds, epochs=CFG["epochs_stage2"], verbose=1, callbacks=cbs)

    # ── валид: порог по лучшему F1
    if CFG["tta_eval"]:
        p_val, _ = predict_tta_paths(model, val_df["filename"].tolist(), inp, preprocess)
    else:
        p_val = model.predict(val_ds, verbose=0)["pathology_output"].ravel().astype("float32")
    y_val = (val_df["pathology_binary"]=="Malignant").astype(int).values
    val_pack = _bin_metrics(y_val, p_val)
    thr_star = val_pack["thr_best"]

    if CFG["tta_eval"]:
        p_test, p_dens = predict_tta_paths(model, test_df["filename"].tolist(), inp, preprocess)
    else:
        pred = model.predict(test_ds, verbose=0)
        p_test = pred["pathology_output"].ravel().astype("float32")
        p_dens = pred["density_output"]

    y_test = (test_df["pathology_binary"]=="Malignant").astype(int).values
    test_pack = _bin_metrics(y_test, p_test, thr=thr_star) 

    dens_true = test_df["density_numeric"].astype(float).values
    dens_pred = p_dens.argmax(1)+1
    dens_pack = _density_metrics(dens_true, dens_pred, p_proba=p_dens)

    time_min = (time.time() - t0) / 60.0

    out = {
        "backbone": backbone_name,
        "input_size": str(inp),
        "params": int(model.count_params()),
        "time_min": float(time_min),

        "test_auc": test_pack["roc_auc"],
        "test_pr_auc": test_pack["pr_auc"],
        "test_brier": test_pack["brier"],
        "test_ece": test_pack["ece"],
        "test_path_acc": test_pack["acc"],         
        "test_acc@bestF1": test_pack["acc"],
        "test_f1@bestF1": test_pack["f1_best"],
        "test_prec@bestF1": test_pack["prec"],
        "test_rec@bestF1": test_pack["recall"],
        "test_sens": test_pack["sens"],
        "test_spec": test_pack["spec"],
        "test_bal_acc": test_pack["bal_acc"],
        "thr_star": test_pack["thr_best"],
        "tp": test_pack["tp"], "fp": test_pack["fp"], "tn": test_pack["tn"], "fn": test_pack["fn"],

        # density
        "test_density_acc": dens_pack["acc"],
        "dens_f1_macro": dens_pack["f1_macro"],
        "dens_f1_micro": dens_pack["f1_micro"],
        "dens_precision_macro": dens_pack["precision_macro"],
        "dens_recall_macro": dens_pack["recall_macro"],
        "dens_kappa": dens_pack["kappa"],
        "dens_auc_macro_ovr": dens_pack["auc_macro_ovr"],
        "dens_n_valid": dens_pack["n_valid"],

        "model": model,
    }
    return out

# ---- plotting for CV ----
import matplotlib.pyplot as plt
def _boxplot_and_save(values_by_name: dict, title: str, out_png: str):
    plt.figure(figsize=(7,4))
    data = [values_by_name[k] for k in values_by_name]
    labels = list(values_by_name.keys())
    plt.boxplot(data, labels=labels, showmeans=True)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_png, dpi=180)
    plt.close()

def cv_experiment(backbone_name, df, n_folds=5):
    results = []
    use_group = CFG["group_by_patient"] and "patient_id" in df.columns
    if use_group:
        splitter = GroupKFold(n_splits=n_folds)
        splits = splitter.split(df, df["pathology_binary"], df["patient_id"])
    else:
        splitter = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=SEED)
        splits = splitter.split(df, df["pathology_binary"])

    cfg_b = BACKBONES[backbone_name]
    preprocess = cfg_b["preprocess"]; inp = cfg_b["input"]
    batch = CFG["batch"] if inp[0] < 600 else max(8, CFG["batch"]//2)

    fold = 0
    for tr_idx, te_idx in splits:
        fold += 1
        trainval = df.iloc[tr_idx].reset_index(drop=True)
        test_df  = df.iloc[te_idx].reset_index(drop=True)
        train_df, val_df = train_test_split(trainval, test_size=0.2, stratify=trainval["pathology_binary"], random_state=SEED)

        model, _, _ = build_model(backbone_name, use_attention=CFG["use_attention"])

        train_ds = build_tfds(train_df, cfg_b, preprocess, batch, augment=True,  shuffle=True)
        val_ds   = build_tfds(val_df,   cfg_b, preprocess, batch, augment=False, shuffle=False)
        test_ds  = build_tfds(test_df,  cfg_b, preprocess, batch, augment=False, shuffle=False)

        compile_model(model, CFG["lr1"], CFG["wd1"], CFG["label_smoothing"], use_attention=CFG["use_attention"])
        cbs = [
            tf.keras.callbacks.EarlyStopping(monitor='val_pathology_output_auc', mode='max', patience=4, restore_best_weights=True, verbose=0),
            tf.keras.callbacks.ReduceLROnPlateau(monitor='val_pathology_output_auc', mode='max', factor=0.5, patience=2, verbose=0),
        ]
        model.fit(train_ds, validation_data=val_ds, epochs=CFG["epochs_stage1"], verbose=0, callbacks=cbs)

        bb = model.layers[1]
        if hasattr(bb, "layers"):
            n = len(bb.layers); unfreeze_from = int(n*(1-CFG["unfreeze_ratio"]))
            for i,l in enumerate(bb.layers):
                l.trainable = (i >= unfreeze_from) and not isinstance(l, tf.keras.layers.BatchNormalization)
            compile_model(model, CFG["lr2"], CFG["wd2"], CFG["label_smoothing"], use_attention=CFG["use_attention"])
            model.fit(train_ds, validation_data=val_ds, epochs=CFG["epochs_stage2"], verbose=0, callbacks=cbs)

        val_paths = val_df["filename"].tolist()
        if CFG["tta_eval"]:
            p_val, _ = predict_tta_paths(model, val_paths, inp, preprocess)
        else:
            p_val = model.predict(val_ds, verbose=0)["pathology_output"].ravel().astype("float32")
        y_val = (val_df["pathology_binary"]=="Malignant").astype(int).values
        P,R,T = precision_recall_curve(y_val, p_val); f1 = 2*P*R/(P+R+1e-9); thr = float(T[np.argmax(f1[:-1])]) if len(T) else 0.5

        # test
        test_paths = test_df["filename"].tolist()
        if CFG["tta_eval"]:
            p_test, p_dens = predict_tta_paths(model, test_paths, inp, preprocess)
        else:
            pred = model.predict(test_ds, verbose=0); p_test = pred["pathology_output"].ravel().astype("float32"); p_dens = pred["density_output"]
        y_test = (test_df["pathology_binary"]=="Malignant").astype(int).values

        pack = _bin_metrics(y_test, p_test, thr=thr)

        # density metrics
        dens_true = test_df["density_numeric"].astype(float).values
        dens_pred = p_dens.argmax(1)+1
        dens_pack = _density_metrics(dens_true, dens_pred, p_proba=p_dens)

        results.append(dict(
            fold=fold,
            path_auc=pack["roc_auc"], path_pr_auc=pack["pr_auc"],
            path_acc=pack["acc"], path_f1=pack["f1_best"],
            path_precision=pack["prec"], path_recall=pack["recall"],
            path_specificity=pack["spec"], path_bal_acc=pack["bal_acc"], path_brier=pack["brier"], path_ece=pack["ece"],
            thr=pack["thr_best"], tp=pack["tp"], fp=pack["fp"], tn=pack["tn"], fn=pack["fn"],
            dens_acc=dens_pack["acc"], dens_f1_macro=dens_pack["f1_macro"], dens_f1_micro=dens_pack["f1_micro"],
            dens_precision_macro=dens_pack["precision_macro"], dens_recall_macro=dens_pack["recall_macro"],
            dens_kappa=dens_pack["kappa"], dens_auc_macro_ovr=dens_pack["auc_macro_ovr"], dens_n_valid=dens_pack["n_valid"]
        ))
        tf.keras.backend.clear_session()

    df_folds = pd.DataFrame(results)
    cv_dir = os.path.join(OUT_DIR, f"{backbone_name}_cv"); os.makedirs(cv_dir, exist_ok=True)
    df_folds.to_csv(os.path.join(cv_dir, "fold_metrics.csv"), index=False)

    summary = {
        "backbone": backbone_name, "n_folds": n_folds,
        "path_auc_mean": df_folds["path_auc"].mean(), "path_auc_std": df_folds["path_auc"].std(ddof=1),
        "path_acc_mean": df_folds["path_acc"].mean(), "path_acc_std": df_folds["path_acc"].std(ddof=1),
        "path_f1_mean":  df_folds["path_f1"].mean(),  "path_f1_std":  df_folds["path_f1"].std(ddof=1),
        "dens_acc_mean": df_folds["dens_acc"].mean(), "dens_acc_std": df_folds["dens_acc"].std(ddof=1),
        "dens_f1_macro_mean": df_folds["dens_f1_macro"].mean(),
        "dens_f1_macro_std":  df_folds["dens_f1_macro"].std(ddof=1),
        "dens_kappa_mean": df_folds["dens_kappa"].mean(),
        "dens_kappa_std":  df_folds["dens_kappa"].std(ddof=1),
    }
    pd.DataFrame([summary]).to_csv(os.path.join(cv_dir, "summary.csv"), index=False)

    _boxplot_and_save(
        {"AUC": df_folds["path_auc"].tolist(),
         "ACC": df_folds["path_acc"].tolist(),
         "F1":  df_folds["path_f1"].tolist()},
        title=f"{backbone_name} – Pathology (5-fold)",
        out_png=os.path.join(cv_dir, "pathology_boxplot.png")
    )
    _boxplot_and_save(
        {"ACC": df_folds["dens_acc"].tolist(),
         "F1-macro": df_folds["dens_f1_macro"].tolist()},
        title=f"{backbone_name} – Density (5-fold)",
        out_png=os.path.join(cv_dir, "density_boxplot.png")
    )
    return summary

if __name__ == "__main__":
    df = load_df(CSV_PATH)

    if CFG["screen_holdout"]:
        order = ["MobileNetV3Large","ResNet50","DenseNet121","EfficientNetB0","EfficientNetB3","EfficientNetB7"]
        leaderboard = []
        for b in order:
            print("\n"+"="*80); print("SCREENING:", b); print("="*80)
            res = holdout_experiment(b, df)
    
            leaderboard.append({k:v for k,v in res.items() if k not in ("model",)})
            tf.keras.backend.clear_session()

        lb = pd.DataFrame(leaderboard).sort_values("test_auc", ascending=False)
        lb_path = os.path.join(OUT_DIR, "leaderboard_holdout.csv")
        lb.to_csv(lb_path, index=False)
        print("\nLEADERBOARD (hold-out) saved ->", lb_path)
        cols_show = ["backbone","test_auc","test_path_acc","dens_f1_macro","test_density_acc","params","input_size","time_min"]
        cols_show = [c for c in cols_show if c in lb.columns]
        print(lb[cols_show].to_string(index=False))

    if CFG["do_cv"]:
        backbones_for_cv = ["EfficientNetB7","DenseNet121","MobileNetV3Large"]
        cv_rows = []
        for b in backbones_for_cv:
            print("\n"+"#"*80); print("CV:", b); print("#"*80)
            out = cv_experiment(b, df, n_folds=CFG["n_folds"])
            cv_rows.append(out)
        cv_sum = pd.DataFrame(cv_rows)
        cv_sum_path = os.path.join(OUT_DIR, "cv_summary.csv")
        cv_sum.to_csv(cv_sum_path, index=False)
        print("\nCV SUMMARY saved ->", cv_sum_path)
        print(cv_sum[["backbone","path_auc_mean","path_auc_std","path_acc_mean","path_acc_std","dens_acc_mean","dens_acc_std"]].to_string(index=False))
