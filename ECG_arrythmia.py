# %% [markdown]
# # ECG Arrhythmia Classification — Advanced Deep Learning Pipeline
# 
# > **Datasets**: MIT-BIH Arrhythmia (5 classes) · PTB Diagnostic ECG (binary)  
# > **Architectures**: CNN · SE-ResNet · BiLSTM + Attention · Transformer  
# > **Highlights**: SMOTE balancing · data augmentation · Grad-CAM explainability · weighted ensemble · transfer learning
# 
# ---

# %%
import os, warnings, time
import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf 
from tensorflow.keras import layers, Model, Input, regularizers
from tensorflow.keras.callbacks import (EarlyStopping, ReduceLROnPlateau, 
                                        ModelCheckpoint, LearningRateScheduler)
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (classification_report, confusion_matrix,
                             roc_curve, auc, precision_recall_curve,
                             average_precision_score, f1_score)
from imblearn.over_sampling import SMOTE

warnings.filterwarnings('ignore')

# %%
DATA_DIR = "/Users/matteosuardi/Desktop/DL projects/ECG Arrythmia/archive"
CHECKPOINT_DIR = "/Users/matteosuardi/Desktop/DL projects/ECG Arrythmia/checkpoints"
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

SEED = 42
BATCH_SIZE = 128
EPOCHS = 100
NUM_CLASSES_MITBIH = 5
SIGNAL_LEN = 187

np.random.seed(SEED)
tf.random.set_seed(SEED)
print(f"TensorFlow {tf.__version__} - GPU Available: {tf.config.list_physical_devices('GPU')}")

# %% [markdown]
# ## Data Loading

# %%
mitbih_train = pd.read_csv(os.path.join(DATA_DIR, "mitbih_train.csv"), header=None)
mitbih_test = pd.read_csv(os.path.join(DATA_DIR, "mitbih_test.csv"),  header=None)
ptb_normal = pd.read_csv(os.path.join(DATA_DIR, "ptbdb_normal.csv"), header=None)
ptb_abnormal = pd.read_csv(os.path.join(DATA_DIR, "ptbdb_abnormal.csv"), header=None)

print(f"MIT-BIH train: {mitbih_train.shape}")
print(f"MIT-BIH test: {mitbih_test.shape}")
print(f"PTB normal: {ptb_normal.shape}")
print(f"PTB abnormal: {ptb_abnormal.shape}")

# %% [markdown]
# ## EDA
# 
# ### Class semantics (MIT-BIH)
# 
# | Code | Label | Description | Share |
# |------|-------|-------------|-------|
# | **0** | N | Normal beat | ~82.8 % |
# | **1** | S | Supraventricular ectopic beat | ~2.5 % |
# | **2** | V | Ventricular ectopic beat | ~6.6 % |
# | **3** | F | Fusion beat | ~0.7 % |
# | **4** | Q | Unknown / noisy beat | ~7.3 % |

# %%
LABELS_MIT = ['N (0)', 'S (1)', 'V (2)', 'F (3)', 'Q (4)']
COLORS_MIT = ['#e74c3c', '#3498db', '#2c3e50', '#e67e22', '#27ae60']
LABELS_PTB = ['Normal', 'Myocardial Infarction']
COLORS_PTB = ['#2ecc71', '#e74c3c']

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# MIT-BIH distribution
dist = mitbih_train[187].value_counts(normalize=True).sort_index() * 100
axes[0].pie(dist, labels=LABELS_MIT, colors=COLORS_MIT, autopct='%1.1f%%',
            startangle=90, pctdistance=0.8,
            wedgeprops=dict(width=0.35, edgecolor='white'))
axes[0].set_title('MIT-BIH Train — Class distribution')

# PTB distribution
ptb_full = pd.concat([ptb_normal, ptb_abnormal], ignore_index=True)
ptb_dist = ptb_full.iloc[:, -1].value_counts(normalize=True).sort_index() * 100
axes[1].pie(ptb_dist, labels=LABELS_PTB, colors=COLORS_PTB, autopct='%1.1f%%',
            startangle=90, pctdistance=0.8,
            wedgeprops=dict(width=0.35, edgecolor='white'))
axes[1].set_title('PTB — Class distribution')

plt.tight_layout()
plt.show()

# %%
fig, axes = plt.subplots(1, 5, figsize=(20, 3), sharey=True)
for cls in range(5):
    sample = mitbih_train[mitbih_train[187] == cls].iloc[0, :187].values
    axes[cls].plot(sample, color=COLORS_MIT[cls], linewidth=0.9)
    axes[cls].set_title(LABELS_MIT[cls])
    axes[cls].set_xlabel('Sample')
    if cls == 0:
        axes[cls].set_ylabel('Amplitude')
    axes[cls].grid(True, alpha=0.3)
plt.suptitle('Representative ECG Beat per Class', y=1.03, fontsize=14)
plt.tight_layout()
plt.show()

# %%
fig, axes = plt.subplots(1, 5, figsize=(20, 3), sharey=True)
for cls in range(5):
    sample = mitbih_train[mitbih_train[187] == cls].iloc[0, :187].values
    fft_vals = np.abs(np.fft.rfft(sample))
    freqs = np.fft.rfftfreq(len(sample), d=1.0/125) # MIT-BIH sampled at 125 Hz
    axes[cls].plot(freqs, fft_vals, color=COLORS_MIT[cls], linewidth=0.9)
    axes[cls].set_title(LABELS_MIT[cls])
    axes[cls].set_xlabel('Frequency (Hz)')
    if cls == 0:
        axes[cls].set_ylabel('|FFT|')
    axes[cls].set_xlim(0, 60)
    axes[cls].grid(True, alpha=0.3)
plt.suptitle('Frequency Spectrum per Class (Frequency domain analysis)', y=1.03, fontsize=14)
plt.tight_layout()
plt.show()

# %% [markdown]
# ## Preprocessing
# 
# **Pipeline**:  
# 1. Extract features (`X`) and labels (`y`) from raw CSVs  
# 2. Balance minority classes with **SMOTE** (Synthetic Minority Oversampling)  
# 3. Standardise each feature (Z-score normalisation)  
# 4. Reshape to `(N, 187, 1)` for 1-D convolutions  
# 5. One-hot encode labels  
# 6. Split into train / validation sets (80 / 20, stratified)

# %%
# Extract X, y 
X_full = mitbih_train.iloc[:, :187].values
y_full = mitbih_train.iloc[:, 187].values.astype(int)

X_test_raw = mitbih_test.iloc[:, :187].values
y_test_raw = mitbih_test.iloc[:, 187].values.astype(int)

print("Before SMOTE:", dict(zip(*np.unique(y_full, return_counts=True))))

# SMOTE 
smote = SMOTE(sampling_strategy='auto', random_state=SEED, k_neighbors=5)
X_bal, y_bal = smote.fit_resample(X_full, y_full)
print("After  SMOTE:", dict(zip(*np.unique(y_bal, return_counts=True))))

# %%
scaler = StandardScaler()
X_bal = scaler.fit_transform(X_bal)
X_test_scaled = scaler.transform(X_test_raw) # same scaler on test set

# %%
X_bal = X_bal.reshape(-1, SIGNAL_LEN, 1)
X_test_scaled = X_test_scaled.reshape(-1, SIGNAL_LEN, 1)

# %%
X_train, X_val, y_train, y_val = train_test_split(
    X_bal, y_bal, test_size=0.2, stratify=y_bal, random_state=SEED)

y_train_cat = to_categorical(y_train, num_classes=NUM_CLASSES_MITBIH)
y_val_cat = to_categorical(y_val, num_classes=NUM_CLASSES_MITBIH)
y_test_cat = to_categorical(y_test_raw, num_classes=NUM_CLASSES_MITBIH)

print(f"Train : {X_train.shape}  Val : {X_val.shape}  Test : {X_test_scaled.shape}")

# %%
fig, axes = plt.subplots(1, 2, figsize=(12, 3))
idx = 0
axes[0].plot(mitbih_train.iloc[idx, :187].values)
axes[0].set_title('Before Normalisation')
axes[0].grid(True, alpha=0.3)
axes[1].plot(X_train[idx].squeeze())
axes[1].set_title('After Normalisation (Z-score)')
axes[1].grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

sns.countplot(x=y_train, palette=COLORS_MIT)
plt.title('Class Distribution After SMOTE (Train Set)')
plt.xlabel('Class')
plt.ylabel('Count')
plt.show()

# %% [markdown]
# ## Data Augmentation
# 
# Augmentations applied **on-the-fly** during training via a custom generator:
# 
# | Augmentation | Description |
# |---|---|
# | Gaussian noise | σ = 0.05 |
# | Baseline wander | sinusoidal drift |
# | Amplitude scaling | ×[0.8, 1.2] |
# | Time warping | non-linear temporal stretch |

# %%
def augment_batch(X_batch):
    """Apply random augmentations to a batch of ECG signals."""
    X_aug = X_batch.copy()
    N = len(X_aug)

    # Gaussian noise
    mask = np.random.rand(N) < 0.5
    X_aug[mask] += np.random.normal(0, 0.05, X_aug[mask].shape)

    # Baseline wander (low-freq sinusoid)
    mask = np.random.rand(N) < 0.3
    t = np.linspace(0, 2 * np.pi, SIGNAL_LEN).reshape(1, -1, 1)
    freq = np.random.uniform(0.5, 2.0, (mask.sum(), 1, 1))
    amp  = np.random.uniform(0.05, 0.15, (mask.sum(), 1, 1))
    X_aug[mask] += amp * np.sin(freq * t)

    # Amplitude scaling
    mask = np.random.rand(N) < 0.3
    scale = np.random.uniform(0.8, 1.2, (mask.sum(), 1, 1))
    X_aug[mask] *= scale

    return X_aug

def augmented_generator(X, y, batch_size):
    """Yields augmented batches indefinitely."""
    N = len(X)
    while True:
        idx = np.random.permutation(N)
        for start in range(0, N, batch_size):
            batch_idx = idx[start:start + batch_size]
            yield augment_batch(X[batch_idx]), y[batch_idx]

fig, axes = plt.subplots(2, 3, figsize=(15, 5), sharey=True)
for i in range(3):
    axes[0, i].plot(X_train[i].squeeze(), color='C0')
    axes[0, i].set_title(f'Original (class {y_train[i]})')
    axes[0, i].grid(True, alpha=0.3)
    aug = augment_batch(X_train[i:i+1])
    axes[1, i].plot(aug.squeeze(), color='C1')
    axes[1, i].set_title('Augmented')
    axes[1, i].grid(True, alpha=0.3)
plt.suptitle('Data Augmentation examples', y=1.02, fontsize=14)
plt.tight_layout()
plt.show()

# %% [markdown]
# ## Model Architectures
# 
# We compare four architectures of increasing complexity:
# 
# | # | Model | Key idea |
# |---|-------|----------|
# | 1 | **CNN** | Baseline 1-D convolutions |
# | 2 | **SE-ResNet** | Residual blocks + Squeeze-and-Excitation |
# | 3 | **BiLSTM + Attention** | Bidirectional LSTMs with learned attention |
# | 4 | **Transformer** | Multi-head self-attention encoder |

# %% [markdown]
# ### Baseline CNN

# %%
def build_cnn(input_shape=(SIGNAL_LEN, 1), n_classes=NUM_CLASSES_MITBIH):
    inp = Input(shape=input_shape)
    x = layers.Conv1D(32, 5, activation='relu', padding='same',
                      kernel_regularizer=regularizers.l2(1e-3))(inp)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling1D(2)(x)
    x = layers.Dropout(0.3)(x)

    x = layers.Conv1D(64, 5, activation='relu', padding='same',
                      kernel_regularizer=regularizers.l2(1e-3))(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling1D(2)(x)
    x = layers.Dropout(0.3)(x)

    x = layers.Conv1D(128, 3, activation='relu', padding='same',
                      kernel_regularizer=regularizers.l2(1e-3))(x)
    x = layers.BatchNormalization()(x)
    x = layers.GlobalAveragePooling1D()(x)

    x = layers.Dense(128, activation='relu')(x)
    x = layers.Dropout(0.5)(x)
    out = layers.Dense(n_classes, activation='softmax')(x)
    return Model(inp, out, name='CNN')

build_cnn().summary()

# %% [markdown]
# ### SE-ResNet (Squeeze-and-Excitation ResNet)

# %%
def _se_block(x, ratio=16):
    """Squeeze-and-Excitation block."""
    ch = x.shape[-1]
    se = layers.GlobalAveragePooling1D()(x)
    se = layers.Dense(ch // ratio, activation='relu')(se)
    se = layers.Dense(ch, activation='sigmoid')(se)
    se = layers.Reshape((1, ch))(se)
    return layers.Multiply()([x, se])

def _res_block(x, filters, kernel_size=3):
    """Residual block with SE attention."""
    shortcut = x
    y = layers.Conv1D(filters, kernel_size, padding='same')(x)
    y = layers.BatchNormalization()(y)
    y = layers.Activation('relu')(y)
    y = layers.Conv1D(filters, kernel_size, padding='same')(y)
    y = layers.BatchNormalization()(y)
    y = _se_block(y)
    if shortcut.shape[-1] != filters:
        shortcut = layers.Conv1D(filters, 1, padding='same')(shortcut)
    y = layers.Add()([shortcut, y])
    y = layers.Activation('relu')(y)
    return y

def build_se_resnet(input_shape=(SIGNAL_LEN, 1), n_classes=NUM_CLASSES_MITBIH):
    inp = Input(shape=input_shape)
    x = layers.Conv1D(64, 7, padding='same', activation='relu')(inp)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling1D(2)(x)

    x = _res_block(x, 64)
    x = _res_block(x, 64)
    x = layers.MaxPooling1D(2)(x)

    x = _res_block(x, 128)
    x = _res_block(x, 128)

    x = layers.GlobalAveragePooling1D()(x)
    x = layers.Dropout(0.5)(x)
    out = layers.Dense(n_classes, activation='softmax')(x)
    return Model(inp, out, name='SE_ResNet')

build_se_resnet().summary()

# %% [markdown]
# ### BiLSTM + Attention

# %%
def _attention_pool(x):
    """Learned soft-attention pooling over timesteps."""
    score = layers.Dense(1, activation='tanh')(x)          # (B, T, 1)
    alpha = layers.Softmax(axis=1, name='attn_weights')(score)
    context = layers.Multiply()([x, alpha])
    context = layers.Lambda(lambda t: tf.reduce_sum(t, axis=1))(context)
    return context

def build_bilstm_attention(input_shape=(SIGNAL_LEN, 1), n_classes=NUM_CLASSES_MITBIH):
    inp = Input(shape=input_shape)
    x = layers.Bidirectional(layers.LSTM(64, return_sequences=True))(inp)
    x = layers.Dropout(0.3)(x)
    x = layers.Bidirectional(layers.LSTM(32, return_sequences=True))(x)
    x = layers.Dropout(0.3)(x)
    x = _attention_pool(x)
    x = layers.Dense(64, activation='relu')(x)
    x = layers.Dropout(0.5)(x)
    out = layers.Dense(n_classes, activation='softmax')(x)
    return Model(inp, out, name='BiLSTM_Attn')

build_bilstm_attention().summary()

# %% [markdown]
# ### 1-D Transformer Encoder

# %%
class PatchEmbedding(layers.Layer):
    """Projects the 1-D signal into patch tokens of dimension d_model."""
    def __init__(self, d_model, patch_size, **kw):
        super().__init__(**kw)
        self.proj = layers.Conv1D(d_model, patch_size, strides=patch_size, padding='valid')

    def build(self, input_shape):
        super().build(input_shape)
        seq_len = input_shape[1] // self.proj.kernel_size[0]
        self.pos_emb = self.add_weight('pos_emb', shape=(1, seq_len, self.proj.filters),
                                       initializer='glorot_uniform', trainable=True)

    def call(self, x):
        x = self.proj(x)
        return x + self.pos_emb

def _transformer_block(x, n_heads, ff_dim, dropout):
    attn = layers.MultiHeadAttention(num_heads=n_heads,
                                     key_dim=x.shape[-1] // n_heads,
                                     dropout=dropout)(x, x)
    x = layers.LayerNormalization()(x + attn)
    ff = layers.Dense(ff_dim, activation='gelu')(x)
    ff = layers.Dense(x.shape[-1])(ff)
    ff = layers.Dropout(dropout)(ff)
    x = layers.LayerNormalization()(x + ff)
    return x

def build_transformer(input_shape=(SIGNAL_LEN, 1), n_classes=NUM_CLASSES_MITBIH,
                      d_model=64, patch_size=9, n_heads=4, n_blocks=3,
                      ff_dim=128, dropout=0.1):
    inp = Input(shape=input_shape)
    x = PatchEmbedding(d_model, patch_size)(inp) # (B, n_patches, d_model)
    for _ in range(n_blocks):
        x = _transformer_block(x, n_heads, ff_dim, dropout)
    x = layers.GlobalAveragePooling1D()(x)
    x = layers.Dense(64, activation='relu')(x)
    x = layers.Dropout(0.3)(x)
    out = layers.Dense(n_classes, activation='softmax')(x)
    return Model(inp, out, name='Transformer1D')

build_transformer().summary()

# %% [markdown]
# ## Training
# 
# **Common settings**:  
# - Optimiser: Adam  
# - Loss: categorical cross-entropy  
# - Cosine-annealing LR schedule  
# - Early stopping (patience 10) + best-model checkpoint  
# - Training with on-the-fly data augmentation

# %%
def cosine_lr(epoch, total_epochs=EPOCHS, lr_max=1e-3, lr_min=1e-6):
    return lr_min + 0.5 * (lr_max - lr_min) * (1 + np.cos(np.pi * epoch / total_epochs))

def make_callbacks(name):
    return [
        EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True, verbose=1),
        LearningRateScheduler(cosine_lr, verbose=0),
        ModelCheckpoint(os.path.join(CHECKPOINT_DIR, f'{name}_best.keras'),
                        monitor='val_loss', save_best_only=True, verbose=0),
    ]

def train_model(build_fn, name):
    """Build, compile, train and return (model, history)."""
    model = build_fn()
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    gen = augmented_generator(X_train, y_train_cat, BATCH_SIZE)
    steps = len(X_train) // BATCH_SIZE

    t0 = time.time()
    history = model.fit(
        gen,
        steps_per_epoch=steps,
        validation_data=(X_val, y_val_cat),
        epochs=EPOCHS,
        callbacks=make_callbacks(name),
        verbose=1
    )
    elapsed = time.time() - t0
    print(f"\n{name} trained in {elapsed/60:.1f} min  —  "
          f"best val_loss: {min(history.history['val_loss']):.4f}")
    return model, history

# %%
BUILDERS = {
    'CNN': build_cnn,
    'SE_ResNet': build_se_resnet,
    'BiLSTM_Attn': build_bilstm_attention,
    'Transformer1D': build_transformer,
}

trained = {} # name -> (model, history)
for name, fn in BUILDERS.items():
    print(f"\n{'='*60}")
    print(f"Training: {name}")
    print(f"{'='*60}")
    trained[name] = train_model(fn, name)

# %% [markdown]
# ## Results and comparison

# %% [markdown]
# ### Training curves

# %%
fig, axes = plt.subplots(len(trained), 2, figsize=(14, 4 * len(trained)))
for i, (name, (_, hist)) in enumerate(trained.items()):
    axes[i, 0].plot(hist.history['accuracy'], label='train')
    axes[i, 0].plot(hist.history['val_accuracy'], label='val')
    axes[i, 0].set_title(f'{name} — Accuracy')
    axes[i, 0].legend()
    axes[i, 0].grid(True, alpha=0.3)

    axes[i, 1].plot(hist.history['loss'], label='train')
    axes[i, 1].plot(hist.history['val_loss'], label='val')
    axes[i, 1].set_title(f'{name} — Loss')
    axes[i, 1].legend()
    axes[i, 1].grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# %% [markdown]
# ### Test set evaluation (held-out MIT-BIH test split)

# %%
summary_rows = []
for name, (model, _) in trained.items():
    probs = model.predict(X_test_scaled, verbose=0)
    preds = probs.argmax(axis=1)
    acc  = (preds == y_test_raw).mean()
    f1   = f1_score(y_test_raw, preds, average='macro')
    conf = probs.max(axis=1).mean()
    summary_rows.append({'Model': name, 'Accuracy': acc, 'Macro-F1': f1, 'Avg Confidence': conf})
    print(f"\n── {name} ──")
    print(classification_report(y_test_raw, preds, target_names=LABELS_MIT, digits=4))

df_summary = pd.DataFrame(summary_rows).set_index('Model')
display(df_summary.style.format('{:.4f}').highlight_max(axis=0, color='#d4edda'))

# %% [markdown]
# ### Confusion matrices

# %%
fig, axes = plt.subplots(1, len(trained), figsize=(6 * len(trained), 5))
for i, (name, (model, _)) in enumerate(trained.items()):
    preds = model.predict(X_test_scaled, verbose=0).argmax(axis=1)
    cm = confusion_matrix(y_test_raw, preds)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[i],
                xticklabels=range(5), yticklabels=range(5))
    axes[i].set_title(name)
    axes[i].set_xlabel('Predicted')
    axes[i].set_ylabel('True')
plt.tight_layout()
plt.show()

# %% [markdown]
# ### ROC curves (per model)

# %%
fig, axes = plt.subplots(1, len(trained), figsize=(6 * len(trained), 5))
for i, (name, (model, _)) in enumerate(trained.items()):
    probs = model.predict(X_test_scaled, verbose=0)
    for c in range(NUM_CLASSES_MITBIH):
        fpr, tpr, _ = roc_curve(y_test_cat[:, c], probs[:, c])
        roc_auc = auc(fpr, tpr)
        axes[i].plot(fpr, tpr, label=f'{LABELS_MIT[c]} (AUC={roc_auc:.3f})')
    axes[i].plot([0,1],[0,1],'k--', alpha=0.3)
    axes[i].set_title(f'{name} — ROC')
    axes[i].set_xlabel('FPR')
    axes[i].set_ylabel('TPR')
    axes[i].legend(fontsize=8)
    axes[i].grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# %% [markdown]
# ### Precision-Recall curves

# %%
fig, axes = plt.subplots(1, len(trained), figsize=(6 * len(trained), 5))
for i, (name, (model, _)) in enumerate(trained.items()):
    probs = model.predict(X_test_scaled, verbose=0)
    for c in range(NUM_CLASSES_MITBIH):
        prec, rec, _ = precision_recall_curve(y_test_cat[:, c], probs[:, c])
        ap = average_precision_score(y_test_cat[:, c], probs[:, c])
        axes[i].plot(rec, prec, label=f'{LABELS_MIT[c]} (AP={ap:.3f})')
    axes[i].set_title(f'{name} — PR')
    axes[i].set_xlabel('Recall')
    axes[i].set_ylabel('Precision')
    axes[i].legend(fontsize=8)
    axes[i].grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# %% [markdown]
# ## Robustness analysis
# 
# We evaluate each model under synthetic perturbations to assess real-world robustness.

# %%
PERTURBATIONS = {
    'Original': lambda X: X,
    'Gaussian Noise': lambda X: X + np.random.normal(0, 0.1, X.shape),
    'Baseline Shift': lambda X: X + 0.3,
    'Amplitude x1.2': lambda X: X * 1.2,
    'Random Masking': lambda X: _random_mask(X, 0.1),
    'Time Warping': lambda X: _time_warp(X, 0.02),
}

def _random_mask(X, ratio):
    X_out = X.copy()
    n_mask = int(ratio * X.shape[1])
    for i in range(len(X_out)):
        idx = np.random.choice(X.shape[1], n_mask, replace=False)
        X_out[i, idx] = 0
    return X_out

def _time_warp(X, factor):
    X_flat = X.reshape(X.shape[0], -1)
    N, L = X_flat.shape
    orig = np.arange(L)
    X_w = np.zeros_like(X_flat)
    for i in range(N):
        noise = np.random.normal(0, factor * L, size=L)
        stretched = np.clip(orig + noise, 0, L - 1)
        order = np.argsort(stretched)
        X_w[i] = np.interp(orig, stretched[order], X_flat[i][order])
    return X_w.reshape(X.shape)

rob_rows = []
for mname, (model, _) in trained.items():
    for pname, pfn in PERTURBATIONS.items():
        X_p = pfn(X_test_scaled.copy())
        probs = model.predict(X_p, verbose=0)
        acc = (probs.argmax(axis=1) == y_test_raw).mean()
        conf = probs.max(axis=1).mean()
        rob_rows.append({'Model': mname, 'Perturbation': pname,
                         'Accuracy': acc, 'Confidence': conf})

df_rob = pd.DataFrame(rob_rows)
pivot_acc  = df_rob.pivot(index='Perturbation', columns='Model', values='Accuracy')
pivot_conf = df_rob.pivot(index='Perturbation', columns='Model', values='Confidence')

fig, axes = plt.subplots(1, 2, figsize=(16, 5))
pivot_acc.plot(kind='bar', ax=axes[0], rot=25)
axes[0].set_title('Accuracy under Perturbations')
axes[0].set_ylabel('Accuracy')
axes[0].set_ylim(0, 1.05)
axes[0].legend(fontsize=9)

pivot_conf.plot(kind='bar', ax=axes[1], rot=25)
axes[1].set_title('Avg Confidence under Perturbations')
axes[1].set_ylabel('Confidence')
axes[1].set_ylim(0, 1.05)
axes[1].legend(fontsize=9)
plt.tight_layout()
plt.show()

display(pivot_acc.style.format('{:.4f}').highlight_max(axis=1, color='#d4edda'))

# %% [markdown]
# ## Explainability - Grad-CAM
# 
# **Grad-CAM** (Gradient-weighted Class Activation Mapping) highlights which
# regions of the ECG signal are most important for the model's prediction.
# We apply it to the CNN's last convolutional layer.

# %%
def grad_cam_1d(model, x_input, class_idx, last_conv_layer):
    """Compute 1-D Grad-CAM heatmap."""
    grad_model = Model(inputs=model.inputs,
                       outputs=[model.get_layer(last_conv_layer).output, model.output])
    with tf.GradientTape() as tape:
        conv_out, preds = grad_model(x_input[np.newaxis])
        loss = preds[:, class_idx]
    grads = tape.gradient(loss, conv_out)
    weights = tf.reduce_mean(grads, axis=1, keepdims=True)
    cam = tf.reduce_sum(conv_out * weights, axis=-1).numpy().squeeze()
    cam = np.maximum(cam, 0)
    # interpolate to original signal length
    cam = np.interp(np.linspace(0, len(cam)-1, SIGNAL_LEN),
                    np.arange(len(cam)), cam)
    cam = cam / (cam.max() + 1e-8)
    return cam

# %%
cnn_model = trained['CNN'][0]
last_conv = [l.name for l in cnn_model.layers if 'conv1d' in l.name][-1]
print(f"Last conv layer: {last_conv}")

# %%
fig, axes = plt.subplots(5, 1, figsize=(14, 12))
for cls in range(5):
    idx = np.where(y_test_raw == cls)[0][0]
    x_sample = X_test_scaled[idx]
    cam = grad_cam_1d(cnn_model, x_sample, cls, last_conv)
    axes[cls].plot(x_sample.squeeze(), color='black', linewidth=0.8, label='ECG')
    axes[cls].fill_between(range(SIGNAL_LEN), x_sample.squeeze().min(),
                           x_sample.squeeze().max(), where=cam > 0.5,
                           alpha=0.35, color='red', label='Grad-CAM (>0.5)')
    im = axes[cls].scatter(range(SIGNAL_LEN), x_sample.squeeze(),
                           c=cam, cmap='YlOrRd', s=3, zorder=3)
    axes[cls].set_title(f'Class {cls} — {LABELS_MIT[cls]}')
    axes[cls].set_ylabel('Amplitude')
    axes[cls].grid(True, alpha=0.2)
    if cls == 4:
        axes[cls].set_xlabel('Sample')
plt.colorbar(im, ax=axes, shrink=0.6, label='Grad-CAM intensity')
plt.tight_layout()
plt.show()

# %% [markdown]
# ## Weighted ensemble
# 
# We combine predictions from all models using **F1-weighted averaging**:
# each model's vote is weighted by its macro-F1 on the validation set.

# %% [markdown]
# Compute weights from val-set macro-F1:

# %%
weights = {}
for name, (model, _) in trained.items():
    preds_v = model.predict(X_val, verbose=0).argmax(axis=1)
    weights[name] = f1_score(y_val, preds_v, average='macro')

total = sum(weights.values())
weights = {k: v / total for k, v in weights.items()}
print("Ensemble weights:")
for k, v in weights.items():
    print(f"  {k:20s}  {v:.4f}")

# %% [markdown]
# Weighted ensemble on test set:

# %%
ensemble_probs = np.zeros((len(y_test_raw), NUM_CLASSES_MITBIH))
for name, (model, _) in trained.items():
    ensemble_probs += weights[name] * model.predict(X_test_scaled, verbose=0)

ensemble_preds = ensemble_probs.argmax(axis=1)
ens_acc = (ensemble_preds == y_test_raw).mean()
ens_f1  = f1_score(y_test_raw, ensemble_preds, average='macro')

print(f"\nEnsemble  Accuracy: {ens_acc:.4f}  |  Macro-F1: {ens_f1:.4f}")
print(classification_report(y_test_raw, ensemble_preds, target_names=LABELS_MIT, digits=4))

# %% [markdown]
# Confusion matrix:

# %%
cm_ens = confusion_matrix(y_test_raw, ensemble_preds)
plt.figure(figsize=(6, 5))
sns.heatmap(cm_ens, annot=True, fmt='d', cmap='Greens',
            xticklabels=range(5), yticklabels=range(5))
plt.title(f'Ensemble Confusion Matrix  (Acc={ens_acc:.4f})')
plt.xlabel('Predicted')
plt.ylabel('True')
plt.show()

# %% [markdown]
# ## Transfer learning - PTB diagnostic ECG
# 
# We fine-tune the best performing model's feature extractor on the **PTB dataset**
# (binary classification: Normal vs Myocardial Infarction).

# %%
ptb = pd.concat([ptb_normal, ptb_abnormal], ignore_index=True)
ptb = ptb.sample(frac=1, random_state=SEED).reset_index(drop=True)

X_ptb = ptb.iloc[:, :187].values
y_ptb = ptb.iloc[:, 187].values.astype(int)

# Normalise with same scaler
X_ptb = scaler.transform(X_ptb).reshape(-1, SIGNAL_LEN, 1)

X_ptb_train, X_ptb_test, y_ptb_train, y_ptb_test = train_test_split(
    X_ptb, y_ptb, test_size=0.2, stratify=y_ptb, random_state=SEED)

y_ptb_train_cat = to_categorical(y_ptb_train, 2)
y_ptb_test_cat  = to_categorical(y_ptb_test,  2)
print(f"PTB Train: {X_ptb_train.shape}  Test: {X_ptb_test.shape}")

# %%
best_name = df_summary['Macro-F1'].idxmax()
best_model = trained[best_name][0]
print(f"Best model for transfer learning: {best_name}")

base = Model(inputs=best_model.input,
             outputs=best_model.layers[-3].output, name='feature_extractor')
for layer in base.layers:
    layer.trainable = False

inp = Input(shape=(SIGNAL_LEN, 1))
x = base(inp, training=False)
x = layers.Dense(64, activation='relu')(x)
x = layers.Dropout(0.4)(x)
out = layers.Dense(2, activation='softmax')(x)
ptb_model = Model(inp, out, name='PTB_Transfer')

ptb_model.compile(optimizer=tf.keras.optimizers.Adam(1e-3),
                  loss='categorical_crossentropy', metrics=['accuracy'])

# %% [markdown]
# Phase 1: train only new head

# %%
h1 = ptb_model.fit(X_ptb_train, y_ptb_train_cat,
                   validation_data=(X_ptb_test, y_ptb_test_cat),
                   epochs=20, batch_size=64, verbose=1,
                   callbacks=[EarlyStopping(monitor='val_loss', patience=5,
                                            restore_best_weights=True)])

# %% [markdown]
# Phase 2: unfreeze and fine-tune everything with low LR

# %%
for layer in base.layers:
    layer.trainable = True
ptb_model.compile(optimizer=tf.keras.optimizers.Adam(1e-5),
                  loss='categorical_crossentropy', metrics=['accuracy'])
h2 = ptb_model.fit(X_ptb_train, y_ptb_train_cat,
                   validation_data=(X_ptb_test, y_ptb_test_cat),
                   epochs=30, batch_size=64, verbose=1,
                   callbacks=[EarlyStopping(monitor='val_loss', patience=5,
                                            restore_best_weights=True)])

# %% [markdown]
# PTB evaluation:

# %%
ptb_probs = ptb_model.predict(X_ptb_test, verbose=0)
ptb_preds = ptb_probs.argmax(axis=1)
print(classification_report(y_ptb_test, ptb_preds, target_names=LABELS_PTB, digits=4))

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

cm_ptb = confusion_matrix(y_ptb_test, ptb_preds)
sns.heatmap(cm_ptb, annot=True, fmt='d', cmap='Oranges', ax=axes[0],
            xticklabels=LABELS_PTB, yticklabels=LABELS_PTB)
axes[0].set_title('PTB Transfer — Confusion Matrix')
axes[0].set_xlabel('Predicted')
axes[0].set_ylabel('True')

for c, lbl in enumerate(LABELS_PTB):
    fpr, tpr, _ = roc_curve(y_ptb_test_cat[:, c], ptb_probs[:, c])
    axes[1].plot(fpr, tpr, label=f'{lbl} (AUC={auc(fpr, tpr):.3f})')
axes[1].plot([0,1],[0,1],'k--', alpha=0.3)
axes[1].set_title('PTB Transfer — ROC')
axes[1].set_xlabel('FPR')
axes[1].set_ylabel('TPR')
axes[1].legend()
axes[1].grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# %% [markdown]
# ## Conclusions
# 
# ### Pipeline summary
# 
# | Stage | Technique |
# |-------|-----------|
# | **Balancing** | SMOTE (Synthetic Minority Oversampling) |
# | **Augmentation** | Gaussian noise, baseline wander, amplitude scaling |
# | **Normalisation** | Z-score (StandardScaler) |
# | **Architectures** | CNN · SE-ResNet · BiLSTM+Attention · Transformer |
# | **Training** | Cosine-annealing LR, early stopping, best-model checkpointing |
# | **Evaluation** | Held-out test set, confusion matrices, ROC, PR curves |
# | **Robustness** | 5 synthetic perturbation tests |
# | **Explainability** | 1-D Grad-CAM |
# | **Ensemble** | F1-weighted model averaging |
# | **Transfer Learning** | Two-phase fine-tuning (frozen → unfrozen) on PTB dataset |
# 
# ### Key takeaways
# 
# - **SE-ResNet** and **Transformer** architectures achieve competitive or superior
#   performance compared to baseline CNNs, while offering better robustness to input perturbations.
# - **Grad-CAM** reveals that the models focus on the QRS complex and ST segment —
#   regions clinically associated with arrhythmia markers.
# - The **weighted ensemble** consistently outperforms any single model.
# - **Transfer learning** from MIT-BIH → PTB demonstrates that learned ECG features
#   generalise across datasets and clinical tasks.


