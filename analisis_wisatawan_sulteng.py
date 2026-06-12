import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import mean_absolute_error, r2_score
import warnings
warnings.filterwarnings('ignore')

# === 1. BACA DATA & PREPROCESSING ===
df = pd.read_csv('Dataset_Wisatawan_Sulteng2024_2026.csv')
BULAN = {1:'Jan',2:'Feb',3:'Mar',4:'Apr',5:'Mei',6:'Jun',7:'Jul',8:'Agu',9:'Sep',10:'Okt',11:'Nov',12:'Des'}
BULAN_P = {i: v + ('uari' if i<=3 else 'er' if i>=9 and i!=8 and i!=10 else 'ustus' if i==8 else 'ober' if i==10 else '') for i, v in BULAN.items()}

train = df[df['tahun'] < 2026].sort_values(['kabupaten_kota','tahun','bulan']).copy()
val = df[df['tahun'] == 2026].copy()
bln_val, kab_list = sorted(val['bulan'].unique()), sorted(train['kabupaten_kota'].unique())
bln_pred = [b for b in range(1,13) if b not in bln_val]

print(f"Training: {len(train)} baris | Validasi: {len(val)} baris\nEstimasi: {[BULAN_P[b] for b in bln_pred]}")

# === 2. ENCODING & REKAYASA FITUR ===
le_kab, le_m = LabelEncoder(), LabelEncoder()
train['kab_enc'] = le_kab.fit_transform(train['kabupaten_kota'])
train['lag'] = train.groupby('kabupaten_kota')['jumlah_wisatawan'].shift(1)
train = train.dropna(subset=['lag'])

q33, q66 = train['jumlah_wisatawan'].quantile([0.33, 0.66])
train['musim'] = train['jumlah_wisatawan'].apply(lambda v: 'Ramai' if v>=q66 else 'Sedang' if v>=q33 else 'Sepi')
train['musim_enc'] = le_m.fit_transform(train['musim'])
print(f"Batas → Sepi:<{q33:,.0f} | Sedang:{q33:,.0f}-{q66:,.0f} | Ramai:>{q66:,.0f}")

# === 3. MODEL TRAINING ===
FITUR = ['bulan','tahun','kab_enc','lag']
sc = StandardScaler()
Xtr = sc.fit_transform(train[FITUR])

reg = LinearRegression().fit(Xtr, train['jumlah_wisatawan'])
svc = SVC(kernel='rbf', C=1.0, random_state=42).fit(Xtr, train['musim_enc'])
print(f"\nHasil Training → R²:{r2_score(train['jumlah_wisatawan'], reg.predict(Xtr)):.4f} | MAE:{mean_absolute_error(train['jumlah_wisatawan'], reg.predict(Xtr)):,.0f}")

def est(kab, bln, thn, lag):
    X = sc.transform([[bln, thn, le_kab.transform([kab])[0], lag]])
    return max(0, round(reg.predict(X)[0])), le_m.inverse_transform(svc.predict(X))[0]

# === 4. PROSES VALIDASI (JAN-MAR 2026) ===
rows_v = []
for kab in kab_list:
    lag = train[(train['kabupaten_kota']==kab) & (train['tahun']==2025) & (train['bulan']==12)]['jumlah_wisatawan'].values[0]
    for b in bln_val:
        akt = val[(val['kabupaten_kota']==kab) & (val['bulan']==b)]['jumlah_wisatawan'].values[0]
        pred, mus = est(kab, b, 2026, lag)
        rows_v.append({'kabupaten_kota':kab, 'bulan':b, 'nama_bulan':BULAN_P[b], 'aktual':akt, 'estimasi':pred, 'error_pct':abs(pred-akt)/akt*100, 'musim':mus})
        lag = akt

dv = pd.DataFrame(rows_v)
mape = dv['error_pct'].mean()
print(f"Validasi   → R²:{r2_score(dv['aktual'], dv['estimasi'])} | MAPE:{mape:.1f}%\n\n{'Kab/Kota':<22}{'Bulan':<10}{'Aktual':>10}{'Estimasi':>10}{'Error':>7}\n" + "-"*60)
for _, r in dv.iterrows():
    print(f"{r['kabupaten_kota']:<22}{r['nama_bulan']:<10}{r['aktual']:>10,.0f}{r['estimasi']:>10,.0f}{r['error_pct']:>6.1f}%")

# === 5. PROSES ESTIMASI (APR-DES 2026) ===
rows_p = []
for kab in kab_list:
    lag = val[(val['kabupaten_kota']==kab) & (val['bulan']==max(bln_val))]['jumlah_wisatawan'].values[0]
    for b in bln_pred:
        pred, mus = est(kab, b, 2026, lag)
        rows_p.append({'kabupaten_kota':kab, 'bulan':b, 'nama_bulan':BULAN_P[b], 'estimasi_wisatawan':pred, 'musim':mus})
        lag = pred

dp = pd.DataFrame(rows_p)
print(f"\n{'Kab/Kota':<22}{'Total Apr-Des 2026':>20}{'vs 2025':>10}\n" + "-"*54)
for kab in kab_list:
    tp = dp[dp['kabupaten_kota']==kab]['estimasi_wisatawan'].sum()
    t25 = df[(df['kabupaten_kota']==kab) & (df['tahun']==2025) & (df['bulan'].isin(bln_pred))]['jumlah_wisatawan'].sum()
    print(f"{kab:<22}{tp:>20,.0f}  {'+' if tp>=t25 else ''}{(tp-t25)/t25*100:.1f}%")

# === 6. OUTPUT PENYIMPANAN ===
dv.to_csv('hasil_validasi_jan_mar_2026.csv', index=False)
dp.to_csv('hasil_estimasi_apr_des_2026.csv', index=False)

# === 7. VISUALISASI GRAFIK ===
fig, axes = plt.subplots(2, 3, figsize=(18,10))
fig.suptitle('Analisis Regresi Linear dan Klasifikasi Musim Kunjungan\nWisatawan Nusantara di Provinsi Sulawesi Tengah 2024-2026', fontsize=14, fontweight='bold', y=0.98)
WM = {'Ramai':'#1a7c4e','Sedang':'#e8a317','Sepi':'#c0392b'}

# G1: Tren Bulanan
ax = axes[0,0]
for thn, clr, lbl in [(2024,'#3498db','Aktual 2024'), (2025,'#2ecc71','Aktual 2025')]:
    d = df[df['tahun']==thn].groupby('bulan')['jumlah_wisatawan'].sum()
    ax.plot(d.index, d.values, marker='o', color=clr, label=lbl, lw=2)
ax.plot(bln_val, [dv.groupby('bulan')['aktual'].sum().get(b,0) for b in bln_val], marker='s', color='#e74c3c', label='Aktual 2026', lw=2)
ax.plot(bln_pred, [dp.groupby('bulan')['estimasi_wisatawan'].sum().get(b,0) for b in bln_pred], marker='D', color='#e67e22', ls='--', label='Estimasi Apr-Des 2026', lw=2)
ax.set_xticks(range(1,13)); ax.set_xticklabels(list(BULAN.values()), fontsize=8)
ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v,_: f'{v/1e6:.1f}M'))
ax.set_title('Tren Wisatawan 2024-2026', fontweight='bold'); ax.legend(fontsize=7)

# G2: Scatter Aktual vs Estimasi
ax = axes[0,1]
ax.scatter(dv['aktual'], dv['estimasi'], color='#8e44ad', alpha=0.7, s=55)
mn, mx = dv['aktual'].min(), dv['aktual'].max()
ax.plot([mn,mx],[mn,mx],'r--', lw=1.5, label='Ideal')
ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda v,_: f'{v/1000:.0f}K'))
ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v,_: f'{v/1000:.0f}K'))
ax.set_title(f'Validasi Aktual vs Estimasi\nR²={r2_score(dv['aktual'], dv['estimasi']):.3f} | MAPE={mape:.1f}%', fontweight='bold'); ax.legend()

# G3: Heatmap Proyeksi
ax = axes[0,2]
pv = dp.pivot_table(index='kabupaten_kota', columns='bulan', values='estimasi_wisatawan', aggfunc='sum')
pv.columns = [list(BULAN.values())[b-1] for b in pv.columns]
sns.heatmap(pv/1000, ax=ax, cmap='YlOrRd', fmt='.0f', annot=True, annot_kws={'size':6.5}, linewidths=0.3, cbar_kws={'label':'(ribu)'})
ax.set_title('Heatmap Estimasi Apr-Des 2026 (ribu)', fontweight='bold'); ax.tick_params(axis='y', labelsize=7); ax.set_xlabel(''); ax.set_ylabel('')

# G4: MAPE per Wilayah
ax = axes[1,0]
mk = dv.groupby('kabupaten_kota')['error_pct'].mean().sort_values()
mk.plot(kind='barh', ax=ax, color=['#1a7c4e' if v<=10 else '#e8a317' if v<=20 else '#c0392b' for v in mk])
ax.axvline(10, color='green', ls='--', lw=1, label='≤10% bagus')
ax.axvline(20, color='orange', ls='--', lw=1, label='≤20% wajar')
ax.set_title('MAPE Validasi per Kab/Kota (%)', fontweight='bold'); ax.tick_params(axis='y', labelsize=7); ax.legend(fontsize=7)

# G5: Sebaran Kategori Musim
ax = axes[1,1]
md = dp.groupby(['kabupaten_kota','musim']).size().unstack(fill_value=0)
for m in ['Ramai','Sedang','Sepi']: 
    if m not in md.columns: md[m]=0
md[['Ramai','Sedang','Sepi']].plot(kind='barh', ax=ax, stacked=True, color=[WM['Ramai'],WM['Sedang'],WM['Sepi']])
ax.set_title('Klasifikasi Musim Apr-Des 2026', fontweight='bold'); ax.tick_params(axis='y', labelsize=7); ax.legend(title='Musim', fontsize=8)

# G6: Nilai Koefisien Regresi
ax = axes[1,2]
koef = pd.Series(reg.coef_, index=['X1:bulan','X2:tahun','X3:kab','X4:lag'])
ax.barh(koef.index, koef.values, color=['#1a7c4e' if v>=0 else '#c0392b' for v in koef])
ax.axvline(0, color='black', lw=0.8); ax.set_title('Koefisien Regresi', fontweight='bold')

plt.tight_layout(rect=[0,0,1,0.96])
plt.savefig('grafik_analisis_wisatawan_sulteng.png', dpi=150, bbox_inches='tight')
plt.show()

print("\nSELESAI — File tersimpan:\n hasil_validasi_jan_mar_2026.csv\n hasil_estimasi_apr_des_2026.csv\n grafik_analisis_wisatawan_sulteng.png")