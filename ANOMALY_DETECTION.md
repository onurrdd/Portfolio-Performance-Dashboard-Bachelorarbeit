# Anomali Tespiti (Portföy vs. Benchmark)

Bu doküman yalnızca **anomali/kırılma günü tespiti** bölümünü kapsar. İlgili kod:
[utils/anomaly.py](utils/anomaly.py) (çekirdek mantık), [callbacks/charts.py](callbacks/charts.py)
(veri hazırlama + `analysis_data['active_return_breaks']`), [callbacks/naive_llm.py](callbacks/naive_llm.py)
(geçici debug tablosu).

## Amaç

"Portföy, S&P 500'e kıyasla hangi günlerde sıra dışı davrandı ve bu sıra dışılık
portföye mi özgü yoksa piyasa geneli mi?" sorusunu, **look-ahead bias içermeden**
ve tek bir uç günün istatistiği bozmasına ("maskeleme") izin vermeden yanıtlamak.

## Girdi serileri

- **Portföy getirisi** R_p[t]: nakit akışından arındırılmış, günlük TWR getirisi.
  `calculate_twr_metrics` ([utils/metrics.py](utils/metrics.py)) her getiriyi kendi
  **bitiş gününe** etiketler; alım tarihindeki dönem geçişleri (sermaye girişi) bir
  getiri üretmeden atlanır, böylece tarih hizalaması ve nakit akışından arındırma
  aynı anda sağlanır.
- **Benchmark getirisi** R_b[t]: S&P 500'ün **PME** (Public Market Equivalent) ile
  portföyle aynı takvim/sermaye akışına göre simüle edilmiş, aynı yöntemle
  hesaplanmış TWR günlük getirisi.
- Temettü/bölünme düzeltmeleri yfinance'ın düzeltilmiş fiyatlarıyla yapılmış.

## Algoritma (Market-Model + robust MAD-z)

**Adım 1 — Beklenen hareketi çıkar.** Her gün t için, yalnızca t'den **önceki**
60 işlem günlük pencereden rollierend Beta/Alpha (Portföy ~ Benchmark regresyonu):
`beta = Cov(R_p, R_b) / Var(R_b)`, `alpha = mean(R_p) − beta·mean(R_b)`.
`beklenen[t] = alpha + beta·R_b[t]`, **`sürpriz[t] = R_p[t] − beklenen[t]`**.
Anomali avı ham aktif getiri üzerinde değil, bu **sürpriz** serisi üzerinde yapılır —
böylece piyasa kaynaklı hareketler (beta ile açıklanan kısım) elenir.

**Adım 2 — Sürprizi kendi tipik büyüklüğüne göre ölç.** Yine t'den önceki pencerede,
sürprizin **medyanı** ve **MAD**'i (Median Absolute Deviation) hesaplanır (gün kendisi
hariç): `mad_z[t] = (sürpriz[t] − medyan) / (1.4826·MAD)`. Ortalama/std yerine
medyan/MAD kullanılır: tek bir devasa gün klasik std'yi şişirip komşu şokları normal
gösterir (maskeleme); MAD buna dayanıklıdır.

**Adım 3 — Sabit eşik.** `|mad_z| > 3` ise anomali. Yüzdelik değil, sabit eşik;
sakin dönemde sıfır anomali doğru cevaptır.

**Adım 4 — Aynı gün sınıflandırma.** Benchmark'ın kendi MAD-z'si de hesaplanır:
- Benchmark **normal**, sürpriz uç → **"Portföye özgü"** (asıl aranan).
- Benchmark **uç** + sürpriz uç → **"Kopma"** (kriz anında beta ilişkisi bozuldu).
- (Benchmark uç, sürpriz normal → işaretlenmez: portföy beklendiği gibi davrandı.)

**Adım 5 — Ucuz elemeler (uyarı bayrağı).** Gün otomatik kabul edilmez; şu durumlar
işaretlenir: alım tarihi ±1 işlem günü içinde mi, o gün aktif bir tickerın fiyatı
bir önceki güne birebir eşit mi (stale/ffill artefaktı).

**Adım 6 — Ticker kırılımı (işaret-tutarlı).** Sürprizin yönüyle **aynı yönde** katkı
yapan (ağırlık × günlük getiri) tickerlar arasından en büyüğü "sorumlu" seçilir.
Sürpriz yönünde hiç ticker yoksa "atfedilemez" denir. Konsantrasyon
(`|en büyük| / Σ|aynı yönlü|`) > 0.6 ise **"Hisseye özgü"**, değilse **"Faktör/Sektör"**.

## Parametreler

| Parametre | Değer | Not |
|---|---|---|
| Pencere | 60 işlem günü | Beta ve MAD tahmini |
| Min. gözlem | 40 | `min_periods=40`: pencere 60'a dolmadan, 40 geçerli gözlemle değer üretilmeye başlanır |
| Isınma | ~40 gün | Raporlanmaz; ilk geçerli değer 40. günde çıkar (60 değil — doğrulandı: [utils/anomaly.py](utils/anomaly.py) `MIN_OBS`) |
| Eşik | \|mad_z\| > 3 | MAD ölçeğinde (≈ 2σ'ya yakın) |
| Konsantrasyon sınırı | 0.6 | Hisse- vs. faktör-olayı |

## Mevcut veri üzerindeki doğrulama (5 tickerlı test portföyü)

- **13 anomali günü**, işaretlenme oranı **~%1.8** (hedef %1-3 ✓).
- Sınıflandırma: 13/13 "Portföye özgü", 0 "Kopma".
- Konsantrasyon: 9 "Hisseye özgü", 4 "Faktör/Sektör".
- Uyarı bayrağı: 0 gün.

### Neden "Kopma" hiç çıkmıyor

Bu, tasarımın beklenen bir sonucu. `sürpriz = gerçekleşen − (alfa + beta·benchmark)`
tanımı gereği benchmark'ın açıklayabildiği kısmı zaten dışlar; ampirik olarak
sürpriz ile benchmark'ın kendi uç-değeri arasındaki korelasyon **≈ 0.006** —
neredeyse tam bağımsız. "Kopma" için gereken iki koşul (sürpriz uç VE benchmark uç)
yaklaşık bağımsız olduğundan, ikisinin aynı anda gerçekleşmesi istatistiksel olarak
nadir: 669 geçerli günde bağımsızlık varsayımıyla beklenen sayı **0.466**, gözlenen **0**
— tamamen normal bir sonuç. Makro/piyasa geneli şoklar bu yöntemde ancak portföyün
beta ilişkisi o gün gerçekten bozulursa (nadir, kriz-tipi bir olay) "Kopma" olarak görünür;
beta ilişkisi geçerliliğini koruduğu sürece büyük piyasa hareketleri sürprize hiç
yansımaz ve haklı olarak işaretlenmez.

## Çıktı yapısı

`analysis_data['active_return_breaks']` — her anomali için:
`date, actual_return_pct, expected_return_pct, surprise_pct, surprise_mad_z,
benchmark_return_pct, benchmark_mad_z, beta, classification, responsible_ticker,
ticker_contribution_pct, concentration, flags`.

## Sınırlamalar

- 60 günlük pencerede beta tahmini gürültülüdür (5 hisseli, az çeşitlendirilmiş
  portföyde beta oynak); sınıflandırmayı sınırda etkileyebilir.
- Ticker kırılımı, sürprizi değil **gerçekleşen getiriyi** ayrıştırır (ticker-bazlı
  beta düzeltmesi yapılmaz) — lean bir yaklaşım, yön ve konsantrasyon için yeterli.
- Sayısal koruma: bir pencerede benchmark varyansı veya MAD tam olarak 0 çıkarsa
  (aşırı durgun/veri eksikliği) o günün z-skoru NaN olur ve otomatik elenir.
- Ticker kırılımında katkısı tam olarak 0 olan bir hisse "aynı yönlü" sayılmaz.
