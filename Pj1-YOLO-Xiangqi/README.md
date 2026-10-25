# RET503 — Project 1: Computer Vision and Deep Learning

## 1. Deskripsi Proyek

Proyek ini merupakan pengerjaan Project 1 mata kuliah Computer Vision and Deep Learning (RET503). Proyek berfokus pada object detection untuk mendeteksi bidak Xiangqi (Chinese Chess) menggunakan YOLO. Tahapan yang dikerjakan mencakup pemeriksaan dataset dan anotasi, pelatihan beberapa konfigurasi model, serta evaluasi dan perbandingan hasil eksperimen.

## 2. Dataset

Dataset berasal dari Roboflow dan menggunakan format anotasi YOLO.

- **Jenis tugas:** Object Detection
- **Objek:** Bidak Xiangqi
- **Jumlah gambar:** 3.600
- **Jumlah kelas:** 14
- **Train:** 3.150 gambar
- **Validation:** 300 gambar
- **Test:** 150 gambar

Dataset sudah memiliki pembagian train, validation, dan test dari sumber. Dataset mentah tidak disertakan dalam repositori ini.

## 3. Program

| File | Fungsi |
|---|---|
| `00_ambil_subset_coco.py` | Mengambil subset dataset COCO dan mengubah anotasinya ke format YOLO. |
| `01_cek_anotasi.py` | Memeriksa format dan visualisasi anotasi YOLO. |
| `02_yolo_ke_coco.py` | Mengonversi anotasi YOLO ke format COCO. |
| `03_coco_ke_yolo.py` | Mengonversi anotasi COCO ke format YOLO. |
| `05_split_dataset_yolo.py` | Membagi dataset YOLO menjadi train, validation, dan test serta membuat `data.yaml`. |
| `06_latih_bandingkan.py` | Melatih dan membandingkan eksperimen model YOLO. |

Catatan: skrip 00–05 merupakan skrip pendukung alur kerja yang disediakan untuk tugas. Eksperimen Xiangqi yang dilaporkan menggunakan dataset Roboflow yang sudah terbagi.

## 4. Konfigurasi Eksperimen

Eksperimen menggunakan model `yolov8n`, 30 epoch, ukuran gambar awal 416, batch size 16, dan GPU Tesla T4.

Tiga konfigurasi yang dibandingkan:

- **E1 — Pretrained:** menggunakan bobot pretrained sebagai baseline.
- **E2 — Scratch:** pelatihan dari bobot awal acak.
- **E3 — Tuning:** konfigurasi baseline dengan perubahan `imgsz=640`.

## 5. Hasil dan Evaluasi

### Perbandingan pada validation set

| Eksperimen | Perubahan | mAP50 | mAP50-95 | Precision (P) | Recall (R) | Waktu (menit) |
|---|---|---:|---:|---:|---:|---:|
| E1 — Pretrained | Tidak ada | 0.995 | 0.968 | 0.999 | 0.998 | 26.0 |
| E2 — Scratch | Tidak ada | 0.992 | 0.923 | 0.977 | 0.976 | 25.5 |
| E3 — Tuning | `imgsz=640` | 0.995 | 0.979 | 1.000 | 0.999 | 31.0 |

E3 dipilih sebagai model terbaik karena memperoleh mAP50-95 validation tertinggi, yaitu **0.979**. E1 memiliki hasil yang mendekati, sedangkan E2 menghasilkan nilai mAP50-95 lebih rendah pada eksperimen ini.

### Evaluasi pada test set

Model terpilih E3 memperoleh:

- **mAP50:** 0.995
- **mAP50-95:** 0.972

Hasil grafik dan evaluasi tiap eksperimen tersedia di folder `Hasil_PJ 1/hasil_praktik/`. Ringkasan eksperimen juga tersedia dalam format TXT dan CSV:

- `Hasil_PJ 1/hasil_praktik_4222401051.txt`
- `Hasil_PJ 1/hasil_praktik_4222401051.csv`

## 6. Struktur Hasil

```text
Hasil_PJ 1/
├── hasil_praktik_4222401051.txt
├── hasil_praktik_4222401051.csv
└── hasil_praktik/
    ├── E1_pretrained/
    ├── E2_scratch/
    └── E3_tuning/
```

Setiap folder eksperimen berisi grafik metrik, confusion matrix, dan file hasil training yang dihasilkan oleh Ultralytics YOLO.

## 7. Menjalankan Skrip Eksperimen

Pastikan Python, `ultralytics`, PyTorch, dan dataset YOLO sudah tersedia. Contoh menjalankan skrip perbandingan:

```bash
python 06_latih_bandingkan.py \
  --nim 4222401051 \
  --nama "Cheryll Hattie Wright Sianipar" \
  --data /path/ke/dataset/data.yaml \
  --model yolov8n \
  --epochs 30 \
  --imgsz 416 \
  --batch 16 \
  --device 0 \
  --ubah imgsz=640
```

Ganti `/path/ke/dataset/data.yaml` dengan lokasi `data.yaml` pada komputer yang digunakan. Perintah di atas menjalankan eksperimen dan dapat menghasilkan ulang file hasil, sehingga tidak perlu dijalankan jika hanya ingin melihat laporan dan grafik yang sudah disertakan.

## 8. Identitas

- **Nama:** Cheryll Hattie Wright Sianipar
- **NIM:** 4222401051
- **Program Studi:** D4 Teknologi Rekayasa Robotika
- **Mata Kuliah:** Computer Vision and Deep Learning (RET503)