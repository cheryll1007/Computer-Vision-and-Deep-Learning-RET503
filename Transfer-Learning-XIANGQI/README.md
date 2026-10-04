# Transfer Learning XIANGQI

## 1. Deskripsi
Proyek ini merupakan implementasi transfer learning untuk klasifikasi dua jenis bidak catur Xiangqi, yaitu horse hitam dan chariot hitam.

Model yang digunakan adalah ResNet-18 dengan tiga pendekatan:
1. Feature Extraction
2. Partial Fine-Tuning
3. Training from Scratch

## 2. Dataset

Dataset terdiri dari dua kelas:
- horse_hitam
- chariot_hitam

Jumlah data:
- 100 gambar per kelas
- Total 200 gambar

Pembagian dataset:
- Training: 80 gambar per kelas
- Validation: 20 gambar per kelas

Sehingga:
- Training: 160 gambar
- Validation: 40 gambar

## 3. Data Augmentation

Pada data training digunakan:
- RandomResizedCrop
- RandomHorizontalFlip
- ColorJitter

Untuk validation digunakan resize dan center crop tanpa augmentasi training.

## 4. Model dan Metode

### Feature Extraction
Menggunakan ResNet-18 pretrained ImageNet. Semua layer dibekukan dan hanya fully connected layer yang dilatih.

Learning rate:
- FC: 1e-3

### Partial Fine-Tuning
Menggunakan ResNet-18 pretrained ImageNet. Layer4 dan fully connected layer dilatih.

Learning rate:
- Layer4: 1e-4
- FC: 1e-3

### Training from Scratch
Menggunakan ResNet-18 tanpa pretrained weights dan seluruh layer dilatih.

Learning rate:
- 1e-3

Training dilakukan selama 10 epoch menggunakan CosineAnnealingLR.

## 5. Hasil Training

| Metode | Best Validation Accuracy | Epoch Pertama >= 90% | Training Time |
|---|---:|---:|---:|
| Feature | 95% | 6 | 1005.03 s |
| Partial | 100% | 1 | 1269.22 s |
| Scratch | 95% | 6 | 1570.06 s |

## 6. Model Terbaik

Model terbaik adalah **Partial Fine-Tuning** dengan:
- Best Validation Accuracy: 100%
- Epoch pertama mencapai >=90%: Epoch 1
- Training time: 1269.22 detik

Model terbaik disimpan pada:

`model/resnet18_partial_best.pth`

## 7. Latency

Pengukuran latency dilakukan pada model Partial menggunakan CPU.

Hasil:
- Average latency: 188.25 ms
- Average FPS: 5.31

Artinya, model membutuhkan rata-rata sekitar 188.25 ms untuk melakukan inferensi satu gambar pada CPU.

## 8. Kesimpulan

Berdasarkan hasil pengujian, metode Partial Fine-Tuning memberikan hasil terbaik dengan validation accuracy sebesar 100%.

Feature Extraction dan Training from Scratch sama-sama memperoleh validation accuracy sebesar 95%.

Partial Fine-Tuning dipilih sebagai model terbaik karena mampu mencapai akurasi validasi tertinggi dan mencapai akurasi >=90% sejak epoch pertama.
