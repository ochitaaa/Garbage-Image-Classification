# Garbage-Image-Classification
This project focuses on image classification using deep learning to classify garbage images into predefined categories.   The trained models are exported into multiple formats, including TensorFlow, TensorFlow Lite, and TensorFlow.js. This project was developed as a module submission for the Asah by Dicoding program.

----

## Deskripsi Proyek
Proyek ini bertujuan untuk mengaplikasikan algorima _deep learning_ yaitu model klasifikasi gambar dengan menggunakan "_Gabrbage Dataset_" yang diambil Kaggle. Dataset tersebut terdiri dari 19.762 gambar yang dikategorikan menjadi 10 kelas. Gambar tersebut nantinya akan diklasifikasikan dengan menggunakan salah satu arsitektur CNN yaitu MobileNetV2 yang telah dilatih pada ImageNet. Setelah itu, model tersebut akan disimpan dalam format savedmodel, TF-Lite, dan TFJS sehingga dapat digunakan diberbagai _platform_. Setelah model selesai dilatih, akan dilakukan inferensi terhadap sejumlah gambar untuk menguji ketepatan klasifikasi pada setiap kelasnya.

----

## Deskripsi Dataset
Dataset yang diambil yaitu berasal dari https://www.kaggle.com/datasets/sumn2u/garbage-classification-v2. Dataset tersebut memiliki jumlah gambar sebanyak 19.762 buah yang dikategorikan menjadi 10 kelas yaitu _battery_, _biological_, _cardboard_, _clothes_, _glass_, _metal_, _paper_, _plastic_, _shoes_, dan _trash_. Kemudian, telah dilkukan eksplorasi terhadap dataset, setiap kelasnya memiliki jumlah distribusi gambar dan dimensi yang berbeda. Dari 30 sampel gambar yang diambil dari setiap kelasnya, dimensi gambar yang dimiliki memiliki ukuran rata-rata 406.5 x 391.9 piksel.

----

## Metodologi
#### 1. Load Dataset
Gambar diambil dari Kaggle dengan cara mengunduh folder ZIP kemudian di ekstrak. Kemudian, dataset dimuat ke direktori lokal _garbage dataset_  dan dengan bantuan pustaka os dan pathlib untuk membaca struktur folder dan mendeteksi kelas gambar.

#### 2. Preprocessing Gambar
Sebelum diproses, dataset akan dipecah menjadi 70% data _train_, 15% data _validation_, dan 15% data _test_ menggunakan `train_test_split`. Kemudian semua gambar akan diubah ukurannya menjadi 224×224 piksel dan dinormalisasikan ke rentang [0, 1]. Setelah itu pada data _train_, dilakukan augmentasi dengan `ImageDataGenerator` yang meliputi rotasi gambar sebesar 25 derajat, menggeser gambar ke kanan, ke kiri, ke atas, dan ke bawah sebesar 0.15 jarak gambar, memiringkan gambar, memperbesar dan memperkecil gambar, menggeser warna RGB gambar, memutar gambar secara _horizontal_, serta mengatur keceraharan gambar.

#### 3. Arsitektur Model CNN
Dataset yang awalnya tidak seimbang akan dihitung bobotnya kemudian dilatih dengan salah satu arsitektur CNN yaitu MobileNetV2 dengan _pre-training_ dari ImageNet. Kemudian dilakukan _fine-tuning_ sebagian yaitu semua lapisan kecuali 30 lapisan terakhir akan dibekukan. Diberikan lapisan tambahan pada model meliputi:
- `layers.SeparableConv2D(64, (3,3), padding='same', activation='relu')` sebagai lapisan konvolusi tambahan dengan jumlah _output_ sebesar 64, kernel size 3x3, dan fungsi aktivasi relu.
- `layers.BatchNormalization(trainable=False)` untuk menormalisasikan nilai aktivasi setiap _batch_.
- `layers.GlobalAveragePooling2D()` untuk mengubah _feature map_ menjadi vektor 1D dengan merata-ratakan setiap _channel_.
- `layers.Dense(128, activation='relu', kernel_regularizer=regularizers.l2(1e-4))` sebagai _dense layer_ dengan 128 neuron yang akan dilatih dengan fungsi aktivasi relu dan regularisasi L2.
- `layers.Dropout(0.5)` untuk mengeluarkan secara acak 50% neuron saat pelatihan.
- `layers.Dense(10, activation='softmax', dtype='float32')` sebagai lapisan _output_ dengan 10 neuron sebagai kelasnya dan fungsi aktivasi softmax sehingga nilai keluarannya memiliki nilai maksimal sebesar 1.

#### 4. Pelatihan
Untuk mencegah _overfitting_ dan menghemat waktu pelatihan, dilakukan _early stopping_ dengan memantau nilai _validation loss_. Apabila selama 3 epoch berturut-turut, nilai _validation loss_ tidak membaik, pelatihan akan dihentikan dan akan diambil nilai bobot terbaik. Selanjutnya, digunakan optimasi Adam dengan _learning rate_ sebesar 0.0005 sebagai bobot model baru. Kemudian, akan digunakan fungsi kerugian `categorical_crossentropy` serta menggunakan metrik penilaian `accuracy` selama pelatihan dan validasi. Setelah itu, model akan mulai melatih dari data latih yang telah di augmentasi dan memantau kinerja model dengan data validasi sebanyak maksimal 20 _epoch_.

----

## Hasil Evaluasi Model
Setelah melakukan pelatihan model, model berhenti setelah 9 _epoch_. Dengan memperoleh nilai evaluasi sebagai berikut,
| Dataset | Akurasi |
|:--------:|:--------:|
| Train | 94.39% |
| Validation | 91.13% |
| Test | 92.88% |

Tabel diatas merupakan nilai evaluasi dari setiap dataset. Dari nilai akurasi yang diperoleh, model menunjukkan performa yang stabil antara data latih, validasi, dan uji, serta menandakan tidak adanya _overfitting_ yang signifikan. Kemudian, diperoleh juga distribusi prediksi benar dan salah untuk setiap kelasnya. 

| Kelas | Precision | Recall | F1-Score |
|:--------|:---------:|:-------:|:---------:|
| battery | 0.9252 | 0.9645 | 0.9444 |
| biological | 0.9408 | 0.9597 | 0.9502 |
| cardboard | 0.9286 | 0.9015 | 0.9148 |
| clothes | 0.9924 | 0.9750 | 0.9836 |
| glass | 0.9643 | 0.8804 | 0.9205 |
| metal | 0.8204 | 0.8954 | 0.8562 |
| paper | 0.8929 | 0.8929 | 0.8929 |
| plastic | 0.8429 | 0.9362 | 0.8871 |
| shoes | 0.9589 | 0.9428 | 0.9508 |
| trash | 0.8039 | 0.8662 | 0.8339 |

| Rata-rata | Precision | Recall | F1-Score |
|:-----------|:---------:|:-------:|:---------:|
| Macro Avg | 0.9070 | 0.9215 | 0.9134 |
| Weighted Avg | 0.9316 | 0.9288 | 0.9295 |

Diatas merupakan _classification report_ untuk setiap kelas, model memiliki rata-rata akurasi sebesar 92.88%, dengan rata-rata F1-_score_ sebesar 0.93. Kelas seperti _clothes_, _biological_, dan _shoes_ memiliki performa sangat baik. Dengan kategori _clothes_ memiliki nilai  F1-_score_ tertinggi yaitu sebesar 0.9836, artinya model mampu menangkap dan mengenali kategori _clothes_ dengan baik. Kemudian, kelas _trash_ dan _metal_ memiliki nilai prediksi sedikit lebih rendah dari semua kelas. 

----

## Konversi Model
Model yang sudah jadi akan di ekspor ke dalam beberapa format yaitu:
1. SavedModel : berguna menyimpan arsitektur model, bobot, serta konfigurasi training, sehingga model bisa langsung digunakan.
2. TFLite : berguna menyimpan model dalam ukuran yang lebih kecil dan ringan sehingga bisa digunakan untuk perangkat _mobile_ atau _edge_.
3. Tensorflowjs : berguna untuk mengubah model ke dalam format yang bisa dijalankan langsung di _browser_ menggunakan JavaScript.

----

## Inferensi Model
Tahap inferensi dilkakukan pada model untuk menguji performanya. Model yang telah dikonversi akan disimpan sebagai `model.tflite`, kemudian dijalankan dengan menggunakan bantuan `TFLite Interpreter`. Akan dilakukan uji dengan dataset yang sama yaiu "_Gabrbage Dataset_", diambil 12 sampel gambar secara acak dengan `sample_images = random.sample(all_images, 12)`. Selama proses prediksinya, gambar akan diubah ukuran menjadi 224×224 piksel, mengganti warna ke RGB, dan membagi nilai piksel menjadi 225 supaya muat dalam rentang 0-1. Kemudian gambar yang sudah di proses akan dimasukkan ke model dan dijlankan inferensi. hasil prediksi berupa _array_ akan dikeluarkan oleh `get_tensor()`, `np.argmax()` akan memilih indeks kelas dengan probabilitas tertinggi, dan `np.max()` akan mengambil nilai confidencenya. Hasil inferensi yang diberikan sangat baik, model mampu memprediksi gambar dengan sangat akurat.

![Inferensi Model](https://lh3.googleusercontent.com/d/1v7AVTxI4aKL98cwZlHogBPp6ytxMOQfX=w1000)

## Kesimpulan
Dari hasil uji coba melatih model dan inferensi model, arsitektur MobileNetV2 dengan teknik _fine-tuning_ berhasil memberikan performa yang sangat stabil dan akurat dalam melakukan klasifikasi gambar dengan nilai akurasi data _test_ sebesar 92.88%, data _train_ sebesar 94.39%, dan data _validation_ sebesar 91.13%
