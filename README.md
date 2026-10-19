# Bittle: Interaksi Suara dan Rangkaian Gerakan

Aplikasi Python untuk berbicara dengan robot Petoi Bittle dan menjalankan
rangkaian skill bawaan serta gerakan kepala khusus. Aplikasi utama adalah
`BittleLLM.py`; pengaturan saat ini menggunakan Whisper small pada CPU,
Groq untuk model bahasa, dan Bluetooth BLE untuk koneksi robot.
Profil opsional `tools/run_gpu.py` memakai Whisper medium pada GPU NVIDIA.

## Kredit proyek asal

Struktur awal proyek ini mengacu pada
[PetoiBittleChatGPT](https://github.com/ocolakoglu/PetoiBittleChatGPT) karya
[Ömer Çolakoğlu (@ocolakoglu)](https://github.com/ocolakoglu), yang menyediakan
dasar integrasi robot Petoi Bittle dengan ChatGPT.

## Struktur folder

```text
PettoiBittleLLM_lokal/
  BittleLLM.py          # aplikasi suara utama
  BittleLLM_1.py        # versi lama berbasis serial
  Commands.json         # daftar skill robot
  action_sequence.py    # perencanaan dan eksekusi rangkaian
  motion_gen.py         # gerakan kepala khusus
  head_templates.py     # pola lokal untuk permintaan kepala sederhana
  robot_link.py         # koneksi BLE / USB
  speechtotext.py       # pengenalan suara Whisper
  cuda_runtime.py       # pencarian library GPU dari venv aktif
  Text2Speech.py        # jawaban suara
  llm_utils.py          # model bahasa dan penghitungan token
  ardSerial.py          # dukungan protokol serial
  SerialCommunication.py
  config.py
  translate.py
  tests/                # tes otomatis tanpa perangkat / API
  tools/                # alat pemeriksaan manual
  logs/                 # log serial, dibuat saat aplikasi dijalankan
  test_sequence.py      # peluncur kompatibilitas ke tools/check_sequence.py
  test_motion.py        # peluncur kompatibilitas ke tools/check_motion.py
  Requirements.txt
  requirements-gpu.txt  # tambahan opsional untuk GPU NVIDIA di Windows
  README.md
  .env                  # konfigurasi lokal, diabaikan Git
  venv/                 # lingkungan Python lokal, diabaikan Git
```

Kode inti dipertahankan di root proyek agar impor dan urutan inisialisasi
model tidak berubah. `config.py`, `translate.py`, dan modul serial masih
dipakai jalur USB. Versi lama dipertahankan untuk referensi.

## Menyiapkan dan menjalankan aplikasi

Gunakan Python 3.11 (proyek ini memakai 3.11.9). Jalankan dari folder proyek:

```powershell
py -3.11 -m venv venv
.\venv\Scripts\Activate.ps1
python -m pip install -r Requirements.txt
```

Untuk Git Bash, aktivasi dengan `source venv/Scripts/activate`. Siapkan
`GROQ_API_KEY` di `.env` proyek. `BITTLE_BLE_ADDRESS` bersifat opsional; jika
tidak diisi, aplikasi mencari robot otomatis. Pemutaran jawaban suara juga
memerlukan FFmpeg terpasang di sistem, seperti dicatat di `Requirements.txt`.

```powershell
python BittleLLM.py
```

Untuk profil Whisper medium pada GPU NVIDIA di Windows, pasang komponen
tambahan ke venv proyek yang sudah aktif:

```powershell
python -m pip install -r requirements-gpu.txt
```

Library GPU ditemukan dari venv aktif. Jalur pencariannya hanya ditambahkan
untuk proses Python tersebut; tidak mengubah PATH Windows, driver NVIDIA,
atau memasang CUDA Toolkit global. Profil ini memerlukan model medium yang
sudah diunduh ke `models/` lewat alat perbandingan di bawah. Model tidak
diunduh otomatis saat aplikasi dijalankan.

Uji suara dahulu tanpa koneksi robot atau panggilan Groq:

```powershell
python tools/check_multilingual.py --guided --gpu --languages zh,hi,ar,fr
```

`--gpu` memilih medium pada CUDA dengan INT8/FP16; bahasa tetap dikenali
otomatis pada setiap ucapan. Untuk menjalankan aplikasi robot dengan profil ini:

```powershell
python tools/run_gpu.py
```

Menjalankan `BittleLLM.py` langsung tetap memakai small pada CPU. Jika profil
GPU gagal dimuat, program menampilkan error tanpa berpindah ke CPU diam-diam.

Pada uji lokal RTX 4050 6 GB tanggal 19 Oktober 2026, medium GPU memproses
rekaman berikutnya sekitar 0,5–0,8 detik setelah transkripsi pertama sekitar
2 detik; pemuatan model terpisah sekitar 4–5 detik. Empat WAV yang sama
memiliki median 6,36 detik pada medium CPU dan 0,72 detik pada GPU yang sudah
siap. Ini waktu transkripsi, belum termasuk merekam, Groq, dan pemutaran suara.
Enam rekaman Inggris/Spanyol/Portugis/Indonesia/Jepang/Korea mempertahankan
makna perintah. Arab dan Prancis juga terbaca lebih jelas daripada small pada
contoh sulit, tetapi Hindi masih salah pada kata duduk dan Mandarin masih
menghasilkan kata menggoyang. Hasil GPU tidak selalu sama dengan CPU;
sepuluh label bahasa yang benar tidak berarti seluruh kata sudah benar.

Lokasi `.env`, `Commands.json`, dan log ditentukan relatif terhadap folder
proyek, sehingga aplikasi tidak membaca konfigurasi dari folder kerja lain.
Log serial tersimpan di `logs/logfile.log`; sesi berikutnya menambahkan log.

## Menguji pengenalan suara

Setelah virtual environment aktif, jalankan `python speechtotext.py` untuk
menguji satu ucapan melalui Whisper yang dipakai aplikasi, tanpa menghubungkan
atau menggerakkan robot. `tools/check_stt_google.py` memakai Google Speech Recognition,
sehingga hasilnya tidak menguji jalur Whisper aplikasi utama.

Diam ketika muncul pesan kalibrasi mikrofon. Mulai bicara setelah `Talk...`.
Pada aplikasi utama, kalibrasi awal dilakukan sekali; ambang suara kemudian
tetap beradaptasi selama menunggu ucapan.

Aplikasi menyiapkan Whisper sebelum membuat koneksi robot. Pertahankan urutan
ini: pada lingkungan Windows proyek ini, memuat model setelah backend BLE/WinRT
dapat menyebabkan program berhenti dengan `Segmentation fault`/`access violation`.
Pesan `Whisper siap.` harus muncul sebelum proses koneksi robot.

Deteksi bahasa berlangsung otomatis untuk setiap ucapan. `STT_LANGUAGE = None`
tetap menjadi pengaturan aplikasi; pengguna bisa berganti bahasa antarucapan
tanpa memilih mode. Sepuluh kandidat di `language_policy.py` adalah Inggris (`en`),
Mandarin (`zh`), Hindi (`hi`), Spanyol (`es`), Arab Standar Modern (`ar`),
Prancis (`fr`), Portugis (`pt`), Indonesia (`id`), Jepang (`ja`), dan Korea (`ko`).
Kode Whisper `ar` mewakili bahasa Arab secara umum, bukan klasifikasi dialek
khusus. Balasan AI diarahkan memakai Arab Standar Modern/Fusha. Melayu dan
Afrikaans tidak lagi menjadi kandidat aplikasi.

Model yang sama mendeteksi probabilitas bahasa, memilih kandidat dalam daftar,
lalu mentranskripsikan audio sekali. Tidak ada tambahan model atau panggilan AI
untuk memilih bahasa. Ucapan di luar daftar yang terdeteksi kuat dengan kandidat
dukungan lemah diabaikan dan pengguna diminta mengucapkan ulang. Daftar ini
membatasi pilihan, tetapi tidak menjamin semua ucapan pendek selalu terdeteksi benar.

Jika seluruh teks cocok dengan frasa perintah yang dikenal dan tag bahasanya
kurang sesuai, program memeriksa tag itu berdasarkan bukti teks dan probabilitas
audio asli. Teks tidak diterjemahkan atau dipangkas. Bahasa terdeteksi
diteruskan ke instruksi Groq dan pemilihan suara gTTS. Tag pesan terbaru
didahulukan daripada bahasa riwayat; balasan Mandarin menggunakan aksara
Tionghoa sederhana, Hindi memakai Devanagari, dan Korea memakai Hangul.
Model Groq tetap `openai/gpt-oss-120b`, dengan temperature 0 untuk mengurangi
variasi keluaran. Tidak ada penguncian pada bahasa ucapan sebelumnya.

Petunjuk kosakata Indonesia/Inggris lama tetap dipakai pada kelompok bahasa
beraksara Latin. Untuk Mandarin, Hindi, Arab, Jepang, dan Korea, petunjuk itu
dimatikan karena pada rekaman uji dapat menggeser teks ke romanisasi atau
bahasa Inggris. Daftar kata asing tambahan tidak dipasang: pada eksperimen
Arab, decoder justru menyalin daftar tersebut.

Decoder memakai `WHISPER_TEMPERATURE = 0.0`, sehingga tidak mengulang sampling
melalui enam tingkat temperature saat keluaran sulit dikenali. Ini mengurangi
sumber lonjakan waktu, bukan batas waktu keras atau jaminan akurasi. Model
tetap `small`, CPU, INT8. `PAUSE_THRESHOLD_S = 1.3` memberi ruang jeda antarkata
sebelum rekaman berakhir; jeda lebih panjang masih dapat memisahkan ucapan.
Selesainya rekaman tetap bergantung pada suara dan mikrofon.

Hasil yang hanya tanda baca diabaikan.
Salah dengar yang sudah ditemukan (`Duh Duk`, `Doudouk`, dan pengulangannya)
dikoreksi ke `duduk` hanya jika seluruh ucapan cocok dan bahasanya Indonesia
atau deteksi bahasanya kurang yakin.
Kalimat seperti `jangan duduk` tetap utuh.
Angka keyakinan bahasa bukan ukuran ketepatan setiap kata. Akurasi audio tetap
perlu diuji menggunakan mikrofon dan kondisi ruangan yang sebenarnya.

Tes regresi suara tanpa mikrofon, robot, jaringan, atau unduhan model:
`python -B -m unittest tests.test_stt_pipeline tests.test_stt_startup -v`.
Tes alur otomatis sepuluh bahasa:
`python -B -m unittest tests.test_language_policy tests.test_multilingual_pipeline -v`.
Tes tiruan memeriksa alur dan pergantian bahasa; akurasi pengenalan setiap bahasa
tetap perlu diuji memakai rekaman nyata bahasa tersebut.

Untuk mencoba beberapa bahasa dalam satu sesi tanpa robot atau LLM:
`python tools/check_multilingual.py`. Tekan Enter sebelum tiap ucapan, lalu
bicara setelah `Talk...`; bahasa dipilih otomatis. Default lima ucapan;
`--samples 10` menambah jumlah contoh. Rekaman WAV dan metrik disimpan lokal
di `logs/measurements/`; `--no-audio` hanya menyimpan metrik/teks.

Untuk mengulang sepuluh kalimat acuan yang sama, jalankan
`python tools/check_multilingual.py --guided`. Tiap contoh menampilkan kalimat
sebelum Enter dan menyimpan teks serta bahasa acuan bersama hasil rekaman.
Bahasa tetap dideteksi dari audio; acuan tidak diberikan ke Whisper.
`--guided --samples 30` mengulang daftar tiga kali. Field
`language_matches_reference` hanya membandingkan label bahasa, bukan makna
atau ketepatan kata. Rekaman kosong/gagal tidak dihitung sebagai cocok.

Untuk memusatkan latihan pada sebagian contoh, gunakan
`python tools/check_multilingual.py --guided --languages zh,hi,ar,fr`.
Perintah ini menampilkan empat contoh dalam urutan Mandarin, Hindi, Arab,
Prancis. `--samples 8` mengulang empat contoh itu dua kali. Seleksi ini hanya
memilih kalimat latihan; bahasa ucapan tetap dikenali otomatis dari audio.

Metrik menyimpan `decoder_temperatures` dari segmen hasil serta
`mean_compression_ratio` untuk memeriksa pengulangan teks. Nilai yang tidak
tersedia tetap kosong/`null`. Potongan akhir kalimat yang tidak terekam perlu
diucapkan ulang; aplikasi tidak menambahkannya dari kalimat acuan.

Untuk memeriksa pemahaman teks Groq, tersedia 30 contoh (tiga per bahasa)
yang mencakup duduk, larangan duduk/berdiri, serta berjalan tiga detik lalu duduk:

```powershell
python tools/check_language_understanding.py --dry
python tools/check_language_understanding.py --run
```

`--dry` memeriksa contoh lokal tanpa API. `--run` memakai kuota Groq dengan
jeda 12 detik antarpanggilan dan berhenti jika terkena 429. Opsi
`--languages fr,pt,ko` membatasi bahasa contoh yang diuji, tanpa mengubah
deteksi otomatis aplikasi. Laporan berada di `logs/measurements/`.
Skor ini mengukur pemahaman aksi/urutan/durasi pada teks contoh, bukan
akurasi mikrofon, kefasihan balasan, atau jaminan konsistensi semua percakapan.

Untuk menilai apakah hasil suara yang meleset masih menghasilkan aksi yang
dimaksud, gunakan transkripsi dari sesi `--guided` yang sudah tersimpan:

```powershell
python tools/check_language_understanding.py --session latest --languages zh,hi,ar,fr --dry
python tools/check_language_understanding.py --session latest --languages zh,hi,ar,fr --run
```

`latest` memilih folder uji multibahasa terbaru; folder tersebut harus berisi
kalimat acuan dari mode `--guided`. Jalur folder sesi atau `utterances.jsonl`
juga bisa diberikan. Model menerima teks hasil Whisper dan label bahasa
aktualnya. Teks acuan hanya dipakai untuk menilai hasil, tidak diberikan kepada
model. Skor membandingkan rencana dengan maksud awal: berjalan tiga detik lalu
duduk. Jika Whisper mengubah arti ucapan, rencana bisa berbeda dari acuan
walaupun Groq mengikuti transkripsi yang diterimanya dengan benar.
Uji ini tidak merencanakan detail gerakan kepala dan tidak mengirim apa pun ke
robot. `--dry` tidak memakai API; `--run` memakai kuota Groq.

Untuk membandingkan model suara pada WAV yang sama tanpa merekam ulang:

```powershell
python tools/compare_whisper_models.py --session latest --languages zh,hi,ar,fr --dry
python tools/compare_whisper_models.py --session latest --languages zh,hi,ar,fr --run
```

Default menguji `small` dan `medium` secara bergantian pada CPU INT8.
`--run` hanya membaca cache lokal. Jika medium belum tersedia, tambahkan
`--download` pada perintah `--run` untuk mengizinkan unduhan ke `models/`
di dalam proyek; berkas model diabaikan Git. Cache small lama boleh dibaca
untuk menghindari unduhan ulang. `--device cuda` memerlukan library NVIDIA
yang sesuai dan memakai INT8/FP16 secara default.

Waktu muat dan waktu transkripsi dicatat terpisah. Teks/bahasa acuan tidak
diberikan ke decoder, dan hasil dinilai melalui teks asli; alat tidak membuat
skor makna otomatis atau mengubah model utama aplikasi. Hasil berada di
`logs/measurements/<waktu>-whisper-comparison/summary.json`. Model yang gagal
tetap dilaporkan sebagai gagal, dengan waktu/teks yang belum tersedia tidak
dibuat seolah-olah berhasil.

## Rangkaian gerakan

`BittleLLM.py` dapat menjalankan beberapa gerakan secara bergantian dari satu
ucapan. Rangkaian boleh mencampur skill dalam `Commands.json` dengan gerakan
kepala yang disusun AI. Contoh ucapan:

- "Duduk, lalu berdiri."
- "Duduk, tengok kanan, lalu berdiri."
- "Jalan selama tiga detik, lalu duduk."
- "Beri salam, gelengkan kepala, lalu istirahat."
- "Duduk, berdiri, lalu duduk lagi."

AI menghasilkan satu jawaban dan daftar aksi berurutan. Semua kode skill,
durasi, dan rencana kepala diperiksa sebelum gerakan pertama. Permintaan
kepala direncanakan per aksi, sehingga tengok kanan dan tengok kiri tidak
tercampur menjadi satu permintaan. Kepala kembali ke posisi netral setelah
setiap aksi kepala. Gerakan kepala baru tetap terbatas pada menoleh/geleng.
Pengaman arah kiri/kanan mengenali frasa dalam sepuluh bahasa dukungan;
batas sudut motor tetap diperiksa sebelum rencana dijalankan.

Sesudah kembali netral, program menunggu 0,2 detik, melepas kontrol manual
kepala dengan perintah `i`, lalu menunggu 0,3 detik sebelum aksi berikutnya.
Jeda setelah `i` memberi firmware waktu membaca perintah itu agar tidak
bergabung dengan skill berikutnya di buffer BLE. Nilai jeda tersedia di
`motion_gen.py` sebagai `HEAD_NEUTRAL_SETTLE_S` dan `HEAD_RELEASE_SETTLE_S`.
Jeda ini termasuk dalam perkiraan batas durasi rangkaian.

Gerakan berulang seperti berjalan, mundur, merangkak, dan berputar dihentikan
ke pose berdiri setelah jedanya sebelum lanjut ke aksi berikutnya. Pose akhir
seperti duduk atau istirahat tetap dipertahankan. Jika pengiriman gagal,
sisa antrean dibatalkan dan program mencoba menghentikan gerakan.

Batas pada `action_sequence.py`: maksimal 8 aksi, jeda skill 0,5–10 detik
(default 2 detik), perkiraan total gerak/jeda maksimal 60 detik. Ini bukan
batas waktu nyata: latensi koneksi dan perencanaan AI menambah waktu. BLE
belum membaca konfirmasi selesainya skill, sehingga jeda adalah perkiraan
yang perlu disesuaikan dengan firmware dan pengujian fisik. Aksi atraksi
mungkin memerlukan jeda lebih lama daripada pose sederhana.

Selama antrean berlangsung, aplikasi menunggu gerakan secara bergantian;
ucapan baru baru didengarkan sesudah antrean selesai. Tekan Ctrl+C untuk
membatalkan antrean saat berjalan.

Pratinjau melalui teks tanpa mikrofon/koneksi robot (tetap memakai API Groq):

```powershell
python tools/check_sequence.py --dry "duduk, tengok kanan, lalu berdiri"
```

Untuk benar-benar menjalankan permintaan teks pada robot:

```powershell
python tools/check_sequence.py --run "duduk, tengok kanan, lalu berdiri"
```

Opsi `--connection serial` memilih USB; default BLE. Di aplikasi suara,
`ACTION_DRY_RUN = True` hanya menampilkan aksi, tetapi aplikasi tetap membuat
koneksi robot saat startup. Untuk pratinjau tanpa koneksi gunakan `--dry` di atas.

Perintah lama `python test_sequence.py ...` dan `python test_motion.py ...`
tetap bisa dipakai melalui peluncur kompatibilitas di root.

## Penghematan token Groq

Kalimat instruksi utama dipertahankan; penulisan JSON memakai lebih sedikit
spasi. Seluruh alias `Commands.json`, sepuluh bahasa, format jawaban, aturan
negasi, urutan, pengulangan, durasi, dan batas gerakan tetap disertakan. Model Groq
tetap `openai/gpt-oss-120b` dengan temperature 0; batas riwayat tetap 20 pesan.

Setelah LLM utama memahami ucapan, permintaan kepala sederhana yang cocok
dengan satu frasa lengkap dapat memakai pola lokal di `head_templates.py`.
Contohnya `tengok kanan`, `tengok kiri`, dan `gelengkan kepala`. Pola ini
menghindari panggilan AI tambahan untuk aksi tersebut. Frasa tersedia dalam
sepuluh bahasa; variasi yang belum terdaftar tetap dapat dipahami oleh
perencana AI yang sudah ada.
Pola tengok standar memakai sudut 30 derajat dan waktu tahan 1 detik;
geleng memakai empat langkah bergantian -40/+40 derajat dengan tahan 0,4 detik.
Sudut memakai arah logis; koreksi tanda motor fisik tetap dilakukan saat eksekusi.

Permintaan dengan sudut, waktu tahan, kecepatan, jumlah, kondisi, negasi,
atau pola khusus tetap memakai perencana AI. Instruksi utama meminta detail
sudut, waktu tahan, kecepatan, pengulangan, dan urutan kepala disalin utuh
ke permintaan perencana. Pola lokal melalui
validator dan pemeriksaan arah yang sama. Batas motor, interpolasi, kembali
netral, pelepasan kontrol kepala, serta validasi seluruh rangkaian tetap berlaku.
Jika pola tidak sesuai batas yang sedang dikonfigurasi, pola tidak dipotong
diam-diam; perencana AI tetap menangani permintaan tersebut.

Perbandingan dengan snapshot prompt sebelum perubahan dapat memakai teks
yang sama, tanpa mikrofon, TTS, atau koneksi robot:

```powershell
python tools/compare_token_usage.py --dry
python tools/compare_token_usage.py --run --delay 15
```

Snapshot lokal dibaca dari `logs/measurements/*-token-optimization/baseline.json`
terbaru; `--baseline <path>` memilih snapshot tertentu di dalam proyek.
`--run` memakai kuota Groq. Laporan dan hasil parsial disimpan lokal di
`logs/measurements/<waktu>-token-comparison/`. Pemeriksaan mencakup sepuluh
bahasa, negasi, urutan, durasi, pengulangan, campuran kepala/skill, permintaan
khusus, obrolan, dan pengulangan gerakan dari konteks 20 pesan. Ketepatan
pengucapan, kefasihan balasan, dan gerakan fisik tetap dinilai terpisah.
Persentase penghematan dihitung dari metadata token provider pada pasangan
ucapan yang sama; hasilnya bergantung pada permintaan dan riwayat percakapan.

Pada 18 pasangan contoh teks tanggal 20 Oktober 2026, total token terukur
turun dari 28.644 menjadi 25.311 (hemat 11,6%), dengan 23 menjadi 19 panggilan.
Seluruh 18 rencana akhir sesuai acuan, termasuk sepuluh bahasa, pengulangan
dari konteks 20 pesan, dan kepala 25 derajat dengan tahan 0,8 detik.
Pada contoh duduk–tengok kanan–berdiri, token turun dari 2.070 menjadi 1.310
dan panggilan dari dua menjadi satu. Penghematan tiap ucapan dapat berbeda.
Angka tersebut menjumlahkan token masuk dan keluar yang dilaporkan provider.
Dampaknya terhadap batas token per menit juga dipengaruhi cache Groq;
metadata cache sebagian respons tidak tersedia.

## Pengukuran awal suara dan token

Jalankan dua sesi dengan delapan kalimat yang sama. Pengaturan Whisper tetap
`small`, CPU, INT8, dan deteksi otomatis pada sepuluh bahasa dukungan. Alat ini memakai prompt dan
perencana kepala aplikasi, tanpa koneksi robot atau pemutaran suara.

```powershell
python tools/measure_baseline.py --condition tenang --with-llm
python tools/measure_baseline.py --condition ramai --with-llm
```

Tekan Enter untuk menyiapkan setiap contoh, diam ketika mikrofon dikalibrasi,
lalu baca kalimat yang tampil persis setelah `Talk...`. Pertahankan perangkat
mikrofon, jarak, dan volume suara pada kedua sesi; ubah kondisi suara latarnya.
Beri jeda sekitar satu menit antar sesi agar penggunaan API sebelumnya sempat
pulih. `--with-llm` memakai kuota Groq termasuk panggilan tambahan untuk kepala.
Tanpa opsi tersebut, uji hanya merekam dan mentranskripsikan suara.

Setiap sesi disimpan di `logs/measurements/<waktu-WIB>-<kondisi>/`:

- `session.json`: pengaturan sesi, contoh referensi, waktu memuat model, dan status selesai.
- `utterances.jsonl`: hasil tiap contoh, waktu, rencana, serta token tiap panggilan AI.
- `summary.json`: kecocokan teks, kesalahan kata (WER), median/p95 waktu, dan total token.
- Berkas `.wav`: audio asli untuk perbandingan berikutnya pada rekaman yang sama.

Rekaman dan teks tersimpan lokal dan folder pengukuran diabaikan Git. Opsi
`--no-audio` menyimpan laporan tanpa WAV. Aplikasi biasa tidak menyimpan WAV.
`q` lalu Enter atau Ctrl+C menghentikan pengujian dan mempertahankan hasil yang
sudah selesai. `--repeat 3` mengulang seluruh daftar tiga kali untuk sampel lebih banyak.

Interpretasi pengukuran:

- Waktu tangkap meliputi menunggu mulai bicara dan jeda akhir. Durasi audio
  adalah panjang rekaman yang masuk ke Whisper; keduanya berbeda dari waktu transkripsi.
- Waktu transkripsi mencakup pembacaan seluruh hasil decoder. Pemuatan model
  dicatat terpisah. Contoh pertama belum dipanaskan; delapan contoh adalah uji
  awal, sehingga p95 masih sangat dipengaruhi satu contoh yang lambat.
- Skor `raw` memakai teks sebelum koreksi perintah, `normalized` setelah koreksi.
  Keduanya mengabaikan kapital dan tanda baca. WER menilai teks; pemeriksaan
  rencana terpisah menilai urutan, arah, negasi, dan durasi untuk contoh uji ini,
  bukan memastikan gerakan fisik robot. Penulisan angka `3` dan `tiga` dapat
  berbeda pada WER walaupun maknanya sama.
  Status rencana `ok` berarti rencana lolos parsing dan validasi; hasil
  `rencana sesuai contoh` menunjukkan apakah isinya cocok dengan contoh uji.
- Token per ucapan mencakup panggilan utama dan semua perencana kepala. Cache
  atau metadata yang tidak dikirim provider ditulis `null` (tidak diketahui).
  Permintaan gagal dapat memakai token tanpa mengembalikan hitungan lengkap.
- Angka 60 detik adalah pengamatan respons yang tercatat pada proses lokal,
  bukan sisa kuota Groq. Penggunaan dari aplikasi lain, retry internal SDK,
  waktu pencatatan server, dan kebijakan cache dapat menghasilkan angka berbeda.
- Metrik bahasa menyimpan hasil awal detektor, kandidat untuk decoder, bahasa
  akhir, dan alasan pemilihannya. Probabilitas tetap nilai asli, bukan skor yang
  dinaikkan setelah pembatasan kandidat. Pilihan bahasa pada decoder tidak
  dilaporkan sebagai keyakinan 100%.

Di `BittleLLM.py`, baris `[Ukur suara]` dan `[Ukur token]` juga muncul selama
pemakaian biasa. Pengukuran tidak menambah panggilan AI pada aplikasi utama.

## Tes otomatis

Semua tes otomatis berada di `tests/`. Jalankan seluruh suite tanpa
mikrofon, robot, API, atau unduhan model:

```powershell
python -B -m unittest discover -s tests -t . -v
python -B tools/check_motion.py --selftest
```

## Alat pemeriksaan manual

Semua alat berikut hanya melakukan pemeriksaan saat dijalankan langsung.
Mengimpor modulnya tidak membuka mikrofon, memanggil API, atau menggerakkan robot.

| Berkas | Fungsi |
| --- | --- |
| `tools/check_sequence.py` | Pratinjau atau menjalankan rangkaian dari input teks |
| `tools/measure_baseline.py` | Mengukur suara dan opsional token Groq tanpa robot |
| `tools/compare_token_usage.py` | Membandingkan token dan rencana aksi dengan snapshot sebelum perubahan |
| `tools/check_multilingual.py` | Mencoba pergantian bahasa otomatis tanpa robot atau LLM |
| `tools/run_gpu.py` | Menjalankan aplikasi robot dengan Whisper medium pada GPU NVIDIA |
| `tools/compare_whisper_models.py` | Membandingkan small/medium pada WAV tersimpan tanpa mengubah model aplikasi |
| `tools/check_language_understanding.py` | Memeriksa pemahaman perintah 10 bahasa; API hanya dengan `--run` |
| `tools/check_motion.py` | Self-test validator, pratinjau, atau menjalankan gerakan kepala |
| `tools/check_ble.py` | `--scan` mencari perangkat; tanpa opsi menguji duduk dan berdiri |
| `tools/check_move.py` | Menguji duduk melalui USB |
| `tools/check_serial.py` | Pemeriksaan beberapa perintah serial |
| `tools/check_groq.py` | Memanggil API Groq dengan kunci dari `.env` proyek |
| `tools/check_stt_google.py` | Merekam satu ucapan dan mengenalinya melalui Google |
| `tools/list_mics.py` | Menampilkan daftar mikrofon |

`__pycache__/`, berkas `.pyc`, dan log adalah hasil menjalankan program.
Aturan `.gitignore` mengabaikan berkas baru tersebut. Cache yang sudah pernah
dilacak Git masih ada dalam riwayat; perapian ini tidak mengubah riwayat Git.
