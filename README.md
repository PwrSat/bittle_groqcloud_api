# Bittle: Interaksi Suara dan Rangkaian Gerakan

Aplikasi Python untuk berbicara dengan robot Petoi Bittle dan menjalankan
rangkaian skill bawaan serta gerakan kepala khusus. Aplikasi utama adalah
`BittleLLM.py`; pengaturan saat ini menggunakan Whisper small pada CPU,
Groq untuk model bahasa, dan Bluetooth BLE untuk koneksi robot.

## Struktur folder

```text
PettoiBittleLLM_lokal/
  BittleLLM.py          # aplikasi suara utama
  BittleLLM_1.py        # versi lama berbasis serial
  Commands.json        # daftar skill robot
  action_sequence.py   # perencanaan dan eksekusi rangkaian
  motion_gen.py        # gerakan kepala khusus
  robot_link.py        # koneksi BLE / USB
  speechtotext.py       # pengenalan suara Whisper
  Text2Speech.py       # jawaban suara
  llm_utils.py         # model bahasa dan penghitungan token
  ardSerial.py         # dukungan protokol serial
  SerialCommunication.py
  config.py
  translate.py
  tests/               # tes otomatis tanpa perangkat / API
  tools/               # alat pemeriksaan manual
  logs/                # log serial, dibuat saat aplikasi dijalankan
  test_sequence.py     # peluncur kompatibilitas ke tools/check_sequence.py
  test_motion.py       # peluncur kompatibilitas ke tools/check_motion.py
  Requirements.txt
  README.md
  .env                 # konfigurasi lokal, diabaikan Git
  venv/                # lingkungan Python lokal, diabaikan Git
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

Di `speechtotext.py`, `STT_LANGUAGE = None` mempertahankan deteksi otomatis
multibahasa. Untuk sesi yang seluruhnya berbahasa Indonesia, ubah menjadi
`STT_LANGUAGE = "id"`, lalu jalankan ulang program. Jangan mengunci ke Indonesia
jika ingin berganti bahasa saat berbicara.

Decoder diberi petunjuk kosakata robot. Hasil yang hanya tanda baca diabaikan.
Salah dengar yang sudah ditemukan (`Duh Duk`, `Doudouk`, dan pengulangannya)
dikoreksi ke `duduk` hanya jika seluruh ucapan cocok dan bahasanya Indonesia
atau deteksi bahasanya kurang yakin. Kalimat seperti `jangan duduk` tetap utuh.
Angka keyakinan bahasa bukan ukuran ketepatan setiap kata. Akurasi audio tetap
perlu diuji menggunakan mikrofon dan kondisi ruangan yang sebenarnya.

Tes regresi suara tanpa mikrofon, robot, jaringan, atau unduhan model:
`python -B -m unittest tests.test_stt_pipeline tests.test_stt_startup -v`.

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
