import cv2
import json
import numpy as np
import ocr
import time
import cnn_detect
from PIL import Image
import os

os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

def upload_file(image_path):
    start_time = time.time()

    try:
        fileimage = Image.open(image_path)

        npimg = np.fromfile(image_path)
        image = cv2.imdecode(npimg, cv2.IMREAD_COLOR)
        isimagektp = cnn_detect.main(fileimage)

        if isimagektp:
            (nik, nama, tempat_lahir, tgl_lahir, jenis_kelamin, agama,
            status_perkawinan, provinsi, kabupaten, alamat, rt_rw, 
            kel_desa, kecamatan, pekerjaan, kewarganegaraan) = ocr.main(image)

            finish_time = time.time() - start_time

            if not nik:
                return json.dumps({
                    'error': True,
                    'message': 'Resolusi foto terlalu rendah, silakan coba lagi.'
                })

            return json.dumps({
                'error': False,
                'message': 'Proses OCR Berhasil',
                'result': {
                    'nik': str(nik),
                    'nama': str(nama),
                    'tempat_lahir': str(tempat_lahir),
                    'tgl_lahir': str(tgl_lahir),
                    'jenis_kelamin': str(jenis_kelamin),
                    'agama': str(agama),
                    'status_perkawinan': str(status_perkawinan),
                    'pekerjaan': str(pekerjaan),
                    'kewarganegaraan': str(kewarganegaraan),
                    'alamat': {
                        'name': str(alamat),
                        'rt_rw': str(rt_rw),
                        'kel_desa': str(kel_desa),
                        'kecamatan': str(kecamatan),
                        'kabupaten': str(kabupaten),
                        'provinsi': str(provinsi)
                    },
                    'time_elapsed': str(round(finish_time, 3))
                }
            })
        else:
            return json.dumps({
                'error': True,
                'message': 'Foto yang diunggah haruslah foto E-KTP'
            })
    except Exception as e:
        print("Error: ", e)
        return json.dumps({
            'error': True,
            'message': 'Maaf, KTP tidak terdeteksi'
        })

if __name__ == "__main__":
    outputReal = upload_file("images/tes6.jpg")
    print(outputReal)
