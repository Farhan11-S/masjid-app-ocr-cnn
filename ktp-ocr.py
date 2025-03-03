import re
import os
import cv2
import sys
import datetime
import numpy as np
import pytesseract
import textdistance
import pandas as pd
import time
import json

goodResultLatest = 0
goodResultOld = 0
# ROOT_PATH = os.getcwd()
ROOT_PATH = os.path.dirname(os.path.realpath(__file__))
LINE_REC_PATH = os.path.join(ROOT_PATH, 'data/ID_CARD_KEYWORDS.csv')
RELIGION_REC_PATH = os.path.join(ROOT_PATH, 'data/RELIGIONS.csv')
JENIS_KELAMIN_REC_PATH = os.path.join(ROOT_PATH, 'data/JENIS_KELAMIN.csv')
NEED_COLON = [3, 4, 6, 8, 10, 11, 12, 13, 14, 15, 17, 18, 19, 21]
NEXT_LINE = 9

def imageCropper(image, img_gray, filename, dir, goodResultLatest, goodResultOld):
    rectKernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 15))
    tophat = cv2.morphologyEx(img_gray, cv2.MORPH_TOPHAT, rectKernel)

    gradX = cv2.Sobel(tophat, ddepth=cv2.CV_32F, dx=1, dy=0, ksize=-1)
    gradX = np.absolute(gradX)
    (minVal, maxVal) = (np.min(gradX), np.max(gradX))
    gradX = (255 * ((gradX - minVal) / (maxVal - minVal)))
    gradX = gradX.astype("uint8")
    gradX = cv2.morphologyEx(gradX, cv2.MORPH_CLOSE, rectKernel)

    threshGradX = cv2.threshold(gradX, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
    threshGradX = cv2.morphologyEx(threshGradX, cv2.MORPH_CLOSE, rectKernel)

    threshCnts, hierarchy = cv2.findContours(threshGradX.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = threshCnts
    cur_img = image.copy()
    cv2.drawContours(cur_img, cnts, -1, (0, 0, 255), 3)
    copy = image.copy()

    locs = []
    nik = ""
    biodata = ""
    for (i, c) in enumerate(cnts):
        (x, y, w, h) = cv2.boundingRect(c)
        
        if (w > 360 and h > 240) or (w > 300 and h < 30):
            crop_img = image[y-4:y+h+4, x-4:x+w+4]
            gray = cv2.cvtColor(crop_img, cv2.COLOR_BGR2GRAY)
            blur = cv2.GaussianBlur(gray, (3, 3), 0)
            thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_TRUNC + cv2.THRESH_OTSU)[1]

            if w > 425 and h > 200:
                biodata = pytesseract.image_to_string(thresh, lang="Arialv2.4", config='--psm 4 --oem 3')
            
            if w > 300 and h < 30:
                nik = pytesseract.image_to_string(thresh, lang="OCRA-11-02-2024v1.4", config='--psm 7 --oem 3')


    return nik, biodata
                           
def writeToFile(ocrResult, filename, img):
    cv2.imwrite(filename+".png", img)
    with open(f"{filename}.gt.txt", "w") as text_file:
        text_file.write(ocrResult)     

def resultChecker(ocrResult, filename):
    txt_filename = f"{filename}.gt.txt"
    
    if os.path.exists(txt_filename):
        with open(txt_filename, 'r') as file:
            expected_result = file.read().strip()
            ocrResult = ocrResult.rstrip('\n')
            return ocrResult.strip() == expected_result
    return False

def listFiles(dir):
    r = []
    for root, dirs, files in os.walk("images/"+dir):
        for name in files:
            r.append(os.path.join(root, name))
    return r

def strip_op(result_raw):
    result_list = result_raw.split('\n')
    new_result_list = []

    for tmp_result in result_list:
        if tmp_result.strip(' '):
            new_result_list.append(tmp_result)

    return new_result_list

def biodataBuilder(biodata):
    raw_df = pd.read_csv(LINE_REC_PATH, header=None)
    result_list = strip_op(biodata)

    loc2index = dict()
    for i, tmp_line in enumerate(result_list):
        for j, tmp_word in enumerate(tmp_line.split(' ')):
            tmp_sim_list = [textdistance.damerau_levenshtein.normalized_similarity(tmp_word_, tmp_word.strip(':')) for tmp_word_ in raw_df[0].values]

            tmp_sim_np = np.asarray(tmp_sim_list)
            arg_max = np.argmax(tmp_sim_np)

            if tmp_sim_np[arg_max] >= 0.6:
                loc2index[(i, j)] = arg_max

    last_result_list = []
    useful_info = False

    for i, tmp_line in enumerate(result_list):
        tmp_list = []
        for j, tmp_word in enumerate(tmp_line.split(' ')):
            tmp_word = tmp_word.strip(':')

            if(i, j) in loc2index:
                useful_info = True
                if loc2index[(i, j)] == NEXT_LINE:
                    last_result_list.append(tmp_list)
                    tmp_list = []
                tmp_list.append(raw_df[0].values[loc2index[(i, j)]])
                if loc2index[(i, j)] in NEED_COLON:
                    tmp_list.append(':')
            elif tmp_word == ':' or tmp_word =='':
                continue
            else:
                tmp_list.append(tmp_word)

        if useful_info:
            if len(last_result_list) > 2 and ':' not in tmp_list:
                last_result_list[-1].extend(tmp_list)
            else:
                last_result_list.append(tmp_list)

    return biodataTransformer(last_result_list)

def biodataTransformer(last_result_list):
    religion_df = pd.read_csv(RELIGION_REC_PATH, header=None)
    jenis_kelamin_df = pd.read_csv(JENIS_KELAMIN_REC_PATH, header=None)
    provinsi = ""
    kabupaten = ""
    nama = ""
    tempat_lahir = ""
    tgl_lahir = ""
    jenis_kelamin = ""
    alamat = ""
    status_perkawinan = ""
    agama = ""
    kel_desa = "" 
    kecamatan = ""
    pekerjaan = ""
    kewarganegaraan = ""

    for tmp_data in last_result_list:
        if '—' in tmp_data:
            tmp_data.remove('—')

        if 'PROVINSI' in tmp_data:
            provinsi = ' '.join(tmp_data[1:])
            provinsi = re.sub('[^A-Z. ]', '', provinsi)

            if len(provinsi.split()) == 1:
                provinsi = re.sub('[^A-Z.]', '', provinsi)

        if 'KABUPATEN' in tmp_data or 'KOTA' in tmp_data:
            kabupaten = ' '.join(tmp_data[1:])
            kabupaten = re.sub('[^A-Z. ]', '', kabupaten)

            if len(kabupaten.split()) == 1:
                kabupaten = re.sub('[^A-Z.]', '', kabupaten)

        if 'Nama' in tmp_data:
            nama = ' '.join(tmp_data[2:])
            nama = re.sub('[^A-Z. ]', '', nama)

            if len(nama.split()) == 1:
                nama = re.sub('[^A-Z.]', '', nama)

        if 'Agama' in tmp_data:
            for tmp_index, tmp_word in enumerate(tmp_data[1:]):
                tmp_sim_list = [textdistance.damerau_levenshtein.normalized_similarity(tmp_word, tmp_word_) for tmp_word_ in religion_df[0].values]

                tmp_sim_np = np.asarray(tmp_sim_list)
                arg_max = np.argmax(tmp_sim_np)

                if tmp_sim_np[arg_max] >= 0.6:
                    tmp_data[tmp_index + 1] = religion_df[0].values[arg_max]
                    agama = tmp_data[tmp_index + 1]

        if 'Status' in tmp_data or 'Perkawinan' in tmp_data:
            try:
                status_perkawinan = ' '.join(tmp_data[2:])
                status_perkawinan = re.findall('\s+([A-Za-z]+)', status_perkawinan)
                status_perkawinan = ' '.join(status_perkawinan)
            except:
                status_perkawinan = ""

        if 'Alamat' in tmp_data:
            for tmp_index in range(len(tmp_data)):
                if "!" in tmp_data[tmp_index]:
                    tmp_data[tmp_index] = tmp_data[tmp_index].replace("!", "I")
                if "1" in tmp_data[tmp_index]:
                    tmp_data[tmp_index] = tmp_data[tmp_index].replace("1", "I")
                if "i" in tmp_data[tmp_index]:
                    tmp_data[tmp_index] = tmp_data[tmp_index].replace("i", "I")
                alamat = ' '.join(tmp_data[1:])
                alamat = re.sub('[^A-Z0-9. ]', '', alamat).strip()

                if len(alamat.split()) == 1:
                    alamat = re.sub('[^A-Z0-9.]', '', alamat).strip()

        # if 'RT/RW' in tmp_data:
        #     for tmp_index in range(len(tmp_data)):
        #         if "!" in tmp_data[tmp_index]:
        #             tmp_data[tmp_index] = tmp_data[tmp_index].replace("!", "1")
        #         if "i" in tmp_data[tmp_index]:
        #             tmp_data[tmp_index] = tmp_data[tmp_index].replace("i", "1")
        #         rt_rw = ' '.join(tmp_data[1:])
        #         rt_rw = re.search(r'\d{3}/\d{3}', rt_rw).group()

        if 'Kel/Desa' in tmp_data:
            for tmp_index in range(len(tmp_data)):
                if "!" in tmp_data[tmp_index]:
                    tmp_data[tmp_index] = tmp_data[tmp_index].replace("!", "I")
                if "1" in tmp_data[tmp_index]:
                    tmp_data[tmp_index] = tmp_data[tmp_index].replace("1", "I")
                if "i" in tmp_data[tmp_index]:
                    tmp_data[tmp_index] = tmp_data[tmp_index].replace("i", "I")
                kel_desa = ' '.join(tmp_data[1:])
                kel_desa = re.sub('[^A-Z0-9. ]', '', kel_desa).strip()

                if len(kel_desa.split()) == 1:
                    kel_desa = re.sub('[^A-Z0-9.]', '', kel_desa).strip()

        if 'Kecamatan' in tmp_data:
            for tmp_index in range(len(tmp_data)):
                if "!" in tmp_data[tmp_index]:
                    tmp_data[tmp_index] = tmp_data[tmp_index].replace("!", "I")
                if "1" in tmp_data[tmp_index]:
                    tmp_data[tmp_index] = tmp_data[tmp_index].replace("1", "I")
                if "i" in tmp_data[tmp_index]:
                    tmp_data[tmp_index] = tmp_data[tmp_index].replace("i", "I")
                kecamatan = ' '.join(tmp_data[1:])
                kecamatan = re.sub('[^A-Z0-9. ]', '', kecamatan).strip()

                if len(kecamatan.split()) == 1:
                    kecamatan = re.sub('[^A-Z0-9.]', '', kecamatan).strip()

        if 'Jenis' in tmp_data or 'Kelamin' in tmp_data:
            for tmp_index, tmp_word in enumerate(tmp_data[2:]):
                tmp_sim_list = [textdistance.damerau_levenshtein.normalized_similarity(tmp_word, tmp_word_) for tmp_word_ in jenis_kelamin_df[0].values]

                tmp_sim_np = np.asarray(tmp_sim_list)
                arg_max = np.argmax(tmp_sim_np)

                if tmp_sim_np[arg_max] >= 0.6:
                    tmp_data[tmp_index + 2] = jenis_kelamin_df[0].values[arg_max]
                    jenis_kelamin = tmp_data[tmp_index + 2]

        if 'Pekerjaan' in tmp_data:
            pekerjaan = ' '.join(tmp_data[2:])
            pekerjaan = re.sub('[^A-Za-z./ ]', '', pekerjaan)

            if len(pekerjaan.split()) == 1:
                pekerjaan = re.sub('[^A-Za-z./]', '', pekerjaan)
                                    
        if 'Kewarganegaraan' in tmp_data:
            kewarganegaraan = ' '.join(tmp_data[2:])
            kewarganegaraan = re.sub('[^A-Z. ]', '', kewarganegaraan)

            if len(kewarganegaraan.split()) == 1:
                kewarganegaraan = re.sub('[^A-Z.]', '', kewarganegaraan)

        if 'Tempat' in tmp_data or 'Tgl' in tmp_data or 'Lahir' in tmp_data:
            join_tmp = ' '.join(tmp_data)

            match_tgl1 = re.search("([0-9]{2}—[0-9]{2}—[0-9]{4})", join_tmp)
            match_tgl2 = re.search("([0-9]{2}\ [0-9]{2}\ [0-9]{4})", join_tmp)
            match_tgl3 = re.search("([0-9]{2}\-[0-9]{2}\ [0-9]{4})", join_tmp)
            match_tgl4 = re.search("([0-9]{2}\ [0-9]{2}\-[0-9]{4})", join_tmp)
            match_tgl5 = re.search("([0-9]{2}-[0-9]{2}-[0-9]{4})", join_tmp)
            match_tgl6 = re.search("([0-9]{2}\-[0-9]{2}\-[0-9]{4})", join_tmp)
            
            if match_tgl1:
                try:
                    tgl_lahir = datetime.datetime.strptime(match_tgl1.group(), '%d—%m—%Y').date()
                    tgl_lahir = tgl_lahir.strftime('%d-%m-%Y')
                except:
                    tgl_lahir = ""
            elif match_tgl2:
                try:
                    tgl_lahir = datetime.datetime.strptime(match_tgl2.group(), '%d %m %Y').date()
                    tgl_lahir = tgl_lahir.strftime('%d-%m-%Y')
                except:
                    tgl_lahir = ""
            elif match_tgl3:
                try:
                    tgl_lahir = datetime.datetime.strptime(match_tgl3.group(), '%d-%m %Y').date()
                    tgl_lahir = tgl_lahir.strftime('%d-%m-%Y')
                except:
                    tgl_lahir = ""
            elif match_tgl4:
                try:
                    tgl_lahir = datetime.datetime.strptime(match_tgl4.group(), '%d %m-%Y').date()
                    tgl_lahir = tgl_lahir.strftime('%d-%m-%Y')
                except:
                    tgl_lahir = ""
            elif match_tgl5:
                try:
                    tgl_lahir = datetime.datetime.strptime(match_tgl5.group(), '%d-%m-%Y').date()
                    tgl_lahir = tgl_lahir.strftime('%d-%m-%Y')
                except:
                    tgl_lahir = ""
            elif match_tgl6:
                try:
                    tgl_lahir = datetime.datetime.strptime(match_tgl6.group(), '%d-%m-%Y').date()
                    tgl_lahir = tgl_lahir.strftime('%d-%m-%Y')
                except:
                    tgl_lahir = ""
            else:
                tgl_lahir = ""
                
            try:
                tempat_lahir = ' '.join(tmp_data[2:])
                tempat_lahir = re.findall("[A-Z\s]", tempat_lahir)
                tempat_lahir = ''.join(tempat_lahir).strip()
            except:
                tempat_lahir = ""

    return {
        'nama': nama,
        'tempat_lahir': tempat_lahir,
        'tgl_lahir': tgl_lahir,
        'jenis_kelamin': jenis_kelamin,
        'agama': agama,
        'status_perkawinan': status_perkawinan,
        'provinsi': provinsi,
        'kabupaten': kabupaten,
        'alamat': alamat,
        'kel_desa': kel_desa,
        'kecamatan': kecamatan,
        'pekerjaan': pekerjaan,
        'kewarganegaraan': kewarganegaraan
    }

def readImage(filename):
    npimg = np.fromfile(filename, np.uint8)
    #         break
    image = cv2.imdecode(npimg, cv2.IMREAD_COLOR)
    image = cv2.resize(image, (50 * 16, 500))
    img_gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    rectKernel = cv2.getStructuringElement(cv2.MORPH_RECT, (25, 7))
    blackhat = cv2.morphologyEx(img_gray, cv2.MORPH_BLACKHAT, rectKernel)

    return image, blackhat

if __name__ == '__main__':
    start_time = time.time()
    filename = sys.argv[1]

    try:
        image, blackhat = readImage(filename)
        nik, biodata = imageCropper(image, blackhat, filename, "1", goodResultLatest, goodResultOld)

        if nik or biodata:
            result = biodataBuilder(biodata)
            result['nik'] = nik.strip('\n')
            finish_time = time.time() - start_time
            result['time_elapsed'] = str(round(finish_time, 3))
            print(json.dumps(result, indent=None, separators=(',', ':')))
        else:
            print("NIK not found")
    except Exception as e:
        print("err : " + str(e))

    sys.exit(0)

    