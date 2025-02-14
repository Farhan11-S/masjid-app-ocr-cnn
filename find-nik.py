import cv2
import numpy as np
import os
import pytesseract


def nikFinder(image, img_gray, filename, dir, grl = 0, gro = 0):
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
    for (i, c) in enumerate(cnts):
        (x, y, w, h) = cv2.boundingRect(c)

        # ar = w / float(h)
        # if ar > 3:
        # if (w > 40 ) and (h > 10 and h < 20):
        if h > 10 and w > 100 and x < 300 and x > 4:
            try:
                crop_img = image[y-4:y+h+4, x-4:x+w+4]
                gray = cv2.cvtColor(crop_img, cv2.COLOR_BGR2GRAY)
                blur = cv2.GaussianBlur(gray, (3, 3), 0)
                # Testing Biodata
                thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_TRUNC + cv2.THRESH_OTSU)[1]

                # Testing NIK
                # thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]

                # Testing Biodata
                aocra_old = pytesseract.image_to_string(thresh, lang="Arialv2.4", config='--psm 4 --oem 3')
                aocra_latest = pytesseract.image_to_string(thresh, lang="Arialv2.5", config='--psm 4 --oem 3')
                
                # Testing NIK
                # aocra_old = pytesseract.image_to_string(thresh, lang="OCRA-NEW", config='--psm 7 --oem 3')
                # aocra_latest = pytesseract.image_to_string(thresh, lang="OCRA-11-02-2024v1.4", config='--psm 7 --oem 3')
                whitespace_count = aocra_latest.count(' ')
                
                newfilename = os.path.splitext(filename)[0]
                newfilename = newfilename.split("images/"+dir+"\\")[-1]
                newfilename = f"{dir}/{dir}-{newfilename}"
                if len(aocra_latest) > 11 and len(aocra_latest) < 20 and whitespace_count <= 0:
                    # writeToFile(aocra_latest, "ocred/"+newfilename, thresh, dir)
                    if resultChecker(aocra_latest, "ground_truth/"+newfilename):
                        grl += 1
                if len(aocra_old) > 11 and len(aocra_old) < 20 and whitespace_count <= 0:
                    # writeToFile(aocra_latest, "ocred/"+newfilename, thresh, dir)
                    if resultChecker(aocra_old, "ground_truth/"+newfilename):
                        gro += 1
                    return grl, gro
            
                # if len(aocra_old) > 11 and len(aocra_old) < 20 and whitespace_count <= 0:
                # # writeToFile(aocra_old, "ocred/"+newfilename, thresh, dir)
                #     if resultChecker(aocra_old, "ground_truth/"+newfilename):
                #         gro += 1
            except e:
                print("err " + e)
                           
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
            
if __name__ == '__main__':
    dirs = ["3"]
    goodResultLatest = 0
    goodResultOld = 0
    for dir in dirs:
        files = listFiles(dir)
        for filename in files:
            try:
                npimg = np.fromfile(filename, dtype='uint8')
                image = cv2.imdecode(npimg, cv2.IMREAD_COLOR)
                image = cv2.resize(image, (50 * 16, 500))
                img_gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
                rectKernel = cv2.getStructuringElement(cv2.MORPH_RECT, (25, 7))
                blackhat = cv2.morphologyEx(img_gray, cv2.MORPH_BLACKHAT, rectKernel)
                goodResultLatest, goodResultOld = nikFinder(image, blackhat, filename, dir, goodResultLatest, goodResultOld)

            except Exception as e:
                continue
    print(f"Good Result Latest: {str(goodResultLatest)}")
    print(f"Good Result Old: {str(goodResultOld)}")
    