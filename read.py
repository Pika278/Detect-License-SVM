import os
import cv2
import numpy as np
from lib_detection import load_model, detect_lp, im2single
# import csv
# import uuid
from flask import Flask, render_template, request, send_file
def read(img_path): 
        # Ham sap xep contour tu trai sang phai
    def sort_contours(cnts):
        reverse = False
        i = 0
        boundingBoxes = [cv2.boundingRect(c) for c in cnts]
        (cnts, boundingBoxes) = zip(*sorted(zip(cnts, boundingBoxes),
                                            key=lambda b: b[1][i], reverse=reverse))
        return cnts

    def sort_contours_two_line(cnts, method="left-to-right"):
        reverse = False
        i = 1
        if method == "right-to-left" or method == "bottom-to-top":
            reverse = True

        if method == "top-to-bottom" or method == "bottom-to-top":
            i = 0

        boundingBoxes = [cv2.boundingRect(c) for c in cnts]
        (cnts, boundingBoxes) = zip(*sorted(zip(cnts, boundingBoxes),
                                            key=lambda b: b[1][1] * Ivehicle.shape[1] + b[1][0] * Ivehicle.shape[0] +
                                                        b[1][1] * Ivehicle.shape[1], reverse=False))

        return (cnts)
    # Dinh nghia cac ky tu tren bien so
    char_list =  '0123456789ABCDEFGHKLMNPRSTUVXYZ'

    # Ham fine tune bien so, loai bo cac ki tu khong hop ly
    def fine_tune(lp):
        newString = ""
        for i in range(len(lp)):
            if lp[i] in char_list:
                newString += lp[i]
        return newString

    # Load model LP detection
    wpod_net_path = "wpod-net_update1.json"
    wpod_net = load_model(wpod_net_path)

        # Đọc file ảnh đầu vào
    Ivehicle = cv2.imread(img_path)
    

    # Kích thước lớn nhất và nhỏ nhất của 1 chiều ảnh
    Dmax = 608
    Dmin = 288

    # Lấy tỷ lệ giữa W và H của ảnh và tìm ra chiều nhỏ nhất
    ratio = float(max(Ivehicle.shape[:2])) / min(Ivehicle.shape[:2])

    side = int(ratio * Dmin)
    bound_dim = min(side, Dmax)
    _ , LpImg, lp_type = detect_lp(wpod_net, im2single(Ivehicle), bound_dim, lp_threshold=0.5)


    # Cau hinh tham so cho model SVM
    digit_w = 30 # Kich thuoc ki tu
    digit_h = 60 # Kich thuoc ki tu

    model_svm = cv2.ml.SVM_load('svm.xml')
    if (lp_type != 0 and len(LpImg)):

        # Chuyen doi anh bien so
        LpImg[0] = cv2.convertScaleAbs(LpImg[0], alpha=(255.0))
        roi = LpImg[0]
        height, width = roi.shape[:2]
        # Chuyen anh bien so ve gray
        gray = cv2.cvtColor( LpImg[0], cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (5, 5), 0)

        fil = cv2.bilateralFilter(blur, 11, 17, 17)

        # # Ap dung threshold de phan tach so va nen
        binary = cv2.adaptiveThreshold(fil, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 6)
        # cv2.imshow("Anh bien so sau threshold", binary)
        # cv2.waitKey()

        # Segment kí tự
        kernel3 = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        thre_mor = cv2.morphologyEx(binary, cv2.MORPH_DILATE, kernel3)
        cont, _  = cv2.findContours(thre_mor, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

        plate_info = ""
        if lp_type == 1:
            for c in sort_contours(cont):
                (x, y, w, h) = cv2.boundingRect(c)
                ratio = h/w
                if 1.5<=ratio<=3.5: # Chon cac contour dam bao ve ratio w/h
                    if h/roi.shape[0]>=0.6: # Chon cac contour cao tu 60% bien so tro len

                        # Ve khung chu nhat quanh so
                        cv2.rectangle(roi, (x, y), (x + w, y + h), (0, 255, 0), 2)

                        # Tach so va predict
                        curr_num = thre_mor[y:y+h,x:x+w]
                        curr_num = cv2.resize(curr_num, dsize=(digit_w, digit_h))
                        _, curr_num = cv2.threshold(curr_num, 30, 255, cv2.THRESH_BINARY)
                        curr_num = np.array(curr_num,dtype=np.float32)
                        curr_num = curr_num.reshape(-1, digit_w * digit_h)

                        # Dua vao model SVM
                        result = model_svm.predict(curr_num)[1]
                        result = int(result[0, 0])

                        if result<=9: # Neu la so thi hien thi luon
                            result = str(result)
                        else: #Neu la chu thi chuyen bang ASCII
                            result = chr(result)

                        plate_info +=result
        else:
            for c in sort_contours_two_line(cont):
                (x, y, w, h) = cv2.boundingRect(c)
                ratio = h/w
                if 1.5<=ratio<=3.5: # Chon cac contour dam bao ve ratio w/h
                    if h/roi.shape[0]>=0.35: # Chon cac contour cao tu 60% bien so tro len

                        # Ve khung chu nhat quanh so
                        cv2.rectangle(roi, (x, y), (x + w, y + h), (0, 255, 0), 2)

                        # Tach so va predict
                        curr_num = thre_mor[y:y+h,x:x+w]
                        curr_num = cv2.resize(curr_num, dsize=(digit_w, digit_h))
                        _, curr_num = cv2.threshold(curr_num, 30, 255, cv2.THRESH_BINARY)
                        curr_num = np.array(curr_num,dtype=np.float32)
                        curr_num = curr_num.reshape(-1, digit_w * digit_h)

                        # Dua vao model SVM
                        result = model_svm.predict(curr_num)[1]
                        result = int(result[0, 0])

                        if result<=9: # Neu la so thi hien thi luon
                            result = str(result)
                        else: #Neu la chu thi chuyen bang ASCII
                            result = chr(result)

                        plate_info +=result

        # Viet bien so len anh
        cv2.putText(Ivehicle,fine_tune(plate_info),(50, 50), cv2.FONT_HERSHEY_PLAIN, 3.0, (0, 0, 255), lineType=cv2.LINE_AA)

        #Luu ket qua
#         def save_results(plate_info,region,csv_filename,folder_path):
#             img_name = '{}.jpg'.format(uuid.uuid1())
#             cv2.imwrite(os.path.join(folder_path,img_name),region)
#             with open(csv_filename,mode = 'a',newline='') as f:
#                 csv_writer = csv.writer(f,delimiter=',',quotechar='"',quoting=csv.QUOTE_MINIMAL)
#                 csv_writer.writerow([img_name,plate_info])
#         save_results(plate_info,roi,'detect_results.csv',"detect_img")
        # Hien thi anh
        print("Bien so=", plate_info)
        cv2.imwrite('saved.jpg', Ivehicle)
        # cv2.imshow("Hinh anh output",Ivehicle)
        # cv2.waitKey()

        # cv2.imshow('object detection',  cv2.resize(image_np, (800, 600)))
        # if cv2.waitKey(10) & 0xFF == ord('q'):
        #     cap.release()
        #     cv2.destroyAllWindows()
        #     break   
    else:
        print("None")

    return binary, roi, plate_info

app=Flask(__name__,template_folder='template',static_folder='static')
app.config["UPLOAD_FOLDER"] = "static"
@app.route("/", methods=["POST", "GET"])
def index():
    if request.method == "GET":
        return render_template("index.html")
    else:
        # lay anh upload
        img_file = request.files["file"]
        path_to_save = os.path.join(app.config["UPLOAD_FOLDER"], "in_img.jpg")
        img_file.save(path_to_save)

        # dua qua model
        read(path_to_save)
        
        # tra ve ket qua
        return send_file("saved.jpg", mimetype="image/gif")
        # print()

if __name__ == "__main__":
    # port = int(os.environ.get('PORT', 5000))
    # app.run(host='0.0.0.0', port=port, debug=True)
    app.run(debug=True)
