# library
import cv2 as cv

# objek
cascade_path = (r"python-menengah\citra_digital\car_detection\cars.xml")
video_path = (r"python-menengah\citra_digital\car_detection\Cars Moving On Road Stock Footage - Free Download.mp4")
video = cv.VideoCapture(video_path)
acuan_obejek = cv.CascadeClassifier(cascade_path)

# function
def car_detect(frame):
    grayscalling = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    car = acuan_obejek.detectMultiScale(grayscalling, scaleFactor=1.1)
    return car

def drawer_box(frame):
    for x,y,w,h in car_detect(frame):
        cv.rectangle(frame, (x,y),(x+w,y+h),(255,0,0),4)

def close_window():
    video.release()
    cv.destroyAllWindows()
    exit()

def main():
    while True:
        _, frame = video.read()
        drawer_box(frame)
        cv.imshow("Deteksi mobil", frame)

        # mematikan video
        if cv.waitKey(1) & 0xFF == ord('q'):
            close_window()

# jalanin function
if __name__ == '__main__':
    main()


# deteksi eror
# if acuan_gambar.empty():
#     print(" haar cascade tidak bisa dimuat")
# else:
#     print(" haar cascade berhasil dimuat")