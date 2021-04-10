import cv2
import os

video_path = "last_jedi.mkv"
directory_path = "C:/Users/chuan/Desktop/CSC413-Project/with_subtitle"
img_name = 'with_subtitle'


if __name__ == "__main__":
    video_stream = cv2.VideoCapture(video_path)
    os.chdir(directory_path)
    i = 0
    while cv2.waitKey(1) < 0:
        has_frame, current_frame = video_stream.read()
        if has_frame:
            cv2.imwrite(img_name + '_{}'.format(i) + '.jpg', current_frame)
            i += 1
        else:
            print('End of video reached!')
            cv2.waitKey(100)
            break
