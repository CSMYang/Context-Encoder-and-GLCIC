import cv2
import os
from tqdm import tqdm
import youtube_dl

video_path = "last_jedi.mkv"
directory_path = "C:/Users/chuan/Desktop/CSC413-Project/with_subtitle"
img_name = 'with_subtitle'
starting_frame = 0


def download_video(url, dest_dir, sub=True):
    """
    Download YouTube video to dest_dir from url.
    """
    video_path = dest_dir + '%(title)s_%(ext)s.mp4'
    ydl_opts = {
        'outtmpl': video_path,
        'format': ' (bestvideo[width<=720][ext=mp4]/bestvideo)+bestaudio/best',
        'writesubtitles': sub,
        'writeautomaticsub': sub,
        # 'convertsubtitles': sub,
        # 'listsubtitles': sub,
        # 'allsubtitles': sub,
        'subtitlesformat': 'srt',
        'subtitleslangs': ["en"],
        # 'postprocessor_args': ['embed-subs'],
        'postprocessors': [
            # {'key': 'FFmpegSubtitlesConvertor', 'format': 'srt'},
            {'key': 'FFmpegEmbedSubtitle'}],
        'keepvideo': True
    }
    with youtube_dl.YoutubeDL(ydl_opts) as ydl:
        ydl.download([url])
    print("Download Successful!")


def build_dataset(video_path, dest_dir, num, start=0):
    """
    https://www.geeksforgeeks.org/extract-images-from-video-in-python/
    """

    # Read the video from specified path
    cam = cv2.VideoCapture(video_path)
    try:
        # creating a folder named data
        if not os.path.exists(dest_dir):
            os.makedirs(dest_dir)
    except OSError:
        print('Error: Failed to creat directory')

    pbar = tqdm(num)
    current_frame = 0
    while pbar.n <= num:
        # reading from frame
        ret, frame = cam.read()

        if current_frame < start:
            current_frame += 1
            continue

        if ret:
            img = dest_dir + "/" + str(pbar.n) + '.png'
            cv2.imwrite(img, frame)
            pbar.set_description('creating dataset | %d.png' % pbar.n)
            pbar.update()
        else:
            break

    pbar.close()
    # Release all space and windows once done
    cam.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    video_stream = cv2.VideoCapture(video_path)
    video_stream.set(cv2.CAP_PROP_POS_FRAMES, starting_frame)
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
