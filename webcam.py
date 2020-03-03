import sys
from pathlib import Path
import time
import cv2


"""
    Usage: python webcam.py <name of person> <number of photos>
    Example usage: python webcam.py claudia 10
"""


def take_photos(person, num):
    """ Takes <num> number of photos and
        saves them to the images/<person> directory

    Args:
        person (str): Person's name
        num (int): Number of photos to take
    """

    webcam = cv2.VideoCapture(1)

    for i in range(num):
        try:
            check, frame = webcam.read()

            # Take photo
            if check:
                img_name = "images/%s/%s_%d.jpg" % (person, person, i)
                cv2.imwrite(filename=img_name, img=frame)
                # webcam.release()
                print("Image saved to %s!" % img_name)

        except KeyboardInterrupt:
            print("Turning off camera.")
            webcam.release()
            print("Camera off.")
            print("Program ended.")
            cv2.destroyAllWindows()
            break

        time.sleep(0.5)


def main():
    person = sys.argv[1]
    num = int(sys.argv[2])

    # Make directory for new person
    new_path = "images/" + person
    Path(new_path).mkdir(parents=True, exist_ok=True)

    take_photos(person, num)


if __name__ == "__main__":
    main()

