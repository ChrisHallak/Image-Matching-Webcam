import cv2
from PIL import Image
from deepface import DeepFace


userImage = 'images/user1.jpg'
userImage2 = 'images/user2.jpg'
userImage3 = 'images/user3.jpg'


def check_matching(img1,img2):
    res = DeepFace.verify(img1,img2,model_name='ArcFace',enforce_detection=False)
    return res['verified'],res['distance']


if __name__ == __name__:
    # imageNumpy = cv2.imread(userImage)
    # print(imageNumpy)
    # cv2.imshow('Chris',imageNumpy)

    ref_image = cv2.imread(userImage3)
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT,430)

    match = False
    counter = 0
    similarity = 0

    while True:
        ret, frame = cap.read()
        if(ret):
            if counter % 30 == 0:
                isVerified, distance = check_matching(ref_image,frame)
                if isVerified:
                    match = True
                    similarity = max(0,(1 - distance) * 100)
                else:
                    match = False
            counter+=1
        else:
            pass

        if match:
            cv2.putText(frame,f'MATCH {similarity}',org=(50,450),fontFace=cv2.FONT_HERSHEY_SIMPLEX,color=(0,255,0),thickness=3,fontScale=2)
        else :
            cv2.putText(frame,'No MATCH',org=(50,450),fontFace=cv2.FONT_HERSHEY_SIMPLEX,color=(0,0,255),thickness=3,fontScale=2)
        cv2.imshow('image',frame)


        key = cv2.waitKey(1)
        if key == ord('q'):
            break

    cv2.destroyAllWindows()