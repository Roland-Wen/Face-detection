import cv2
import glob
import insightface
import time
import numpy as np
from os.path import exists
from tqdm import tqdm
from insightface.app import FaceAnalysis
from insightface.model_zoo.model_zoo import get_model

def main():
    # Build face database
    print('Building database...')
    t0 = time.time()

    dbPath, exts = 'faces\\', ['*.png', '*.jpg']
    modelName = 'buffalo_sc'
    dbPathLen = len(dbPath)
    
    model = get_model(modelName)
    face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
    
    dbNames, dbFeats = [], []
    checkInTime, checkInTimeThreshold, simThreshold = {}, 5, 0.25

    # Load db features if possible
    if exists(dbPath + modelName + '_feats.npy') and exists(dbPath + modelName + '_names.txt'):
        print('Found existing database.')
        with open(dbPath + modelName + '_feats.npy', 'rb') as f:
            dbFeats = np.load(f)
        with open(dbPath + modelName + '_names.txt', 'r') as f:
            data = f.read()
            dbNames = data.split('\n')
        dbNames.pop()
        print(f'Loaded {len(dbNames)} people')
        
    else:
        print('Did not find existing database. Building a new one...')
        # Build db with the given faces
        app = FaceAnalysis()
        app.prepare(ctx_id=0, det_size=(640, 640))
        for ext in exts:
            for im_path in tqdm(glob.glob(dbPath + ext)):
                img = cv2.imread(im_path)
                faces = app.get(img)
                ok = False
                
                if len(faces) > 0:
                    box = faces[0].bbox.astype(np.int32)
                    if all(box >= 0):
                        img = img[box[1]:box[3], box[0]:box[2]]
                        ok = True
                    
                if not ok:
                    print(im_path[dbPathLen:-4], 'is bad')
                    continue
                dbFeats.append(model.get_feat(img))
                dbNames.append(im_path[dbPathLen:-4])
        # Save db for next time
        dbFeats = np.array(dbFeats)
        with open(dbPath + modelName + '_feats.npy', 'wb') as f:
            np.save(f, dbFeats)
        with open(dbPath + modelName + '_names.txt', 'w') as f:
            for name in dbNames: f.write(name + '\n')
        
    dbLen = dbFeats.shape[0]
    assert dbLen == len(dbNames), f'dbLen{dbLen} and dbNames{len(dbNames)} have different len'

    t1 = time.time()
    print('Finished building database.\nTime used:', t1 - t0)

    # Real time face recg
    cap = cv2.VideoCapture(0)
    while 1:
        # Read frame
        ret, frame = cap.read()
        if not ret:
            print('No frame')
            continue

        color = (0, 0, 255)
        text = 'N/A'

        # Detect face
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.1, 5)
        
        if len(faces) == 0:
            frame = cv2.putText(frame, text, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2, cv2.LINE_AA)
            cv2.imshow('frame', frame)
            if cv2.waitKey(1) == ord('q'):
                break
            continue

        color, text = (0, 255, 0), ''
        
        for currFace in faces:
            (x, y, w, h) = currFace
            if h >= 75 and w >= 75:
                face = frame[y:y+h, x:x+w]
                frame = cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 255), 2)
            
                # Calculate similarity
                testFeat = model.get_feat(face)
                mostSim = [-1.0, '']
                for i in range(dbLen):
                    sim = model.compute_sim(dbFeats[i], testFeat)
                    if sim > mostSim[0]:
                        mostSim = sim, dbNames[i]

                # Similar enough, count as a hit
                if mostSim[0] > simThreshold:
                    name = mostSim[1]
                    if name[-1:].isnumeric(): name = name[:-1]
                    currTime = time.time()
                        
                    # Check in and out
                    if name not in checkInTime.keys():
                        checkInTime[name] = [currTime, 0.0]
                        print(f'Hello, {name}')
                    elif checkInTime[name][1] < 1.0 and currTime - checkInTime[name][0] > checkInTimeThreshold:
                        checkInTime[name][1] = currTime
                        print(f'See you, {name}')

                    text = name + ' ' + '{0:.3g}'.format(mostSim[0] * 100) + '%'
                    frame = cv2.putText(frame, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2, cv2.LINE_AA)
                    
        # Show frame
        cv2.imshow('frame', frame)
        if cv2.waitKey(1) == ord('q'):
            break

    # Show record
    for ppl in checkInTime.keys():
        inStr = time.strftime("%Y/%m/%d %H:%M:%S", time.localtime(checkInTime[ppl][0]))
        print(f'{ppl},\nchecked in  at {inStr} and')

        if checkInTime[ppl][1] < 1.0:
            print('never checked out')
            continue

        outStr = time.strftime("%Y/%m/%d %H:%M:%S", time.localtime(checkInTime[ppl][1]))
        print(f'checked out at {outStr}')
        
    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
