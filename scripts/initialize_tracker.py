"""modified from Adam Calhoun's version"""

import numpy as np
import cv2
import copy
try: # python 2
    import tkFileDialog as filedialog
except ImportError: # python 3
    import tkinter.filedialog as filedialog
import os.path
import sys
import yaml
# step one: load fly video
# step two: choose area that contains fly arena
# step three: select arena width
# step four: align microphone 1
# step three: choose area that shows blinky light
# select center position
# select initial flies (M/F)
# choose mating frame


def updateImage(x, y):
    global arenaX, arenaY, rectRadius, rectCenterX, rectCenterY, showFrame

    xdelta = abs(arenaX - x)
    ydelta = abs(arenaY - y)
    delta = max(xdelta, ydelta)

    rectRadius = delta / scaling
    rectCenterX = arenaX
    rectCenterY = arenaY

    img = copy.copy(showFrame)
    cv2.circle(img, (arenaX, arenaY), delta, (0, 255, 0), 3)
    cv2.imshow('display', img)


def areaCallback(event, x, y, flags, param):
    global currentlyDrawing, arenaX, arenaY

    if event == cv2.EVENT_LBUTTONDOWN and currentlyDrawing == False:
        arenaX = x
        arenaY = y

    if event == cv2.EVENT_LBUTTONDOWN:
        currentlyDrawing = True
    elif event == cv2.EVENT_LBUTTONUP:
        currentlyDrawing = False

    if currentlyDrawing:
        updateImage(x, y)


def updateCenterImage(x, y):
    global arenaX, arenaY, showFrame

    # img = copy.copy(showFrame)
    cv2.circle(showFrame, (x, y), 4, (0, 255, 0), 1)
    cv2.imshow('display', showFrame)
    cv2.waitKey(1)


def updateCircleImage(x, y, flags):
    global arenaX, arenaY, showFrame

    img = copy.copy(showFrame)
    leftX = flags[0]
    leftY = flags[1]
    rightX, rightY = arenaX, arenaY
    dX = (leftX - rightX) / 2
    dY = (leftY - rightY) / 2
    centerX = int(leftX - dX)
    centerY = int(leftY - dY)
    radius = int(np.sqrt((leftX - rightX) ** 2 + (leftY - rightY) ** 2) / 2)
    # show detected circle
    cv2.circle(img, (centerX, centerY), int(radius),(0, 255, 0), 3)
    frame2 = cv2.resize(img, (int(width / scaling), int(height / scaling)))
    cv2.imshow('display', img)

    # cv2.circle(showFrame, (x, y), 5, (0, 255, 0), 3)
    # cv2.imshow('display', showFrame)
    cv2.waitKey(1)


def centerCallback(event, x, y, flags, param):
    global currentlyDrawing, arenaX, arenaY

    if event == cv2.EVENT_LBUTTONDOWN and currentlyDrawing == False:
        arenaX = x
        arenaY = y

    if event == cv2.EVENT_LBUTTONDOWN:
        currentlyDrawing = True
        arenaX = x
        arenaY = y
    if event == cv2.EVENT_LBUTTONUP:
        currentlyDrawing = False

    if currentlyDrawing:
        updateCenterImage(x, y)


def circleCallback(event, x, y, flags, param):
    global currentlyDrawing, arenaX, arenaY

    if event == cv2.EVENT_LBUTTONDOWN and currentlyDrawing == False:
        arenaX = x
        arenaY = y

    if event == cv2.EVENT_LBUTTONDOWN:
        currentlyDrawing = True
        arenaX = x
        arenaY = y
    if event == cv2.EVENT_LBUTTONUP:
        currentlyDrawing = False

    if currentlyDrawing:
        # updateCenterImage(x, y)
        updateCircleImage(x, y, param)



def findFliesCallback(event, x, y, flags, param):
    global nFlies, fly1CenterX, fly1CenterY, fly2CenterX, fly2CenterY

    if event == cv2.EVENT_LBUTTONUP:
        nFlies += 1
        print(f'fly {nFlies} = ({x}, {y})')
        updateCenterImage(x, y)
        flyCenters[nFlies - 1, 0] = x
        flyCenters[nFlies - 1, 1] = y


def trackerCallback(position):
    global angle

    M = cv2.getRotationMatrix2D((width / 4, height / 4), position - 180, 1)
    angle = position
    img = copy.copy(showFrame)
    img = cv2.warpAffine(img, M, (int(width / scaling), int(height / scaling)))
    cv2.imshow('display', img)


def doNothing(event, x, y, flags, param):
    pass


def getStartTime(dirName):
    trackingFile = os.path.join(dirName, 'startTrackingFrame.txt')
    if (os.path.isfile(trackingFile)):
        f = open(trackingFile)
        startFrame = int(f.readline())
        print('starting frame is ' + str(startFrame))
        f.close()
        return startFrame
    else:
        return 1


def convert(filename):
    """Convert old annotation format to dict"""
    annotation = dict()
    with open(filename, 'r') as f:
        annotation['width'] = int(f.readline().split(',')[1].strip())
        annotation['height'] = int(f.readline().split(',')[1].strip())
        annotation['centerX'] = float(f.readline().split(',')[1].strip())
        annotation['centerY'] = float(f.readline().split(',')[1].strip())
        annotation['radius'] = float(f.readline().split(',')[1].strip())
        annotation['rectCenterX'] = float(f.readline().split(',')[1].strip())
        annotation['rectCenterY'] = float(f.readline().split(',')[1].strip())
        annotation['rectRadius'] = float(f.readline().split(',')[1].strip())
        annotation['start_frame'] = float(f.readline().split(',')[1].strip())
        annotation['angle'] = float(f.readline().split(',')[1].strip())
        data = f.read(1)
        flypos = []

        index = 0
        annotation['nFlies'] = int(f.readline().split(',')[1].strip())


        flypos = []
        while data!='':
            data = f.readline()
            if (data == ''):
                break
            flyXY = [None]*2
            flyXY[0] = float(data)

            data = f.readline()
            if (data!=''):
                flyXY[1] = float(data)
            flypos.append(flyXY)

        annotation['flypositions'] = flypos
    return annotation


def save(filename, dict):
    """Save dict as YAML file."""
    with open(filename, 'w') as outfile:
        yaml.dump(dict, outfile, default_flow_style=False)


def load(filename):
    """Read dict from YAML"""
    with open(filename, 'r') as stream:
        data_loaded = yaml.load(stream)
    return data_loaded


# these are the global variables that we want to assign (make these cmd args?)
doDefineArena = True
doAlignMike = False
doBlinky = True
currentlyDrawing = False
scaling = 2.0
arenaX = 0
arenaY = 0
delta = 0
rectCenterX = 0
rectCenterY = 0
rectRadius = 0
angle = 0
nFlies = 0
flyCenters = np.zeros((100, 2))
fly1CenterX = 0
fly1CenterY = 0
fly2CenterX = 0
fly2CenterY = 0

# do the damage
if len(sys.argv) > 1:
    movieName = sys.argv[1]
else:
    movieName = filedialog.askopenfilename()
print(movieName)
# startFrame = getStartTime(movieName[0:(movieName.rindex('/')+1)])
if len(sys.argv) > 2:
    startFrame = int(sys.argv[2])
else:
    startFrame = 1000#getStartTime(os.path.dirname(movieName))

vr = cv2.VideoCapture(movieName, cv2.CAP_FFMPEG)
# v = cv2.VideoCapture('/data/out', cv2.CAP_FFMPEG)
vr.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('H', '2', '6', '4'))
NumberOfFrames = int(vr.get(cv2.CAP_PROP_FRAME_COUNT))
fps = vr.get(cv2.CAP_PROP_FPS)
width = int(vr.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(vr.get(cv2.CAP_PROP_FRAME_HEIGHT))

if (startFrame != 1):
    vr.set(cv2.CAP_PROP_POS_FRAMES, startFrame + 1)
ret, frame = vr.read()

# fixed issue with reading early frames on windows
if np.max(frame)==0:
    vr.set(cv2.CAP_PROP_POS_FRAMES, np.min((100, NumberOfFrames)))
    vr.read()
    vr.set(cv2.CAP_PROP_POS_FRAMES, startFrame + 1)
    ret, frame = vr.read()

if not ret:
    print("error reading frame")

cv2.namedWindow('display')
cv2.resizeWindow('display', width, height)


if doDefineArena:

    # choose arena width
    frame2 = copy.copy(frame)
    cv2.putText(frame2, 'Select LEFT boundary', (50, 50), cv2.FONT_HERSHEY_PLAIN, 2, 255)
    showFrame = cv2.resize(frame2,(int(width / scaling), int(height / scaling)))
    cv2.imshow('display', showFrame)
    cv2.setMouseCallback('display', centerCallback)  #boundary callback
    cv2.waitKey(0)
    leftX, leftY = arenaX, arenaY

    frame2 = copy.copy(frame)
    cv2.putText(frame2, 'Select RIGHT boundary', (50, 50), cv2.FONT_HERSHEY_PLAIN, 2, 255)
    showFrame = cv2.resize(frame2, (int(width / scaling), int(height / scaling)))
    cv2.imshow('display', showFrame)
    cv2.setMouseCallback('display', circleCallback, (leftX, leftY))  #boundary callback
    cv2.waitKey(0)
    rightX, rightY = arenaX, arenaY
    dX = (leftX - rightX) / 2
    dY = (leftY - rightY) / 2
    centerX = int(leftX - dX)
    centerY = int(leftY - dY)

    radius = np.sqrt((leftX - rightX) ** 2 + (leftY - rightY) ** 2) / 2

else:
    centerX, centerY = width / 4, height / 4 # order may be wrong...
    # rectCenterX, rectCenterY = width/2, height/2 # order may be wrong...
    leftX, leftY = 10 / 3, 10 / 3
    rightX, rightY = (width - 10) / 3, (height - 10) / 3
    radius = np.sqrt((leftX - rightX) ** 2 + (leftY - rightY) ** 2) / scaling


# select flies
frame2 = copy.copy(frame)
cv2.putText(frame2, 'Select ALL flies', (50, 50), cv2.FONT_HERSHEY_PLAIN, 2, 255)
showFrame = cv2.resize(frame2, (int(width / scaling), int(height / scaling)))
cv2.imshow('display', showFrame)
cv2.setMouseCallback('display', findFliesCallback)  #boundary callback
cv2.waitKey(0)
leftX, leftY = arenaX, arenaY

flyCenters = flyCenters[0:nFlies, :]

if doAlignMike:
    # align microphone 1
    frame2 = copy.copy(frame)
    cv2.putText(frame2, 'Align microphone 1', (50, 50), cv2.FONT_HERSHEY_PLAIN, 2, 255)
    cv2.createTrackbar('imageRotation', 'display', 180, 360, trackerCallback)
    showFrame = cv2.resize(frame2, (int(width / scaling), int(height / scaling)))
    cv2.setMouseCallback('display', doNothing)
    cv2.imshow('display', showFrame)
    cv2.waitKey(0)

# we want to show the estimated position of the microphones to make sure we have them all...
# maybe this should be shown in the 'align microphone' phase

if doBlinky:
    # choose blinky light area
    frame2 = copy.copy(frame)
    cv2.putText(frame2, 'Select blinky light', (50, 50), cv2.FONT_HERSHEY_PLAIN, 2, 255)
    showFrame = cv2.resize(frame2, (int(width / scaling), int(height / scaling)))
    cv2.imshow('display', showFrame)
    cv2.setMouseCallback('display', areaCallback)  # boundary callback
    cv2.waitKey(0)

vr.release()
cv2.destroyAllWindows()

# save the data
filename = os.path.splitext(movieName)[0] + '_annotated.txt'
with open(filename, 'w') as f:
    f.write(f"width, {width}\n")
    print('width = ' + str(width))
    f.write(f"height, {height}\n")
    print('height = ' + str(height))
    f.write(f'centerX, {centerX * scaling}\n')
    print('centerX = ' + str(centerX * scaling))
    f.write(f'centerY, {centerY * scaling}\n')
    print('centerY = ' + str(centerY * scaling))
    f.write(f'radius, {radius * scaling}\n')
    print('radius = ' + str(radius * scaling))
    f.write(f'rectCenterX, {rectCenterX * scaling}\n')
    print('rectCenterX = ' + str(rectCenterX * scaling))
    f.write(f'rectCenterY, {rectCenterY * scaling}\n')
    print('rectCenterY = ' + str(rectCenterY * scaling))
    f.write(f'rectRadius, {rectRadius * scaling}\n')
    print('rectRadius = ' + str(rectRadius * scaling))
    f.write(f'start_frame, {startFrame}\n')
    print('start_frame = ' + str(startFrame))
    f.write(f'angle, {angle}\n')
    print('angle = ' + str(angle))
    f.write(f'nFlies, {nFlies}\n')
    print('nFlies = ' + str(nFlies))

    for ii in range(nFlies):
        f.write(str(flyCenters[ii, 0] * scaling) + '\n')
        f.write(str(flyCenters[ii, 1] * scaling) + '\n')
        print('fly center = ' + str((flyCenters[ii, 0], flyCenters[ii, 1])))

d = convert(filename)
save(filename, d)
