from ephaccess import *
import numpy as np
import math
from timezonefinder import TimezoneFinder
import datetime
from zoneinfo import ZoneInfo
import cv2

def sin(x):
    return math.sin(x)

def cos(x):
    return math.cos(x)

def RADtoD(x): # радианы в градусы (с дробной частью)
    return x * 180 / math.pi

def DtoRAD(x):
    return x * math.pi /180

def HHMMSStoRAD(xHH, xMM, xSS):
    x = xHH * 15
    x += xMM / 4
    x += xSS / 240
    return DtoRAD(x)

def DTtoJD(year, month, day, hour, min, sec): # гринв DT
    a = (14 - month) // 12
    y = year + 4800 - a
    m = month + 12 * a - 3
    jdn = day + (153 * m + 2) // 5 + 365 * y + y // 4 - y // 100 + y // 400 - 32045
    jd = jdn + (hour - 12) / 24 + min / 1440 + sec / 86400
    return jd

def CartesianToPolar(x,y,z): # ra - phi, p - theta
    ra = math.atan2(y, x)
    if ra < 0: ra += 2*math.pi
    p = math.atan2(z, math.sqrt(x*x + y*y))
    r = math.sqrt(x*x + y*y + z*z)
    return (ra,p,r)

def PolarToCartesian(r,a1,a2): # a1 - theta, a2 - phi
    return [r*sin(a1)*cos(a2), r*sin(a1)*sin(a2), r*cos(a1)]

def HHMMSStoHHhh(hour, min, sec):
    return hour + (sec / 60 + min) / 60

def HHhhtoHHMMSS(hour):
    min = (hour - int(hour)) * 60
    sec = (min - int(min)) * 60
    return (int(hour), int(min), sec)

def GST(year, month, day, hour, min, sec):
    a = 0.0657098
    b = 17.388895
    c = 1.002738
    d = 0.997270
    jd0 = DTtoJD(2024,1,1,0,0,0)
    jd = DTtoJD(year, month, day, hour, min, sec)
    dif = int(jd - jd0)
    dif *= a
    t0 = dif - b
    gmt = HHMMSStoHHhh(hour, min, sec)
    res = t0 + gmt * c
    if res > 24: res -= 24
    elif res < 0: res += 24
    return HHhhtoHHMMSS(res)

def LST(year, month, day, hour, min, sec, lngtd): # на вход GMT!
    gst = GST(year, month, day, hour, min, sec)
    gst = HHMMSStoHHhh(gst[0], gst[1], gst[2])
    lngtd = RADtoD(lngtd)
    lngtd = lngtd / 15 # в часы
    res = gst + lngtd
    if res > 24: res -= 24
    elif res < 0: res += 24
    return HHhhtoHHMMSS(res)

def timeDiff(lattitude, longitude, Y, M, D, h, m, s): # в радианах
    # pip install tzdata - бд с часовыми поясами
    lattitude = RADtoD(lattitude)
    longitude = RADtoD(longitude)
    tf = TimezoneFinder(in_memory=True)
    timezone = tf.timezone_at(lng=longitude, lat=lattitude)
    moment = datetime.datetime(Y, M, D, h, m, s, tzinfo=ZoneInfo(timezone))
    diff = datetime.datetime.utcoffset(moment)
    return diff.total_seconds() / 3600

def GMT(lattitude, longitude, Y, M, D, h, m, s):
    offset = timeDiff(lattitude, longitude, Y, M, D, h, m, s)
    dt = datetime.datetime(Y, M, D, h, m, s)
    return dt - datetime.timedelta(hours=offset)

def HorizontalSunCoords(longitude, lattitude, Y, M, D, h, m, s): #lngtd,lttd в рад, Az from Notrh to East, Az,El in rad
    tm = GMT(longitude, lattitude, Y, M, D, h, m, s)

    # Получаем координаты Солнца в экваториальной (на самом деле ICRF) СК в заданный момент времени
    eph = EphAccess()
    eph.load_file("epm2021.bsp")
    eph.set_distance_units(EPH_KM)
    eph.set_time_units(EPH_SEC)
    jd = DTtoJD(tm.year, tm.month, tm.day, tm.hour, tm.minute, tm.second)
    coords = eph.calculate_rectangular(eph.object_by_name("sun"), eph.object_by_name("earth"), jd, 0.0)[0]

    # Переводим их в полярные (для получения прямого восхождения ra)
    x = coords[0]
    y = coords[1]
    z = coords[2]
    (ra,delta,r) = CartesianToPolar(x,y,z)

    # Вычисляем звездное время для перевода в горизонтальную СК
    st = LST(tm.year, tm.month, tm.day, tm.hour, tm.minute, tm.second, longitude)
    st = HHMMSStoRAD(st[0], st[1], st[2])
    h = st - ra

    El = math.asin(sin(delta) * sin(lattitude) + cos(delta) * cos(lattitude) * cos(h))
    Az = math.acos((sin(delta) - sin(lattitude) * sin(El)) / (cos(lattitude) * cos(El)))
    if sin(h) > 0: Az = math.pi * 2 - Az
    horC = PolarToCartesian(r, DtoRAD(90) - El, Az)

    # Топоцентрируем
    horC[2] -= 6362.760 

    # Переводим в полярные для получения искомых азимута и высоты
    horCPolar = CartesianToPolar(horC[0], horC[1], horC[2])

    Az = horCPolar[0]
    El = horCPolar[1]

    return (Az, El)

def get_kernel_size(img, thrs):
    [x,y] = img.shape
    max = 0
    for i in range(0, x):
        count = 0
        for j in range(0, y):
            if img[i][j] >= thrs:
                count += 1
            else: 
                if max < count: 
                    max = count
                count = 0

    for j in range(0, y):
        count = 0
        for i in range(0, x):
            if img[i][j] >= thrs:
                count += 1
            else: 
                if max < count: 
                    max = count
                count = 0

    return 1 * max // 5

def SunPixel(img, thrs): # координаты пискеля цента Солнца (начало координат - левый нижний угол)
    # в серое
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # в бинарное
    img_bin = cv2.threshold(img_gray, thrs, 255, cv2.THRESH_BINARY)[1]

    # размер структурного элемента
    kernel_size = get_kernel_size(img_bin, thrs)
    if kernel_size <= 0:
        return None
    kernel = np.ones((kernel_size, kernel_size), np.uint8) 
    
    # эрозия
    img_eros = cv2.erode(img_bin, kernel, iterations=1)

    # dilation
    img_dilat = cv2.dilate(img_eros, kernel, iterations=1)

    moments = cv2.moments(img_dilat, 0)
    dArea = moments['m00']
    mx = moments['m10']
    my = moments['m01']
    if dArea > 100:
        x = int(mx / dArea)
        y = int(my / dArea)
    else: return None

    width = img.shape[1]
    height = img.shape[0]
    
    return x, height - y