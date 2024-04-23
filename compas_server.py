from flask import Flask, request
from flask import jsonify
import numpy as np
import cv2
import compas_server_funcs


# Initialize the Flask application
app = Flask(__name__)


@app.route("/", methods=["POST"])
def f():

    data = request.form.to_dict(flat=True)
    lngtd = float(data['lngtd'])
    lttd = float(data['lttd'])
    img=None
    for item in request.files.getlist('photo'):
        data = item.read()
        nparr = np.fromstring(data, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    now = compas_server_funcs.datetime.datetime.now()

    sun_coords = compas_server_funcs.HorizontalSunCoords(
         compas_server_funcs.DtoRAD(lngtd), compas_server_funcs.DtoRAD(lttd), now.year, now.month, now.day, now.hour, now.minute, now.second)
    pix_coords = compas_server_funcs.SunPixel(img, 220)

    x_pix = y_pix = -1

    if pix_coords!=None:
         x_pix = pix_coords[0]
         y_pix = pix_coords[1]

    return jsonify(
        Az = sun_coords[0],
        El = sun_coords[1],
        x = x_pix,
        y = y_pix
        )
    

if(__name__ == "__main__"):
	app.run(debug=True, port=8080, host='192.168.0.103')
