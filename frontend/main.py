from flask import Flask, render_template, request
import resultprocess
from io import BytesIO
import base64

app = Flask(__name__)


@app.route('/')
def index():
    return render_template("template.html")


@app.route('/upload', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        f = request.files['file']
        yolo = resultprocess.YOLOProcess(name_classes=[i for i in range(26)],
                                         colors=['green' for i in range(26)], font_file_name="static/fonts/font.otf")
        yolo.process(f)
        yolo.applay_mask(0.1)
        yolo.applay_max_suppression_fast_filter(0.1, 0.20)
        img_io = BytesIO()
        yolo.image_pred.save(img_io, 'JPEG', quality=70)
        img_io.seek(0)
        del yolo
        return render_template("template_answer.html",
                               processed_image=base64.b64encode(img_io.getvalue()).decode("utf-8"))
    return 'file upload fail'


if __name__ == "__main__":
    app.run(host='0.0.0.0',debug=False)
else:
    application = app