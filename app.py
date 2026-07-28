from flask import Flask, render_template, request
from ultralytics import YOLO
from flask_sqlalchemy import SQLAlchemy
from datetime import datetime
import os
import cv2

app = Flask(__name__)

UPLOAD_FOLDER = "static/uploads"
OUTPUT_FOLDER = "static/outputs"

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# Database config
app.config["SQLALCHEMY_DATABASE_URI"] = "sqlite:///detections.db"
app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False

db = SQLAlchemy(app)

# Database Table
class Detection(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    image_name = db.Column(db.String(200)) 
    input_image = db.Column(db.String(200))
    output_image = db.Column(db.String(200))
    tree_count = db.Column(db.Integer)
    date = db.Column(db.DateTime, default=datetime.utcnow)

model = YOLO("runs/detect/train/weights/best.pt")


@app.route("/", methods=["GET","POST"])
def index():

    input_image = None
    output_image = None
    count = None

    if request.method == "POST":

        file = request.files["image"]

        if file:

            filepath = os.path.join(UPLOAD_FOLDER, file.filename)
            file.save(filepath)

            results = model(filepath)

            r = results[0]
            count = len(r.boxes)

            plotted = r.plot()

            output_path = os.path.join(OUTPUT_FOLDER, file.filename)
            cv2.imwrite(output_path, plotted)

            input_image = "uploads/" + file.filename
            output_image = "outputs/" + file.filename

            # Save to database
            new_data = Detection(
                image_name=file.filename,
                input_image=input_image,
                output_image=output_image,
                tree_count=count
            )

            db.session.add(new_data)
            db.session.commit()

    return render_template(
        "index.html",
        input_image=input_image,
        output_image=output_image,
        count=count
    )


@app.route("/history")
def history():

    records = Detection.query.order_by(Detection.date.desc()).all()

    return render_template("history.html", records=records)


if __name__ == "__main__":
    with app.app_context():
        db.create_all()

    app.run(debug=True)