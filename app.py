from flask import Flask, request, render_template
from flask_sqlalchemy import SQLAlchemy
import joblib
import numpy as np
import lime
from lime.lime_tabular import LimeTabularExplainer
import os 


template_dir = os.path.join(os.getcwd(), 'templates')

app = Flask(__name__, template_folder=template_dir)


app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///heart_disease.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False


db = SQLAlchemy(app)


class UserPrediction(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(100), nullable=False)
    location = db.Column(db.String(100), nullable=False)
    prediction = db.Column(db.String(50), nullable=False)
    recommendations = db.Column(db.String(500), nullable=False)


with app.app_context():
    db.create_all()


model = joblib.load('knn.pkl')


feature_names = ['age', 'sex', 'cp', 'fbs', 'restecg']
explainer = LimeTabularExplainer(
    training_data=np.random.rand(100, 5),  
    feature_names=feature_names,
    class_names=['No Heart Disease', 'Heart Disease'],
    mode='classification'
)

@app.route('/')
def home():
    return render_template("index.html")
@app.route('/predict', methods=['POST'])
def predict():
    try:
        name = request.form.get('name')
        location = request.form.get('location')

        input_data = [float(request.form.get(field)) for field in feature_names]
        features = np.array(input_data).reshape(1, -1)  


        prediction = model.predict(features)
        probability = model.predict_proba(features)[0]


        data_row = np.array(input_data)  


        explanation = explainer.explain_instance(
            data_row=data_row,   
            predict_fn=model.predict_proba
        )
        explanations = explanation.as_list()

        if prediction[0] == 1:  # Heart disease detected
            result = "Heart Disease Present"
            recommendations = (
                "We recommend focusing on heart-healthy habits. "
                "Consider regular exercise like walking or cycling, a balanced diet, and managing stress."
            )
            style = "color: red;"
        else:  # No heart disease detected
            result = "No Heart Disease"
            recommendations = (
                "Congratulations! Keep up the great work with a balanced diet and regular exercise."
            )
            style = "color: green;"


        user_prediction = UserPrediction(
            name=name,
            location=location,
            prediction=result,
            recommendations=recommendations
        )
        db.session.add(user_prediction)
        db.session.commit()

        # Pass data to template
        return render_template(
            "index.html",
            prediction_text=f'Result: {result}', 
            recommendation_text=recommendations,
            text_style=style,
            explanations=explanations
        )

    except Exception as e:

        print(f"Error: {e}")
        return render_template(
            "index.html",
            prediction_text="Error occurred while processing the prediction.",
            recommendation_text="Please check the inputs and try again.",
            text_style="color: red;"
        )


if __name__ == "__main__":
    app.run(debug=True)
