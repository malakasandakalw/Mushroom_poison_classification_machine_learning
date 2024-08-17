from flask import Flask, request, render_template
import pandas as pd
import pickle
import numpy as np
from sklearn.preprocessing import OrdinalEncoder

app = Flask(__name__)


# unique_values.csv was created to get all unique values in each column to populate "<select>" in html file
# df = pd.read_csv('./train.csv')
# df = df.drop(['id', 'class', 'stem-root', 'veil-type', 'veil-color', 'spore-print-color'], axis=1)
# unique_values = {}
# for column in df.columns:
#     unique_values[column] = df[column].dropna().unique()

# unique_df = pd.DataFrame(dict([(k, pd.Series(v)) for k, v in unique_values.items()]))
# unique_df.to_csv('unique_values.csv', index=False)


df = pd.read_csv('./unique_values.csv')

cap_shape_values = df['cap-shape'].unique()
cap_surface_values = df['cap-surface'].unique()
cap_color_values = df['cap-color'].unique()
does_bruise_or_bleed_values = df['does-bruise-or-bleed'].unique()
gill_attachment_values = df['gill-attachment'].unique()
gill_spacing_values = df['gill-spacing'].unique()
gill_color_values = df['gill-color'].unique()
stem_surface_values = df['stem-surface'].unique()
stem_color_values = df['stem-color'].unique()
has_ring_values = df['has-ring'].unique()
ring_type_values = df['ring-type'].unique()
habitat_values = df['habitat'].unique()
season_values = df['season'].unique()

@app.route('/')
def home():
   return render_template('index.html', 
                          cap_shape_values=cap_shape_values,
                          cap_surface_values=cap_surface_values,
                          cap_color_values=cap_color_values,
                          does_bruise_or_bleed_values=does_bruise_or_bleed_values,
                          gill_attachment_values=gill_attachment_values,
                          gill_spacing_values=gill_spacing_values,
                          gill_color_values=gill_color_values,
                          stem_surface_values=stem_surface_values,
                          stem_color_values=stem_color_values,
                          has_ring_values=has_ring_values,
                          ring_type_values=ring_type_values,
                          habitat_values=habitat_values,
                          season_values=season_values
                         )


# predictions
xgb_model = pickle.load(open("./xgb_model.pkl", "rb"))

form_params = (
    "cap-diameter", "cap-shape", "cap-surface", "cap-color", "does-bruise-or-bleed", "gill-attachment", "gill-spacing", "gill-color", "stem-height",
    "stem-width", "stem-surface", "stem-color", "has-ring", "ring-type", "habitat", "season"
)

cat_features = [ "cap-shape", "cap-surface", "cap-color", "does-bruise-or-bleed", "gill-attachment", "gill-spacing", "gill-color", "stem-surface", "stem-color", "has-ring", "ring-type", "habitat", "season"]

@app.post('/predict')
def prediction():
    form=request.form
    print(f"{form}")
    data={}
    for param in form_params:
       param_value=form.get(param)
       print(f"{param} : {param_value}")
       data[param]=float(param_value) if 'diameter' in param or 'height' in param or 'width' in param else param_value
    data_frame=pd.DataFrame(data, index=[0])
    ordinal_encoder = OrdinalEncoder()
    encoded_ordinal_features = ordinal_encoder.fit_transform(data_frame[cat_features])
    data_frame[cat_features] = encoded_ordinal_features
    final_features = data_frame.values
    prediction = xgb_model.predict(final_features)[0]
    result = 'Edible' if prediction == 0 else 'Poisonous'
    print(f"*******Result: {result}")
    return render_template("result.html", predicted=result)

if __name__ == '__main__':
   app.debug = True
   app.run()
   app.run(debug = True)