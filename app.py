#================= IMPORT LIBRARIES =================
from flask import Flask, render_template, request, jsonify
import joblib
import pandas as pd
import numpy as np
import plotly.express as px

app = Flask(__name__)

# ================= LOAD =================
model = joblib.load("aqi_model.pkl")
le_state = joblib.load("state_encoder.pkl")
le_area = joblib.load("area_encoder.pkl")

df = pd.read_csv("cleaned_aqi.csv")


# ================= LANDING =================
@app.route("/")
def landing():
    return render_template("home.html")


# ================= CATEGORY =================
def get_category(aqi):
    if aqi <= 50:
        return "Good"
    elif aqi <= 100:
        return "Satisfactory"
    elif aqi <= 200:
        return "Moderate"
    elif aqi <= 300:
        return "Poor"
    elif aqi <= 400:
        return "Very Poor"
    else:
        return "Severe"


# ================= COLOR =================
def get_color(category):
    colors = {
        "Good": "#00e400",
        "Satisfactory": "#0099cc",
        "Moderate": "#ffde33",
        "Poor": "#ff9933",
        "Very Poor": "#cc0033",
        "Severe": "#660099",
    }
    return colors.get(category, "#ffffff")


# ================= PREDICT PAGE =================
@app.route("/predict_page")
def home():
    states = sorted(df["state"].unique())
    return render_template("index.html", states=states)


# ================= GET AREAS =================
@app.route("/get_areas/<state>")
def get_areas(state):
    areas = df[df["state"] == state]["area"].unique().tolist()
    return jsonify(areas)


# ================= PREDICT =================
@app.route("/predict", methods=["POST"])
def predict():
    state = request.form["state"]
    area = request.form["area"]
    year = int(request.form["year"])
    month = int(request.form["month"])

    state_encoded = le_state.transform([state])[0]
    area_encoded = le_area.transform([area])[0]

    features = np.array([[state_encoded, area_encoded, year, month]])

    prediction = model.predict(features)[0]
    category = get_category(prediction)
    color = get_color(category)

    return render_template(
        "result.html", aqi=round(prediction, 2), category=category, color=color
    )


# ================= VISUALIZE =================
@app.route("/visualize", methods=["GET", "POST"])
def visualize():
    if request.method == "POST":
        year = int(request.form["year"])

        #  VALIDATION
        if year < 2015 or year > 2025:
            return render_template(
                "visualize.html", error="Please enter a year between 2015 and 2025"
            )

        data = df[df["year"] == year]

        if data.empty:
            return render_template(
                "visualize.html", error="No data available for selected year"
            )

        # ================= TOP AREAS =================
        top_areas = (
            data.groupby("area")["aqi_value"]
            .mean()
            .sort_values(ascending=False)
            .head(10)
        )

        # ================= TOP STATES =================
        top_states = (
            data.groupby("state")["aqi_value"]
            .mean()
            .sort_values(ascending=False)
            .head(10)
        )

        # ================= MONTH TREND =================
        month_trend = data.groupby("month")["aqi_value"].mean()

        #  CATEGORY DISTRIBUTION
        categories = data["air_quality_status"].fillna("Unknown").value_counts()

        #  HEATMAP
        heatmap_df = data.pivot_table(
            values="aqi_value", index="state", columns="month", aggfunc="mean"
        ).fillna(0)

        heatmap_fig = px.imshow(
            heatmap_df,
            aspect="auto",
            color_continuous_scale="Reds",
            labels=dict(x="Month", y="State", color="AQI"),
        )

        #  REMOVE HOVER
        heatmap_fig.update_traces(hoverinfo="skip", hovertemplate=None)

        heatmap_fig.update_layout(
            margin=dict(l=40, r=40, t=40, b=40),
            height=500,
            hovermode=False,
            paper_bgcolor="#0f172a",
            plot_bgcolor="#0f172a",
        )

        heatmap_html = heatmap_fig.to_html(
            full_html=False, config={"displayModeBar": False, "staticPlot": True}
        )

        return render_template(
            "visualize.html",
            year=year,
            top_areas=top_areas.to_dict(),
            top_states=top_states.to_dict(),
            categories=categories.to_dict(),
            months=month_trend.to_dict(),
            heatmap=heatmap_html,
        )
    return render_template(
        "visualize.html",
        error="Please Enter a Year Between 2015 and 2025",
        top_areas={},
        top_states={},
        categories={},
        months={},
        worst_month="",
        worst_value="",
        heatmap="",
    )


# ================= RUN =================
if __name__ == "__main__":
    app.run(debug=True)
