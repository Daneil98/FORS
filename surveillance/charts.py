import matplotlib.pyplot as plt
import io, base64
import pandas as pd
import plotly.express as px
from .models import Logs
import openai

"""
def insights():
    df = pd.DataFrame(list(Logs.objects.values("weapon", "date")))

    summary = df.groupby("weapon").size().to_dict()

    prompt = f'
    Here is data on detected weapons: {summary}.
    Provide a clear, 3-sentence analysis highlighting trends or risks.'

    response = openai.ChatCompletion.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}]
    )

    insights = response["choices"][0]["message"]["content"]
    return insights
"""

def plotly_Chart():
    data = Logs.objects.values_list('weapon')
    fig = px.histogram(x=list(data), title="Weapon Type Distribution")
    fig.update_layout(showlegend=False)
    chart_html = fig.to_html(full_html=False)
    
    return chart_html
    
def weapon_stats():
    weapon_count = Logs.objects.values('weapon').order_by('weapon')
    
    weapon_dict = {}
    
    for w in weapon_count:
        weapon_dict[w['weapon']] = weapon_dict.get(w['weapon'], 0) + 1 
        
    # Bigger figure (8x8 inches instead of default 6x4)
    fig, ax = plt.subplots(figsize=(8, 8))

    # Draw pie chart
    wedges, texts, autotexts = ax.pie(
        weapon_dict.values(),
        labels=weapon_dict.keys(),
        autopct='%1.1f%%',
        startangle=90,
        textprops={'fontsize': 12}   # make labels bigger
    )
    ax.axis('equal')  # keeps it circular

    # Adjust label position so they don’t overlap
    plt.tight_layout()

    # Save to buffer
    buf = io.BytesIO()
    plt.savefig(buf, format='png', bbox_inches="tight")  # bbox makes sure labels are not cut off
    buf.seek(0)
    image_base64 = base64.b64encode(buf.getvalue()).decode('utf-8')
    buf.close()
    
    return image_base64

def weapon_histogram():
    qs = Logs.objects.values_list('date', flat=True)
    dates = [d.date() for d in qs]
    
    # Histogram
    # Bigger and taller figure
    fig, ax = plt.subplots(figsize=(6, 6))  # wider=10, taller=6

    # Plot histogram
    ax.hist(dates, bins=len(set(dates)), color='skyblue', edgecolor='black')

    # Formatting
    ax.set_title("Weapon Detections Over Time")
    ax.set_xlabel("Date")
    ax.set_ylabel("Detections")

    # Rotate x-axis labels so they’re readable
    plt.xticks(rotation=45, ha='right')

    # Auto-adjust layout so labels don’t get cut
    plt.tight_layout()

    # Save to buffer
    buf = io.BytesIO()
    plt.savefig(buf, format="png", bbox_inches="tight")
    buf.seek(0)
    image_base64 = base64.b64encode(buf.getvalue()).decode("utf-8")
    buf.close()
    return image_base64
