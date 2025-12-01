import plotly.express as px
import pandas as pd

def draw_confidence_score(score : float):
    """
    Рисует donut-chart для быстрой визуальной интерпретации
    """
    data = {
        'label' : ['confidence', ''],
        'value' : [score * 100, 100 -score * 100]
    }
    df = pd.DataFrame(data)

    fig = px.pie(
        df,
        names="label",
        values="value",
        title="Sentiment Distribution",
        hole=0.8  
    )

    return fig