import plotly.express as px
import pandas as pd


def draw_confidence_score(score: float):
    """
    Рисует donut-chart, где цветная часть кольца = точность модели.
    score ожидается в диапазоне [0, 1].
    """
    # Защита от выходов за диапазон
    score = max(0.0, min(float(score), 1.0))
    value = score * 100

    data = {
        "label": ["confidence", "rest"],
        "value": [value, 100 - value],
    }
    df = pd.DataFrame(data)

    fig = px.pie(
        df,
        names="label",
        values="value",
        hole=0.8,
    )

    # Окрашивается только "полукольцо" = точность, остальная
    # часть кольца - в цвет фона.
    fig.update_traces(
        sort=False,
        textinfo="none",
        hoverinfo="skip",
        marker=dict(
            colors=[
                "#60A5FA",  # активная (точность)
                "#1E1E1E",  # фон, "пустая" часть кольца
            ],
            line=dict(color="#1E1E1E", width=2),
        ),
    )

    # 
    fig.update_layout(
        showlegend=False,
        title=None,
        annotations=[
            dict(
                text=f"{value:.1f}%",
                x=0.5,
                y=0.5,
                font=dict(size=52, color="white"),
                showarrow=False,
                xanchor="center",
                yanchor="middle",
            )
        ],
        margin=dict(l=0, r=0, t=0, b=0),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
    )

    return fig