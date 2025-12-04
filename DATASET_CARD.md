# Источник
Kaggle-dataset:
https://www.kaggle.com/datasets/chaudharyanshul/airline-reviews/data

# Структура данных
## Файлы:
Файл BA_AirlineReviews.csv - все отзывы

## Поля:
Вот перевод описаний колонок:
- OverallRating — общая оценка, выставленная клиентом.
- ReviewHeader — заголовок отзыва.
- Name — имя автора отзыва.
- Datetime — дата и время публикации отзыва.
- VerifiedReview — признак того, что отзыв является проверенным.
- ReviewBody — полный текст отзыва.
- TypeOfTraveller — тип путешественника (например, деловой, турист).
- SeatType — класс перелёта (например, бизнес, эконом).
- Route — маршрут перелёта.
- DateFlown — дата совершённого рейса.
- SeatComfort — оценка удобства сидений.
- CabinStaffService — оценка работы бортпроводников.
- GroundService — оценка наземного обслуживания.
- ValueForMoney — оценка соответствия «цена–качество».
- Recommended — рекомендует ли клиент авиакомпанию British Airways.
- Aircraft — тип самолёта.
- Food&Beverages — оценка питания и напитков на борту.
- InflightEntertainment — оценка развлекательной системы на борту.
- Wifi&Connectivity — оценка качества Wi-Fi и бортовой связи.

При обучении модели (в различных версиях препроцессинга) использовались только поля OverallRating и ReviewBody.