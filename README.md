# stockMarketAnalysis
StockMarketAnalysis - простая программа для получения аналитики по всем активам пользователя тинькофф брокера через телеграм бота утром на телефоне. с кофе. Оч круто! Всем советую!


## Функционал программы

1.  скачивает из tinkoff брокера список активов, находящихся на счете
2. скачивает историю цен покаждому из активов
3. с помощью prophet предсказывает цену активов (акций, облигаций) на следующий день.

## файлы в папке test:

*  ```analyse.py``` - здесь прописана основная логика мучений с tinkoff API
*  ```__init__.py``` -  здесь происходит подгрузка всех основный файлов
*  ```pystanmodel.py``` = здесь происходит процесс прогнозирования цен активов
*  ```settings.py``` - здесь происходит подгрузка токена тинькофф API из файла ```.env``` , который вы должны создать локально
*  
P.S.  учтите, без токена tinkoff api  программа работать не будет 



![image](https://github.com/dobrotvorn/stockMarketAnalysis/assets/124947816/fb962d9d-a9a7-4b83-8050-bc425d9515a3)
