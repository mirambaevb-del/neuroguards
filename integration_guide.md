# NeuroGuard: подключение к сайту Lovable

Исходные адреса, указанные владельцем:

- https://neuro-guard-monitor.lovable.app
- https://github.com/mirambaevb-del/Neuroguard
- https://github.com/mirambaevb-del/neuro-guard-monitor

На этапе сборки этого архива доступ к приватному редактированию сайта/репозиториев **не был подтверждён**. Сайт **пока не подключён** к новой модели.

## Архитектура

`Lovable frontend → ваш защищённый backend → NeuroGuardDetector → JSON → frontend`

Нужен постоянно доступный HTTPS-сервер для `api.py`. Google Colab подходит для исследования, но не для production-hosting.

1. Запустите обучение на настоящих CHB-MIT EDF и дождитесь создания `model/seizure_detector.joblib`, `model/config.json`, `model/features.json`, `model/manifest.json`.
2. Разверните **весь** пакет Python на управляемом вами сервере (Docker, облачный Python runtime или своя машина), установите зависимости и укажите `NEUROGUARD_MODEL_DIR`. Для публичного сервера установите `NEUROGUARD_PUBLIC=1` и обязательно `NEUROGUARD_API_KEY`.
3. Проверьте `GET /health` и `GET /model-info`, последний должен отвечать `status: ready`.
4. Ограничьте разрешённые origin через `NEUROGUARD_ALLOWED_ORIGINS=https://neuro-guard-monitor.lovable.app`.
5. Настройте авторизацию **на сервере**. Не держите API-ключ прямо в коде браузера; для клиентского фронтенда используйте посредник на вашем сервере с авторизацией пользователей и ограничением запросов.
6. Реализуйте на сайте форму загрузки EDF или отдельную форму 8-канального EEG. На backend `POST /predict-edf` передавайте сырые байты EDF, заголовок `Content-Type: application/octet-stream`; `POST /predict` требует JSON 8 × (1024 × число окон) значений в вольтах.
7. Показывайте `INTERICTAL` или `ICTAL` и время каждого окна. `seizure_score_uncalibrated` нельзя подписывать как «вероятность приступа»; отдельно объясните пользователю, что это исследовательская оценка модели.
8. До валидации скрывайте элементы UI вида «приступ через 10 минут» и все автоматические медицинские тревоги.

## Пример frontend-интеграции (TypeScript)

```typescript
// ВЫЗЫВАТЬ НА ВАШЕМ ДОВЕРЕННОМ СЕРВЕРЕ ИЛИ ЧЕРЕЗ ПРОВЕРЕННЫЙ ПРОКСИ,
// не размещать приватный токен в браузерном исходном коде.
async function analyzeEdf(bytes: Uint8Array, backendUrl: string, serverSideKey: string) {
  const response = await fetch(`${backendUrl}/predict-edf`, {
    method: "POST",
    headers: { "Content-Type": "application/octet-stream", "X-API-Key": serverSideKey },
    body: bytes,
  });
  if (!response.ok) throw new Error(`NeuroGuard API: ${response.status} ${await response.text()}`);
  return response.json();
}
```

## Отличие Prediction

Будущая функция **предсказания** будет отдельной: `task=SEIZURE_PREDICTION_RESEARCH` с новой моделью, горизонтом предупреждения, оценкой ложных тревог/сутки и проверкой независимых пациентов. Текущие `/predict` и `/predict-edf` не должны меняться с detection на prediction незаметно.

## Что нужно владельцу

- Доступ к исходному проекту Lovable/GitHub, если требуется править реальный frontend.
- Доверенный сервер и HTTPS-домен для backend.
- Настоящие EDF и официальные patient summary для обучения.
- Обсуждение безопасности и конфиденциальности перед использованием любых реальных EEG пациентов.
