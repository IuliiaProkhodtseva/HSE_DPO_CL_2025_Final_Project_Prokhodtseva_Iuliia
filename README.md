# HSE_DPO_CL_2025_Final_Project_Prokhodtseva_Iuliia
Финальный проект курса НИУ ВШЭ "Компьютерная лингвистика" 

Тема: Дообучение моделей BERT (google-bert/bert-base-uncased) и Llama-3.2 для классификации научных статей

Цель: дообучить модели машинного обучения для автоматическиой классификаций научных статей, сравнив два подхода: дообучение Llama-3.2 через Unsloth и BERT через PEFT с созданием кастомной функции потерь. 

Данные: спарсенный набор статей с https://arxiv.org по теме Computer Science


Этапы: 
1) Предобработка данных (очистка и нормализация текстов, фильтрация редких классов, токенизация текстов и формирование тренировочной и тестовой выборок)
   
3) Дообучение
   
для BERT:
- Использование LoRA-адаптеров для эффективной настройки весов
- Применение Contrastive Loss совместно с CrossEntropy Loss
- Learning Rate: 2e-4
- batch size=64
- contrastive_alpha=0.05
- temperature=0.07
- epochs=15 (Early Stopping, остановлено на 12 эпохе)

Для Llama-3.2:
- Также использование LoRA-адаптеров
- Дообучение в режиме 4-битной квантизации для экономии памяти GPU
- Fine-tuning на генерацию короткого ответа с названием категории
- Learning Rate: 2e-5
- batch size = 1 с gradient_accumulation_steps = 8
- epochs=56

Оценка результатов: 

Метрики: Accuracy, Precision, Recall, F1-score (Macro и Weighted)

Визуализация: Confusion Matrix

BERT: Accuracy ~0.78 

Используемые технологии:

Модели: google-bert/bert-base-uncased и Llama-3.2-3B

Библиотеки: 
- transformers: Для моделей BERT, токенизаторов, Trainer, TrainingArguments, DataCollatorWithPadding, EarlyStoppingCallback
- unsloth: Для быстрой квантизации и дообучения Llama-3 (LoRA, 4-bit)
- peft: Для реализации LoRA-адаптации BERT
- datasets: Для работы с датасетами (Dataset, DatasetDict, .map(), .filter(), .shuffle(), .select())
- pandas: Для первоначальной обработки и фильтрации CSV-данных
- numpy: Для числовых операций, уникальных значений, конкатенации массивов
- matplotlib.pyplot & seaborn: Для визуализации результатов (Confusion Matrix)
- sklearn.metrics: Для расчета метрик (classification_report, confusion_matrix)
- torch: Для тензорных операций и работы с GPU/CPU
- google.colab: Для монтирования Google Drive
- re (Regular Expressions): Для очистки текста
