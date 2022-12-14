## Экспериметы с SVM.

- ***Порядок просмотра результатов:***

1. [Стандартный SVC](https://github.com/kalashnikova04/HSE_mlhls_project_SA/blob/master/checkpoint4/SVM/SVM_1.ipynb)
2. [Понижение размерности](https://github.com/kalashnikova04/HSE_mlhls_project_SA/blob/master/checkpoint4/SVM/SVM_2.ipynb)
3. [Word2Vec](https://github.com/kalashnikova04/HSE_mlhls_project_SA/blob/master/checkpoint4/SVM/SVM_3.ipynb)

- ***Что было сделано?***<br>
    - *Константный прогноз* для сравнения результатов обученных моделей.
    - Обучен *SVC* по частям на *всей выборке* (уни-, биграммы).
    - *SVC + GridSearchCV* на уни - тетраграммах.
    - *SVC / LinearSVC + TruncatedSVD*.
    - Обучена модель *Word2Vec* на всей выборке.
    - Векторизация текста методом *усреднения векторов* слов / *взвешиванием TF-IDF*.


- ***Результаты***

    | Метрика \ DummyClassifier | train | test |
    | ------------- |:------------------:| :-----:|
    | `f1-score 'micro'` / `precision` / `recall`| `0.5079387296094643` | `0.5083430052906863` |

    | Метрика \ SVC | train | test |
    | ------------- |:------------------:| :-----:|
    | `f1-score 'micro'`| `0.604514853020184` | `0.5757176089916545` |

    | Метрика \ SVC + GridSearchCV + TruncatedSVD | train | test |
    | ------------- |:------------------:| :-----:|
    | `f1-score / precision / recall 'micro'`| `0.7508000000000001` | `0.552093023255814` |
    | `precision 'macro'`| `0.6489656370688169` | `0.3551073519843852` |
    | `recall 'macro'`| `0.3770802595053126` | `0.35871300878205253` |
    | `ROC-AUC score`| `0.6734674761381081` | `0.5864924777693895` |

    | Метрика \ LinearSVC + TruncatedSVD | train | test |
    | ------------- |:------------------:| :-----:|
    | `f1-score 'micro'`| `0.7431` | `0.546046511627907` |

    | Метрика \ MeanEmbeddingVectorizer + SVC | train | test |
    | ------------- |:------------------:| :-----:|
    | `f1-score 'micro' / precision / recall`| `0.4165` | `0.42` |

    | Метрика \ TfidfEmbeddingVectorizer + LinearSVC | train | test |
    | ------------- |:------------------:| :-----:|
    | `f1-score 'micro' / precision / recall`| `0.50043` | `0.5` |

    Эксперименты показали, что лучший результат (по метрике `f1-micro`) показала обычная модель SVC с уни и биграммами. Конечно, на это повлиял и тот факт, что остальные эксперименты проводились на подвыборках сильно меньшего размера. Обучение на больших выборках могло длиться несколько часов и затем прерываться, тк ядро прекращало работу.


    ![Image alt](./images/colab_break.png)



    ![Image alt](./images/bagg_class.png)

- **Вывод:** 
    - *По времени*: использовать SVM на выборках большого размера неэффективно;
    - *По качеству*: вывод такой же, как и в разделе с EDA - разметка оставляет желать лучшего (:
