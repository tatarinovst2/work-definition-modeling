# Автоматическое выявление семантических изменений в русском языке с помощью методов глубинного обучения

## Содержание

Введение

2. Теоретические аспекты автоматического выявления семантических изменений
3. Имплементация автоматического выявления семантических изменений
   1. Обучение языковой модели на данных тезауруса
   2. Создание алгоритма автоматического определения семантических сдвигов на основе их
векторного представления
   3. Написание компьютерной программу для возможности публичного доступа к модели
4. Комплексный лингвистический анализ результатов работы компьютерной программы

Заключение

Список использованной литературы

Приложение


## Введение

Целью настоящей работы является создание компьютерной программы, позволяющей осуществлять
автоматическое выявление семантических изменений в русском языке с помощью методов
глубинного обучения.

Из поставленной цели были сформулированы следующие задачи:
1. Собрать тезаурус русского языка в качестве материала для обучения модели, а также диахронический
корпус текстов на основе НКРЯ.
2. Обучить языковую модель на данных тезауруса для того, чтобы генерировать определения.
3. Создать алгоритм автоматического определения семантических сдвигов на основе их векторного
представления. 
4. Написать компьютерную программу для возможности публичного доступа к модели.
5. Провести комплексный лингвистический анализ результатов работы компьютерной программы.

Для решения поставленных задач будут использованы следующие методы:
1. Метод анализа и синтеза для создания теоретической базы для данного исследования на основе
литературы.
2. Компьютерный метод для написания алгоритмов программы и обучения модели.
3. Методы обработки естественного языка для предобработки текстов.
4. Методы машинного обучения для алгоритма автоматического определения семантических сдвигов на
основе их векторного представления.
5. Метод комплексного лингвистического анализа результатов работы алгоритма.

Актуальность настоящей работы состоит в том, что, во-первых, вопрос анализа семантических
изменений в русском языке на основе автоматически сгенерированных определений недостаточно
изучен. Так, в настоящее время представление слов с помощью сгенерированных определений
является перспективной темой для поиска семантических изменений, с еще небольшим
количеством статей на данную тему на английском языке и отсутствием таких для русского.
Во-вторых, традиционные методы поиска семантических изменений недостаточно информативны
для основных потенциальных пользователей, таких как лексикографы или историки языка
(Giulianelli et al., 2023). Им хотелось бы получать описания старых и новых значений слов в
пригодной для чтения форме, возможно, даже с дополнительными пояснениями.

Новизна настоящей работы состоит в создании компьютерной программы, позволяющей автоматически
определять семантические изменения, с использованием автоматически сгенерированных определений,
а также применением этого метода на русском языке.

Практическая значимость данной работы заключается в том, что результаты работы программы можно
применять для определения степени семантического сдвига лексем, с наличием визуализаций и
определений для каждого выявленного значения, что может быть использовано в лексикологии,
где необходимы актуальные данные построения новых словарей.

В качестве материала исследования используется диахронический корпус НКРЯ, охватывающий три
периода (1700—1916, 1918—1991 и 1992—2016 годы) и имеющий в совокупности 250 миллионов
словоупотреблений. Данный корпус выбран, поскольку золотой датасет слов с изменившимся
и неизменившимся значением, использующийся для оценки модели, основан на данном корпусе.
Корпус был получен по запросу к авторам НКРЯ.

## Имплементация автоматического выявления семантических изменений

### Обучение языковой модели на данных тезауруса

В качестве модели была выбрана FRED-T5-1.7B, являющаяся одной из новейших языковых моделей,
выпущенных SberDevices и обученных с нуля на материале русского языка.
Для выбора модели мы использовали бенчмарк для оценки продвинутого понимания русского языка
«RussianSuperGLUE». В бенчмарке присутствуют шесть групп задач, охватывая общую диагностику
языковых моделей и различные лингвистические задачи: понимание здравого смысла, логическое
следование в естественном языке, рассуждения, машинное чтение и знания о мире.
FRED-T5-1.7B занимает самое высокое место в лидерборде данного бенчмарка, со значением 0.762,
уступая лишь результатам выполнения данных заданий людьми со значением 0.811,
что свидетельствует о ее способности к выдающемуся языковому пониманию и анализу.
Таким образом, FRED-T5-1.7B представляется нам наиболее подходящей языковой моделью
для задачи генерации определений.

В качестве материала, используемого для обучения модели, выступила русская версия Викисловаря.
Материал получен с помощью самостоятельно написанного скрипта на языке Python, позволяющего
извлечь данные из выгрузки Викисловаря в формат JSONL, где в каждом вхождении присутствовали
идентификатор статьи, лексема, про которую написана данная статья, а также определения
с примерами использования.

FRED-T5-1.7B была дообучена на полученном из Викисловаря материале, где на вход модель
принимает лексему и контекст, в которой она употреблялась, а на выход ожидается сгенерированное
определение.

Для оценки качества обучения модели используются метрики BLEU и ROUGE-L,
которые оценивают формальную схожесть текста: BLEU оценивает точность совпадений n-грамм
в сгенерированном тексте по сравнению с эталонным текстом, а ROUGE-L измеряет схожесть между
сгенерированным текстом и эталонным текстом на основе наибольшей общей последовательности слов.
Также использовалась метрика BERT-F1, которая учитывает семантику сравниваемых текстов, так как
использует модель BERT, представитель семейства моделей Transformer, обученные
на больших объемах текста и имеющие глубокое понимание семантики.
Использование нескольких метрик позволяет получить более полную картину качества модели,
поскольку каждая из них оценивает разные аспекты сгенерированного текста.
Как традиционные BLEU и ROUGE-L, так и более современный BERT-F1 активно используются в
задачах обработки естественного языка, в том числе в задачах генерации текста.
Так, в настоящей статье результаты данных метрик будут сравниваться с таковыми из статьи
Giulianelli M. et al., где сообщаются результаты трёх вышеперечисленных метрик при обучении модели
Т5 для задаче генерации определений на английском языке.

## Библиография

1. Giulianelli M., Luden I., Fernández R., Kutuzov A. Interpretable Word Sense Representations
via Definition Generation: The Case of Semantic Change Analysis. – 2023.
2. Shavrina T., Fenogenova A., Emelyanov A., Shevelev D., Artemova E., Malykh V., Mikhailov V.,
3. Tikhonova M., Chertok A., Evlampiev A. RussianSuperGLUE: A Russian Language Understanding
Evaluation Benchmark. – 2020.
3. Бенчмарк RussianSuperGLUE [Электронный ресурс] URL: https://russiansuperglue.com
(дата обращения: 12.10.2023)
4. Wang A., Pruksachatkun Y., Nangia N., Singh A., Michael J., Hill F., Levy O.,
Bowman S. SuperGLUE: A Stickier Benchmark for General-Purpose Language
Understanding Systems. – 2019.
5. Raffel C., Shazeer N., Roberts A., Lee K., Narang S., Matena M., Zhou Y.,
Li W., Liu P. Exploring the Limits of Transfer Learning with a Unified Text-to-Text Transformer.
– 2019.
6. Tay Y., Dehghani M., Tran V., Garcia X., Bahri D., Schuster T., Zheng H., Houlsby N.,
Metzler D. UL2: Unifying Language Learning Paradigms. – 2022.
7. FRED-T5. Новая SOTA модель для русского языка от SberDevices URL:
https://habr.com/ru/companies/sberdevices/articles/730088/ (дата обращения: 12.10.2023)
8. Zhang T., Kishore, V., Wu F., Weinberger K., Artzi Y. BERTScore: Evaluating Text Generation
with BERT. – 2019.
9. Rodina J., Trofimova Y., Kutuzov A., Artemova E. ELMo and BERT in Semantic Change Detection
for Russian. – 2021.
10. Rodina J., Kutuzov A. RuSemShift: a dataset of historical lexical semantic change in Russian
// Proceedings of the 28th International Conference on Computational Linguistics. – 2020.
– P. 1037-1047.
11. Kutuzov A., Fomin V., Mikhailov V., Rodina J. ShiftRy: Web service for diachronic analysis
of russian news // Computational Linguistics and Intellectual Technologies Papers from the
Annual International Conference «Dialogue». – 2020. – P. 500-516.
12. Fomin V., Bakshandaeva D., Rodina J., Kutuzov A. Tracing cultural diachronic semantic
shifts in Russian using word embeddings // Computational Linguistics and Intellectual
Technologies: Proceedings of the International Conference «Dialogue 2019». – 2019.
13. Сервис ShiftRy [Электронный ресурс] URL: https://shiftry.rusvectores.org
(дата обращения: 12.10.2023)
14. Schlechtweg D., McGillivray B., Hengchen S., Dubossarsky H., Tahmasebi N. SemEval-2020
Task 1: Unsupervised Lexical Semantic Change Detection. – 2020.
15. Kutuzov A., Giulianelli M. UiO-UvA at SemEval-2020 Task 1: Contextualised Embeddings for
Lexical Semantic Change Detection. – 2020.
16. Kutuzov A., Velldal E., Øvrelid L.. Contextualized language models for semantic change
detection: lessons learned. – 2022.
17. Gardner N., Khan H., Hung C-C. Definition modeling: Literature review and dataset analysis.
// Applied Computing and Intelligence, 2(1) 2022. – P. 83-98.
18. Noraset T., Liang C., Birnbaum L., Downey D. Definition modeling: Learning to define word
embeddings in natural language. // In Thirty-First AAAI Conference on Artificial Intelligence.
– 2017.
19. Gadetsky A., Yakubovskiy I., Vetrov D. Conditional generators of words definitions.
// In Proceedings of the 56th Annual Meeting of the Association for Computational Linguistics (Volume 2: Short Papers). – 2018. – P. 266–271.
20. N, K., Wang W. Y. Learning to explain non-standard English words and phrases.
// In Proceedings of the Eighth International Joint Conference on Natural Language Processing (Volume 2: Short Papers). – 2017. – P. 413–417
21. Mickus T., Paperno D., Constant M. Mark my word: A sequence-to-sequence approach to
definition modeling. // In Proceedings of the First NLPL Workshop on Deep Learning for Natural Language Processing. – 2019. – P. 1-11.
22. Сервис Викисловарь [Электронный ресурс] URL: https://ru.wiktionary.org/
(дата обращения: 12.10.2023)
23. Сервис RusVectores [Электронный ресурс] URL: https://github.com/akutuzov/webvectors
(дата обращения: 12.10.2023)
24. Сервис HuggingFace [Электронный ресурс] URL: https://huggingface.co
(дата обращения: 12.10.2023)
25. Сервис immers.cloud [Электронный ресурс] URL: https://immers.cloud
(дата обращения: 12.10.2023)
26. Сервис НКРЯ [Электронный ресурс] URL: https://ruscorpora.ru
(дата обращения: 12.10.2023)
27. Kutuzov A., Pivovarova L. Three-part diachronic semantic change dataset for Russian. – 2021.
28. Boleda G. Distributional Semantics and Linguistic Theory
// Annu. Rev. Linguist. – 2020. – P. 213-234.
29. Oklah S. Semantic Change. – 2014. – P. 11.
30. Petersen E., Potts C. Lexical Semantics with Large Language Models:
A Case Study of English “break” // In Findings of the Association for Computational Linguistics:
EACL 2023. – 2023. – P. 490–511.
