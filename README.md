# RuAuthorRnnCuda
# Наилучший результат обучения! 

![alt text](https://github.com/YuioiuY/RuAuthorRnnCuda/blob/main/Acc.png)

В качетве текста был выбран Максим Горький. Есть мнение, что Горький был преемником и продолжателем творчества Пушкина, не в формальном отношении, а по существу и по духу.


# Ресурсы 
https://storage.yandexcloud.net/natasha-navec/packs/navec_hudlit_v1_12B_500K_300d_100q.tar
https://storage.yandexcloud.net/academy.ai/russian_literature.zip

#Справка по работе с GPU 

Для работы понадобится: 

- tensorflow == 2.10
- [CUDA Version](https://developer.nvidia.com/cuda-11-8-0-download-archive?target_os=Windows&target_arch=x86_64&target_version=11&target_type=exe_network): 11.8 
- [Python 3.10.13](https://www.python.org/downloads/release/python-31013/)

# Комманды для проверки

nvidia-smi

# Установка CUDA и cuDNN
Если драйвер есть, но TensorFlow всё ещё не видит видеокарту, нужно установить CUDA Toolkit и cuDNN.

Установка CUDA:
Скачайте CUDA Toolkit 11.8 (нужна версия 11.2+ для TensorFlow 2.10+).

Установите с дефолтными настройками.

Установка cuDNN:
Зарегистрируйтесь на NVIDIA Developer.

Скачайте cuDNN 8.6 для CUDA 11.8.

Разархивируйте файлы и поместите их в папку C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.8\.

#