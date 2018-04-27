

AI
机器学习 入门


线性回归


内容编译自这里：
https://www.toptal.com/machine-learning/machine-learning-theory-an-introductory-primer



示例数据：
训练：
https://www.rapidtables.com/convert/temperature/celsius-to-fahrenheit.html


参考：
https://www.metric-conversions.org/temperature/celsius-to-fahrenheit.htm
https://www.rapidtables.com/convert/temperature/fahrenheit-to-celsius.html



使用tensor flow做in action

tensor flow 来自这里：
https://github.com/aymericdamien/TensorFlow-Examples



train_X = numpy.asarray([-4.0,14.0,30.2, 32.0, 41.0 , 50.0, 68.0, 69.8, 86.0, 98.6, 104.0, 122.0 , 176.0])
train_Y = numpy.asarray([-20,-20, -1, 0, 5,10, 20, 21,30, 37, 40, 50, 80 ])

train_X = numpy.asarray([-4.0,14.0,30.2, 32.0, 41.0 , 50.0, 68.0, 69.8, 86.0, 98.6, 104.0, 122.0 , 176.0, 212.0])
train_Y = numpy.asarray([-20,-20, -1, 0, 5,10, 20, 21,30, 37, 40, 50, 80,100 ])


learning_rate = 0.001

```
Epoch: 0050 cost= 65.585853577 W= 0.384437 b= 1.17281
Epoch: 0100 cost= 64.173957825 W= 0.386289 b= 0.96773
Epoch: 0150 cost= 62.792522430 W= 0.388121 b= 0.764866
Epoch: 0200 cost= 61.440742493 W= 0.389933 b= 0.564196
Epoch: 0250 cost= 60.118137360 W= 0.391725 b= 0.3657
Epoch: 0300 cost= 58.824001312 W= 0.393498 b= 0.169351
Epoch: 0350 cost= 57.557708740 W= 0.395252 b= -0.0248725
Epoch: 0400 cost= 56.318645477 W= 0.396987 b= -0.216994
Epoch: 0450 cost= 55.106288910 W= 0.398703 b= -0.407037
Epoch: 0500 cost= 53.920036316 W= 0.4004 b= -0.595023
Epoch: 0550 cost= 52.759281158 W= 0.402079 b= -0.780975
Epoch: 0600 cost= 51.623577118 W= 0.40374 b= -0.964914
Epoch: 0650 cost= 50.512268066 W= 0.405383 b= -1.14686
Epoch: 0700 cost= 49.424903870 W= 0.407008 b= -1.32684
Epoch: 0750 cost= 48.360912323 W= 0.408616 b= -1.50487
Epoch: 0800 cost= 47.319885254 W= 0.410206 b= -1.68098
Epoch: 0850 cost= 46.301242828 W= 0.411779 b= -1.85518
Epoch: 0900 cost= 45.304496765 W= 0.413335 b= -2.02749
Epoch: 0950 cost= 44.329242706 W= 0.414874 b= -2.19794
Epoch: 1000 cost= 43.374973297 W= 0.416396 b= -2.36655
Optimization Finished!
Training cost= 43.375 W= 0.416396 b= -2.36655

```


learning_rate = 0.01


```
Epoch: 0050 cost= nan W= nan b= nan
Epoch: 0100 cost= nan W= nan b= nan
Epoch: 0150 cost= nan W= nan b= nan
Epoch: 0200 cost= nan W= nan b= nan
Epoch: 0250 cost= nan W= nan b= nan
Epoch: 0300 cost= nan W= nan b= nan
Epoch: 0350 cost= nan W= nan b= nan
Epoch: 0400 cost= nan W= nan b= nan
Epoch: 0450 cost= nan W= nan b= nan
Epoch: 0500 cost= nan W= nan b= nan
Epoch: 0550 cost= nan W= nan b= nan
Epoch: 0600 cost= nan W= nan b= nan
Epoch: 0650 cost= nan W= nan b= nan
Epoch: 0700 cost= nan W= nan b= nan
Epoch: 0750 cost= nan W= nan b= nan
Epoch: 0800 cost= nan W= nan b= nan
Epoch: 0850 cost= nan W= nan b= nan
Epoch: 0900 cost= nan W= nan b= nan
Epoch: 0950 cost= nan W= nan b= nan
Epoch: 1000 cost= nan W= nan b= nan
Optimization Finished!
Training cost= nan W= nan b= nan
```


learning_rate = 0.0001
```
Epoch: 0050 cost= 56.438053131 W= 0.436205 b= -0.413851
Epoch: 0100 cost= 56.232036591 W= 0.436423 b= -0.445571
Epoch: 0150 cost= 56.026775360 W= 0.436641 b= -0.477234
Epoch: 0200 cost= 55.822273254 W= 0.436858 b= -0.508837
Epoch: 0250 cost= 55.618503571 W= 0.437075 b= -0.540383
Epoch: 0300 cost= 55.415496826 W= 0.437291 b= -0.571872
Epoch: 0350 cost= 55.213203430 W= 0.437507 b= -0.603304
Epoch: 0400 cost= 55.011661530 W= 0.437723 b= -0.634679
Epoch: 0450 cost= 54.810874939 W= 0.437938 b= -0.665995
Epoch: 0500 cost= 54.610797882 W= 0.438153 b= -0.697255
Epoch: 0550 cost= 54.411460876 W= 0.438368 b= -0.728456
Epoch: 0600 cost= 54.212837219 W= 0.438582 b= -0.759601
Epoch: 0650 cost= 54.014957428 W= 0.438795 b= -0.79069
Epoch: 0700 cost= 53.817771912 W= 0.439009 b= -0.821723
Epoch: 0750 cost= 53.621330261 W= 0.439222 b= -0.852697
Epoch: 0800 cost= 53.425609589 W= 0.439434 b= -0.883615
Epoch: 0850 cost= 53.230598450 W= 0.439646 b= -0.914477
Epoch: 0900 cost= 53.036304474 W= 0.439858 b= -0.945281
Epoch: 0950 cost= 52.842700958 W= 0.440069 b= -0.976033
Epoch: 1000 cost= 52.649806976 W= 0.44028 b= -1.00673
Optimization Finished!
Training cost= 52.6498 W= 0.44028 b= -1.00673

```



# Parameters
learning_rate = 0.001
training_epochs = 10000
display_step = 500

```
Epoch: 0500 cost= 35.235664368 W= 0.430131 b= -3.88756
Epoch: 1000 cost= 28.344736099 W= 0.443062 b= -5.31962
Epoch: 1500 cost= 22.801347733 W= 0.45466 b= -6.60405
Epoch: 2000 cost= 18.342121124 W= 0.465062 b= -7.75606
Epoch: 2500 cost= 14.754896164 W= 0.474392 b= -8.78931
Epoch: 3000 cost= 11.869299889 W= 0.48276 b= -9.71601
Epoch: 3500 cost= 9.548123360 W= 0.490265 b= -10.5471
Epoch: 4000 cost= 7.680812359 W= 0.496996 b= -11.2926
Epoch: 4500 cost= 6.178786755 W= 0.503033 b= -11.9612
Epoch: 5000 cost= 4.970383167 W= 0.508448 b= -12.5609
Epoch: 5500 cost= 3.998302698 W= 0.513305 b= -13.0987
Epoch: 6000 cost= 3.216356516 W= 0.517661 b= -13.5812
Epoch: 6500 cost= 2.587427855 W= 0.521568 b= -14.0138
Epoch: 7000 cost= 2.081362963 W= 0.525072 b= -14.4019
Epoch: 7500 cost= 1.674250603 W= 0.528215 b= -14.75
Epoch: 8000 cost= 1.346877456 W= 0.531034 b= -15.0621
Epoch: 8500 cost= 1.083403945 W= 0.533562 b= -15.3421
Epoch: 9000 cost= 0.871582687 W= 0.535829 b= -15.5932
Epoch: 9500 cost= 0.701112330 W= 0.537863 b= -15.8184
Epoch: 10000 cost= 0.564054787 W= 0.539687 b= -16.0203
Optimization Finished!
Training cost= 0.564055 W= 0.539687 b= -16.0203
```


http://58.241.217.181:15111/notebooks/work/github/TensorFlow-Examples/notebooks/2_BasicModels/linear_regression.ipynb


```
http://58.241.217.181:15111/notebooks/work/github/TensorFlow-Examples/notebooks/2_BasicModels/linear_regression.ipynb


```

->

Reducing Loss: An Iterative Approach


https://developers.google.com/machine-learning/crash-course/reducing-loss/an-iterative-approach

https://www.howcast.com/videos/258352-how-to-play-hot-and-cold/
