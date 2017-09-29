# Replicate-and-Review-of-Winner-Solutions-for-O2O-Coupon-Usage-Forecast-of-Alibaba   
Review of the winner solutions for machine learning challenge - 'O2O Coupon Usage Forecast' (O2O优惠券使用预测)
page of the competition:   
https://tianchi.shuju.aliyun.com/competition/introduction.htm?spm=5176.100065.200879.2.6r6s4g&raceId=231587&_lang=en_US   
page of winner solution:   
https://github.com/wepe/O2O-Coupon-Usage-Forecast   
I have put the data of season 1 in my folder: data_season_1.zip. In case it is not available, check out: https://pan.baidu.com/s/1nvFG2ff 


In the last quarter of 2016, Tianchi of Alibaba Cloud, one of the most popular machine learning challenge platforms in China, hosted the competition 'O2O Coupon Usage Forecast' (O2O优惠券使用预测). The task of the competition is to predict if the user is going to use the coupons in hand. 

This game received attentions from thousands of players and eventually a team that consists of students from Peking University and University of Science and Technology of China won the first place. Their strategy is highlighted with feature engineering. Compared with the given information, new features that were extracted from user, shop, coupon, shopping date, distance between user and shop and other perspectives describe the underlying problem more effectively.

Considering the data set of high-quality and the outstanding problem-solving skills of the solution, it is in favor to promote them to people who have limited Chinese skills. In this sense, this article : firstly elaborate the competition in details. In particular some background on the rising O2O industry in China and the necessity of accurate predictions of coupon usage are provided. Secondly, some related work in this topic is discussed. Thirdly, the strategy of the winner solution is presented, especially each constructed feature is explained. Lastly, an XGBoost model fitting is examined so that the work of feature construction could be justified by feature importance of XGBoost.



