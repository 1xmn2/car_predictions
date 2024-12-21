import inline
import matplotlib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from xgboost import XGBRegressor
from sklearn.model_selection import cross_val_score, GridSearchCV
from sklearn.impute import SimpleImputer
pd.options.display.max_columns = None
warnings.filterwarnings('ignore')
# %matplotlib inline
# 数据的读取和初步处理
df_train = pd.read_csv('D:/PythonProject/car_predictions/used_car_train_20200313.csv', sep=' ')
df_test = pd.read_csv('D:/PythonProject/car_predictions/used_car_testB_20200421.csv', sep=' ')
train = df_train.drop(['SaleID'], axis=1)
test = df_test.drop(['SaleID'], axis=1)
train.head()
test.head()
# 查看总览 - 训练集
train.info()
# 查看总览 - 测试集
test.info()
# 转换'-'
train['notRepairedDamage'] = train['notRepairedDamage'].replace('-', np.nan)
test['notRepairedDamage'] = test['notRepairedDamage'].replace('-', np.nan)

# 转换数据类型
train['notRepairedDamage'] = train['notRepairedDamage'].astype('float64')
test['notRepairedDamage'] = test['notRepairedDamage'].astype('float64')

# 检查是否转换成功
train['notRepairedDamage'].unique(), test['notRepairedDamage'].unique()
# 查看数值统计描述 - 测试集
test.describe()
# 查看数值统计描述 - 训练集
train.describe()
train.drop(['seller'], axis=1, inplace=True)
test.drop(['seller'], axis=1, inplace=True)
train = train.drop(['offerType'], axis=1)
test = test.drop(['offerType'], axis=1)
train.shape, test.shape

# 有143个值不合法，需要用别的值替换
train[train['power'] > 600]['power'].count()
test[test['power'] > 600]['power'].count()
# 查看各特征与销售价格之间的线性相关系数
train.corr().unstack()['price'].sort_values(ascending=False)
train.drop(['v_2', 'v_6', 'v_1', 'v_14', 'v_13', 'v_7', 'name', 'creatDate'], axis=1, inplace=True)
test.drop(['v_2', 'v_6', 'v_1', 'v_14', 'v_13', 'v_7', 'name', 'creatDate'], axis=1, inplace=True)
train.shape, test.shape
# 再次查看各特征与销售价格之间的线性相关系数
train.corr().unstack()['price'].sort_values(ascending=False)
# 使用map函数，以power列的中位数来替换数值超出范围的power
train['power'] = train['power'].map(lambda x: train['power'].median() if x > 600 else x)
test['power'] = test['power'].map(lambda x: test['power'].median() if x > 600 else x)
# 检查是否替换成功
train['power'].plot.hist()
test['power'].plot.hist()
# 查看训练集缺失值存在情况
train.isnull().sum()[train.isnull().sum() > 0]
# 查看测试集缺失值存在情况
test.isnull().sum()[test.isnull().sum() > 0]
train[train['model'].isnull()]
# model(车型编码)一般与brand, bodyType, gearbox, power有关，选择以上4个特征与该车相同的车辆的model，选择出现次数最多的值
train[(train['brand'] == 37) &
      (train['bodyType'] == 6.0) &
      (train['gearbox'] == 1.0) &
      (train['power'] == 190)]['model'].value_counts()
# 用157.0填充缺失值
train.loc[38424, 'model'] = 157.0
train.loc[38424, :]
# 查看填充结果
train.info()
# 看缺失值数量
print(train['bodyType'].isnull().value_counts())
print('\n')
print(test['bodyType'].isnull().value_counts())
# 可见不同车身类型的汽车售价差别还是比较大的，故保留该特征，填充缺失值
# 看看车身类型数量分布
print(train['bodyType'].value_counts())
print('\n')
print(test['bodyType'].value_counts())
# 在两个数据集上，车身类型为0.0（豪华轿车）的汽车数量都是最多，所以用0.0来填充缺失值
train.loc[:, 'bodyType'] = train['bodyType'].map(lambda x: 0.0 if pd.isnull(x) else x)
test.loc[:, 'bodyType'] = test['bodyType'].map(lambda x: 0.0 if pd.isnull(x) else x)
# 看缺失值数量
print(train['fuelType'].isnull().value_counts())
print('\n')
print(test['fuelType'].isnull().value_counts())
# 猜想：燃油类型与车身类型相关，如豪华轿车更可能是汽油或电动， 而搅拌车大多是柴油
# 创建字典，保存不同bodyType下， fuelType的众数，并以此填充fuelTyp的缺失值
dict_enu_train, dict_enu_test = {}, {}
for i in [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0]:
    dict_enu_train[i] = train[train['bodyType'] == i]['fuelType'].mode()[0]
    dict_enu_test[i] = test[test['bodyType'] == i]['fuelType'].mode()[0]

# 发现dict_enu_train, dict_enu_test是一样的内容
# 开始填充fuelType缺失值
# 在含fuelType缺失值的条目中，将不同bodyType对应的index输出保存到一个字典中
dict_index_train, dict_index_test = {}, {}

for bodytype in [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0]:
    # 初始化为 []，即使没有对应的index
    dict_index_train[bodytype] = train[(train['bodyType'] == bodytype) & (train['fuelType'].isnull())].index.tolist()
    dict_index_test[bodytype] = test[(test['bodyType'] == bodytype) & (test['fuelType'].isnull())].index.tolist()
    if bodytype not in dict_index_train:
        dict_index_train[bodytype] = []
    if bodytype not in dict_index_test:
        dict_index_test[bodytype] = []
    print(dict_index_train)
    # 分别对每个bodyTYpe所对应的index来填充fuelType列
    for bt, ft in dict_enu_train.items():
        #     train.loc[tuple(dict_index[bt]), :]['fuelType'] = ft  # 注意：链式索引 (chained indexing)很可能导致赋值失败！
        train.loc[dict_index_train[bt], 'fuelType'] = ft  # Pandas推荐使用这种方法来索引/赋值
        test.loc[dict_index_test[bt], 'fuelType'] = ft
        # 看缺失值数量
        print(train['gearbox'].isnull().value_counts())
        print('\n')
        print(test['gearbox'].isnull().value_counts())
        # 可见变速箱类型的不同不会显著影响售价，删去测试集中带缺失值的行或许是可行的做法，但为避免样本量减少带来的过拟合，还是决定保留该特征并填充其缺失值
        # 看看车身类型数量分布
        print(train['gearbox'].value_counts())
        print('\n')
        print(test['gearbox'].value_counts())
        # 训练集
        train.loc[:, 'gearbox'] = train['gearbox'].map(lambda x: 0.0 if pd.isnull(x) else x)
        # # 对于测试集，为保证预测结果完整性，不能删去任何行。测试集仅有1910个gearbox缺失值，用数量占绝大多数的0.0（手动档）来填充缺失值
        test.loc[:, 'gearbox'] = test['gearbox'].map(lambda x: 0.0 if pd.isnull(x) else x)
        # 检查填充是否成功
        train.info()
        test.info()
        # 看缺失值数量
        # 缺失值数量在两个数据集中的占比都不低
        print(train['notRepairedDamage'].isnull().value_counts())
        print('\n')
        print(test['notRepairedDamage'].isnull().value_counts())
        # 查看数量分布
        print(train['notRepairedDamage'].value_counts())
        print('\n')
        print(test['notRepairedDamage'].value_counts())
        # 查看线性相关系数
        train[['notRepairedDamage', 'price']].corr()['price']
        # 很奇怪，在整个训练集上有尚未修复损坏的汽车比损坏已修复的汽车售价还要高。考虑到剩余接近20个特征的存在，这应该是巧合
        # 为简单化问题，仍使用数量占比最大的0.0来填充所有缺失值
        train.loc[:, 'notRepairedDamage'] = train['notRepairedDamage'].map(lambda x: 0.0 if pd.isnull(x) else x)
        test.loc[:, 'notRepairedDamage'] = test['notRepairedDamage'].map(lambda x: 0.0 if pd.isnull(x) else x)
        # 最后。检查填充结果
        train.info()
        test.info()
        rf = RandomForestRegressor(n_estimators=100, max_depth=8, random_state=1)
        xgb = XGBRegressor(n_estimators=150, max_depth=8, learning_rate=0.1, random_state=1)
        gbdt = GradientBoostingRegressor(subsample=0.8, random_state=1)  # subsample小于1可降低方差，但会加大偏差
        # 定义一个SimpleImputer填充缺失值
        imputer = SimpleImputer(strategy='median')  # 使用中位数填充，也可以选择'mean'、'most_frequent'等

        # 对训练集和测试集分别进行填充
        X = train.drop(['price'], axis=1)
        y = train['price']
        X = imputer.fit_transform(X)  # 对训练集进行填充
        test_imputed = imputer.transform(test)  # 对测试集进行填充

        # 检查填充结果
        print("训练集填充后的形状：", X.shape)
        print("测试集填充后的形状：", test_imputed.shape)
        # 随机森林
        score_rf = -1 * cross_val_score(rf, X, y, scoring='neg_mean_absolute_error', cv=5).mean()
        print('随机森林模型的平均MAE为：', score_rf)

        # XGBoost
        score_xgb = -1 * cross_val_score(xgb, X, y, scoring='neg_mean_absolute_error', cv=5).mean()
        print('XGBoost模型的平均MAE为：', score_xgb)

        # 梯度提升树 GBDT
        score_gbdt = -1 * cross_val_score(gbdt, X, y, scoring='neg_mean_absolute_error', cv=5).mean()
        print('梯度提升树模型的平均MAE为：', score_gbdt)

        # 网格搜索优化 XGBoost 参数
        params = {'n_estimators': [150, 200, 250],
                  'learning_rate': [0.1],
                  'subsample': [0.5, 0.8]}

        model = GridSearchCV(estimator=xgb, param_grid=params, scoring='neg_mean_absolute_error', cv=3)
        model.fit(X, y)

        # 输出最佳参数和预测结果
        print('最佳参数为：\n', model.best_params_)
        print('最佳分数为：\n', model.best_score_)
        print('最佳模型为：\n', model.best_estimator_)

        # 预测并保存结果
        predictions = model.predict(test_imputed)
        result = pd.DataFrame({'SaleID': df_test['SaleID'], 'price': predictions})
        result.to_csv('D:/PythonProject/car_predictions/My_submission.csv', index=False)