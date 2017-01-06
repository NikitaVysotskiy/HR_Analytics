import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier

# from itertools import combinations as combs


def show_info(df):
    df.head()
    df.info()
    df.describe()
    df["sales"].unique()
    df["salary"].unique()
    df["sales"].unique()
    df["salary"].unique()
    df.corr()


def show_plots(df):
    labels = "Left", "Stayed"
    left_size = df.loc[df["left"] == 1].shape[0]
    print("Number of employees, who had left: {0}".format(left_size))
    print("Number of employees, who stayed: {0}".format(df.shape[0] - left_size))
    sizes = [left_size, df.shape[0] - left_size]
    colors = ['red', 'blue']
    explode = (0, 0.1)
    plt.pie(sizes, explode=explode, labels=labels, colors=colors,
            autopct='%1.1f%%', startangle=90)
    plt.axis('equal')
    plt.show()

    labels = df["salary"].unique()
    salary_distribution = []
    for salary in labels:
        size = df.loc[df["salary"] == salary].shape[0]
        print("Number of employees with {0} salary: {1}".format(salary, size))
        salary_distribution.append(size)

    colors = ['grey', 'magenta', "green"]
    explode = (0, 0, 0.1)
    plt.pie(salary_distribution, explode=explode, labels=labels, colors=colors,
            autopct='%1.1f%%', startangle=90)
    plt.axis('equal')
    plt.show()

    labels = df["sales"].unique()
    sales_distribution = []
    explode = []
    for sales in labels:
        size = df.loc[df["sales"] == sales].shape[0]
        print("Number of employees in {0}: {1}".format(sales, size))
        sales_distribution.append(size)
        explode.append(0.1) if size < 1000 else explode.append(0)

    plt.pie(sales_distribution, explode=explode, labels=labels, autopct='%1.1f%%', startangle=90)
    plt.axis('equal')

    plt.show()


def make_predictions(df):
    train, test = train_test_split(df, test_size=0.3333).copy()
    # print("Train length: {0}".format(train.shape[0]))
    train_target = train["left"]
    test_target = test["left"]
    train = train.drop('left', axis=1)
    test = test.drop('left', axis=1)

    gnb = GaussianNB()
    gnb.fit(train, train_target)
    y_gnb = gnb.predict(test)

    lr = LogisticRegression()
    lr.fit(train, train_target)
    y_lr = lr.predict(test)

    rforest = RandomForestClassifier()
    rforest.fit(train, train_target)
    y_rforest = rforest.predict(test)

    print("Gaussian Naive Bayes")
    print("Number of mislabeled points out of a total %d points : %d"
          % (test.shape[0], (test_target != y_gnb).sum()))

    print("\nLogistic regression")
    print("Number of mislabeled points out of a total %d points : %d"
          % (test.shape[0], (test_target != y_lr).sum()))

    print("\nRandom Forest")
    print("Number of mislabeled points out of a total %d points : %d"
          % (test.shape[0], (test_target != y_rforest).sum()))


if __name__ == '__main__':
    df = pd.read_csv("HR.csv")

    #show_info(df)
    #show_plots(df)

    df = df.replace(['sales', 'accounting', 'hr', 'technical', 'support', 'management', 'IT', 'product_mng',
                         'marketing', 'RandD'], [1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    df = df.replace(['low', 'medium', 'high'], [1, 2, 3])

    make_predictions(df)

    # tried to find out the best combination of features(with lowest error)
    # ---------------------------------------------------------
    # ind = np.array([
    #     "satisfaction_level",
    #     "average_montly_hours",
    #     "last_evaluation",
    #     "time_spend_company",
    #     "number_project",
    #     "Work_accident",
    #     "sales",
    #     "salary",
    #     "promotion_last_5years"
    # ])
    # a = range(9)
    # errors = []
    # for i in range(1, 9):
    #     for j in combs(a, i):
    #         model = train_nb(train[ind[list(j)]], train["left"])
    #         y_pred = predict_nb(model, test[ind[list(j)]])
    #         errors.append((test["left"] != y_pred).sum())
    #         if len(errors) == 464:
    #             print(j)
    # ---------------------------------------------------------
    # print("Features, used to get the least error:\n", ind[[2, 4, 5, 6, 7, 8]])
    # print(np.array(errors).min())
    # print(np.array(errors).max())
    # print(np.array(errors).mean())