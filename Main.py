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
    df.corr()


def show_plots(df):
    labels = "Left", "Stayed"
    colors = ['red', 'blue']
    explode = (0, 0.1)
    plt.pie(df["left"].value_counts(), explode=explode, labels=labels, colors=colors,
            autopct='%1.1f%%', startangle=90)
    plt.axis('equal')
    # plt.savefig("pie1.png")
    # plt.show()

    labels = df["salary"].unique()
    colors = ['grey', 'magenta', "green"]
    explode = (0, 0, 0.1)
    plt.pie(df["salary"].value_counts(), explode=explode, labels=labels, colors=colors,
            autopct='%1.1f%%', startangle=90)
    plt.axis('equal')
    # plt.savefig("pie2.png")
    # plt.show()

    labels = df["sales"].unique()
    plt.pie(df["sales"].value_counts(), labels=labels, autopct='%1.1f%%', startangle=90)
    plt.axis('equal')
    # plt.savefig("pie3.png")
    # plt.show()
    l_sat = df.loc[df["left"] == 1]["satisfaction_level"]
    s_sat = df.loc[df["left"] == 0]["satisfaction_level"]
    l_tsc = df.loc[df["left"] == 1]["time_spend_company"]
    s_tsc = df.loc[df["left"] == 0]["time_spend_company"]
    plt.figure(figsize=(12, 4))
    plt.xlabel("satisfaction_level")
    plt.ylabel("time_spend_company")
    scat_s = plt.scatter(s_sat, s_tsc, color='#3979BC')
    scat_l = plt.scatter(l_sat, l_tsc, color="#CC2B2B")
    plt.legend((scat_s, scat_l),
               ("left = 0", "left = 1"),
               loc="upper right")
    # plt.savefig("scat1.png")
    # plt.show()
    #
    l_sat = df.loc[df["left"] == 1]["satisfaction_level"]
    s_sat = df.loc[df["left"] == 0]["satisfaction_level"]
    l_wa = df.loc[df["left"] == 1]["Work_accident"]
    s_wa = df.loc[df["left"] == 0]["Work_accident"]
    plt.figure(figsize=(15, 2))
    plt.xlabel("satisfaction_level")
    plt.ylabel("work_accident")
    plt.yticks([0, 1])
    scat_s = plt.scatter(s_sat, s_wa, color='#3979BC')
    scat_l = plt.scatter(l_sat, l_wa, color="#CC2B2B")
    plt.legend((scat_s, scat_l),
               ("left = 0", "left = 1"),
               loc="upper right")
    # plt.show()

    l_sat = df.loc[df["left"] == 1]["satisfaction_level"]
    s_sat = df.loc[df["left"] == 0]["satisfaction_level"]
    l_sal = df.loc[df["left"] == 1]["salary"]
    s_sal = df.loc[df["left"] == 0]["salary"]
    plt.figure(figsize=(15, 2))
    plt.xlabel("satisfaction_level")
    plt.ylabel("salary")
    plt.yticks([1, 2, 3])
    scat_s = plt.scatter(s_sat, s_sal, color='#3979BC')
    scat_l = plt.scatter(l_sat, l_sal, color="#CC2B2B")
    plt.legend((scat_s, scat_l),
               ("left = 0", "left = 1"),
               loc="upper right")
    # plt.show()

    l_sat = df.loc[df["left"] == 1]["satisfaction_level"]
    s_sat = df.loc[df["left"] == 0]["satisfaction_level"]
    l_ev = df.loc[df["left"] == 1]["last_evaluation"]
    s_ev = df.loc[df["left"] == 0]["last_evaluation"]
    plt.figure(figsize=(17, 10))
    plt.xlabel("satisfaction_level")
    plt.ylabel("last_evaluation")
    scat_s = plt.scatter(s_sat, s_ev, color='#3979BC')
    scat_l = plt.scatter(l_sat, l_ev, color="#CC2B2B")
    plt.legend((scat_s, scat_l),
               ("left = 0", "left = 1"),
               loc="upper right")
    # plt.show()


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

    df = df.replace(['sales', 'accounting', 'hr', 'technical', 'support', 'management', 'IT', 'product_mng',
                         'marketing', 'RandD'], [1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    df = df.replace(['low', 'medium', 'high'], [1, 2, 3])

    # show_info(df)
    # show_plots(df)

    # make_predictions(df)

    # tried to find out the best combination of features(with lowest error) for Naive Bayes
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





