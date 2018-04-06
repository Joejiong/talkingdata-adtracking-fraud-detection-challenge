#####
# Source modified from: http://blog.datadive.net/selecting-good-features-part-iv-stability-selection-rfe-and-everything-side-by-side/
#####


import numpy as np
import pandas as pd
import itertools

import warnings
warnings.filterwarnings("ignore")
import os
os.environ["OMP_NUM_THREADS"] = "4"
import gc

class Feature:

    #####
    # Constants
    #####

    # Initialize instance
    def __init__(self):
        # Initialize
        print("Initializing")


    # Describe fields
    def describe(self):
        print("Describing")


    def rank_to_dict(self, ranks, names, order=1):
        # from sklearn.preprocessing import MinMaxScaler
        # minmax = MinMaxScaler()
        # ranks = minmax.fit_transform(order*np.array([ranks]).T).T[0]
        ranks = map(lambda x: round(x, 6), ranks)
        return dict(zip(names, ranks ))


    # Ridge analysis
    def ridge(self, X, y):
        print("Performing ridge analysis")

        from sklearn.linear_model import Ridge
        ridge = Ridge(alpha=7)
        ridge.fit(X, y)

        scores = np.absolute(ridge.coef_) / np.absolute(ridge.coef_).sum()
        ranks = self.rank_to_dict(np.abs(scores), X.columns.values)
        return ranks


    # Lasso analysis
    def lasso(self, X, y):
        print("Performing lasso analysis")

        from sklearn.linear_model import Lasso
        lasso = Lasso(alpha=.05)
        lasso.fit(X, y)

        scores = np.absolute(lasso.coef_) / np.absolute(lasso.coef_).sum()
        ranks = self.rank_to_dict(np.abs(scores), X.columns.values)
        return ranks


    # Stability analysis
    def stability(self, X, y):
        print("Performing stability (rlasso) analysis")

        from sklearn.linear_model import RandomizedLasso
        rlasso = RandomizedLasso(alpha=0.04)
        rlasso.fit(X, y)

        scores = np.absolute(rlasso.scores_) / np.absolute(rlasso.scores_).sum()
        ranks = self.rank_to_dict(np.abs(scores), X.columns.values)
        return ranks


    # Mine analysis
    # NOTE: takes a LONG time...
    def mine(self, X, y):
        print("Performing mine analysis")
        from minepy import MINE
        mine = MINE()
        mic_scores = []
        for i in range(X.shape[1]):
            if i%10 == 0:
                print("  processing: ", i)
            name = X.columns.values[i]
            xvalues = np.array(X[name])
            yvalues = np.array(y)
            mine.compute_score(xvalues, yvalues)
            m = mine.mic()
            mic_scores.append(m)

        mic_scores = np.array(mic_scores)
        scores = np.absolute(mic_scores) / np.absolute(mic_scores).sum()
        ranks = self.rank_to_dict(np.abs(scores), X.columns.values)
        return ranks

    # Principal component analysis
    def pca(self, X, y):
        print("Performing principal component analysis")

        from sklearn.decomposition import PCA
        # pca = PCA(n_components=3)
        pca = PCA()
        fit = pca.fit(X)

        scores = np.absolute(fit.explained_variance_ratio_) / np.absolute(fit.explained_variance_ratio_).sum()
        ranks = self.rank_to_dict(np.abs(scores), X.columns.values)
        return ranks


    def regression(self, X, y):
        print("Performing linear regression analysis")

        from sklearn.linear_model import LinearRegression
        lr = LinearRegression(normalize=True)
        lr.fit(X, y)

        scores = np.absolute(lr.coef_) / np.absolute(lr.coef_).sum()
        ranks = self.rank_to_dict(np.abs(scores), X.columns.values)
        return ranks


    # Recursive feature elimination
    # NOTE: takes a LONG time...
    def recursive(self, X, y):
        print("Performing recursive feature elimination")

        # feature extraction
        from sklearn.feature_selection import RFE
        from sklearn.linear_model import LogisticRegression
        model = LogisticRegression()
        rfe = RFE(model)
        fit = rfe.fit(X, y)

        print("Num Features: ", fit.n_features_)
        print("Selected Features: ", fit.support_)
        print("Feature Ranking: ", fit.ranking_)

    # Univariate statistical analysis
    def univariate(self, X, y):
        print("Performing univariate statistical analysis")

        from sklearn.feature_selection import SelectKBest
        from sklearn.feature_selection import chi2

        # feature extraction
        test = SelectKBest(score_func=chi2, k=4)
        fit = test.fit(X, y)

        # Normalize to sum to 1.0
        # scores = fit.scores_ / fit.scores_.sum()
        #
        # fi = {}
        # for i, importance in enumerate(scores):
        #     columnname = X.columns.values[i]
        #     fi[columnname] = importance

        scores = np.absolute(fit.scores_) / np.absolute(fit.scores_).sum()
        ranks = self.rank_to_dict(np.abs(scores), X.columns.values)
        return ranks

        return fi

    # Determine feature importance
    def importance(self, X, y):
        print("Performing feature importance analysis")

        from sklearn.ensemble import ExtraTreesClassifier
        clf = ExtraTreesClassifier()
        clf = clf.fit(X, y)

        # Normalize to sum to 1.0
        # scores = clf.feature_importances_ / clf.feature_importances_.sum()
        #
        # fi = {}
        # for i, importance in enumerate(scores):
        #     columnname = X.columns.values[i]
        #     fi[columnname] = importance
        #
        # return fi

        scores = np.absolute(clf.feature_importances_) / np.absolute(clf.feature_importances_).sum()
        ranks = self.rank_to_dict(np.abs(scores), X.columns.values)
        return ranks



    # Create useful combinations (powerset) of column names
    def powerset(self, iterable):
        # powerset([1,2,3]) --> () (1,) (2,) (3,) (1,2) (1,3) (2,3) (1,2,3)
        s = list(iterable)
        return itertools.chain.from_iterable(itertools.combinations(s, r) for r in range(len(s)+1))


    # Assemble data table field combinations
    def assemble(self, infilename):
        print("Assembling data")

        df = pd.read_csv(infilename)

        print("Adding hour, day, day of week")
        # df["minute"] = pd.to_datetime(df.click_time).dt.minute.astype("uint8")
        df["hour"] = pd.to_datetime(df.click_time).dt.hour.astype("uint8")
        df["day"] = pd.to_datetime(df.click_time).dt.day.astype("uint8")
        df["wday"]  = pd.to_datetime(df.click_time).dt.dayofweek.astype("uint8")
        df = df.drop(["click_time", "attributed_time"], 1)

        newcolumns = []
        columns = df.columns.values
        xtuples = self.powerset(columns)
        for xtuple in xtuples:
            if not xtuple:
                continue
            if len(xtuple) == 1:
                continue
            if len(xtuple) > 4:
                continue
            if "is_attributed" in xtuple:
                continue

            newcolumns.append(xtuple)

        # remove items composed only of date elements
        # xremoves = self.powerset(["minute", "hour", "day", "wday"])
        xremoves = self.powerset(["hour", "day", "wday"])
        for xremove in xremoves:
            if xremove in newcolumns:
                print("removing: ", xremove)
                newcolumns.remove(xremove)

        print("num newcolumns: ", len(newcolumns))
        print("newcolumns: ", newcolumns)

        for newcolumn in newcolumns:
            print("Grouping by: ", newcolumn)
            columnlist = list(newcolumn)
            key = columnlist[0]
            fields = columnlist[1:]
            newname = "_".join(columnlist)
            print("\n---- Creating new column: ", newname)
            print("columnlist: ", columnlist)
            print("key: ", key)
            print("fields: ", fields)

            gp = df[columnlist].groupby(by=fields)[[key]].count().reset_index().rename(index=str, columns={key: newname})
            df = df.merge(gp, on=fields, how="left")
            del gp; gc.collect()


        y = df["is_attributed"]
        X = df.drop(["is_attributed"], 1)
        print("X: ", X.shape)
        print("y: ", y.shape)

        return X, y
